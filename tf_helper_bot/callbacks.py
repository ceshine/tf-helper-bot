import socket
from datetime import datetime, timedelta
from time import time
from collections import deque, defaultdict
from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path

import numpy as np
import tensorflow as tf
try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False

from .bot import BaseBot

__all__ = [
    "Callback", "MovingAverageStatsTrackerCallback",
    "CheckpointCallback"
]


class Callback:
    def on_batch_inputs(self, bot: BaseBot, input_tensors: tf.Tensor, targets: tf.Tensor):
        return input_tensors, targets

    def on_train_starts(self, bot: BaseBot):
        return

    def on_train_ends(self, bot: BaseBot):
        return

    def on_epoch_ends(self, bot: BaseBot, epoch: int):
        return

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        return

    def on_step_ends(self, bot: BaseBot, train_loss: float, train_weight: int):
        return

    def on_load_checkpoint(self, **kwargs):
        return

    def on_save_checkpoint(self):
        return

    def reset(self):
        return


class MovingAverageStatsTrackerCallback(Callback):
    """Log moving average of training losses, and report evaluation metrics.
    """

    def __init__(self, avg_window: int, log_interval: int):
        super().__init__()
        self.avg_window = avg_window
        self.log_interval = log_interval
        self.reset()
        self.timer: float = 0.0

    def on_train_starts(self, bot: BaseBot):
        self.timer = time()

    def on_step_ends(self, bot: BaseBot, train_loss, train_weight):
        self.train_losses.append(train_loss)
        self.train_weights.append(train_weight)
        if bot.step % self.log_interval == 0:
            # print(len(self.train_weights), len(self.train_losses))
            train_loss_avg = np.average(
                self.train_losses, weights=self.train_weights, axis=0)
            lr = (
                bot.optimizer.lr(bot.step) if callable(bot.optimizer.lr)
                else bot.optimizer.lr
            )
            if not isinstance(lr, float):
                lr = lr.numpy()
            speed = (time() - self.timer) / self.log_interval
            # reset timer
            self.timer = time()
            bot.logger.info(
                f"Step %5d | loss {bot.loss_format} | lr %.2e | %.3fs per step",
                bot.step, train_loss_avg, lr, speed)
            self.train_logs.append(train_loss_avg)

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        self.metrics["step"].append(bot.step)
        history_length = len(self.metrics["step"])
        bot.logger.info(f"Metrics at step {bot.step}:")
        for metric_name, (metric_value, metric_string) in metrics.items():
            self.metrics[metric_name].append((metric_value, metric_string))
            assert history_length == len(
                self.metrics[metric_name]), "Inconsistent metric found!"
            bot.logger.info(f"{metric_name}: {metric_string}")

    def on_train_ends(self, bot: BaseBot):
        if self.metrics["step"]:
            bot.logger.info("Training finished. Best step(s):")
            for metric_name, metric_values in self.metrics.items():
                if metric_name == "step":
                    continue
                best_idx = np.argmin(
                    np.array([x[0] for x in metric_values]))
                bot.logger.info(
                    "%s: %s @ step %d",
                    metric_name, metric_values[best_idx][1],
                    self.metrics["step"][best_idx]
                )

    def reset(self):
        self.train_losses = deque(maxlen=self.avg_window)
        self.train_weights = deque(maxlen=self.avg_window)
        self.metrics = defaultdict(list)
        self.train_logs = []


class CheckpointCallback(Callback):
    """Save and manage checkpoints.

    TODO: Checkpoints that can be used to resume training
    """

    def __init__(
            self, keep_n_checkpoints: int = 1,
            checkpoint_dir: Union[Path, str] = Path("./data/cache/model_cache/"),
            monitor_metric: str = "loss"):
        super().__init__()
        assert keep_n_checkpoints > 0
        self.keep_n_checkpoints = keep_n_checkpoints
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor_metric = monitor_metric
        self.best_performers: List[Tuple[float, Path, int]] = []
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def on_eval_ends(self, bot: BaseBot, metrics: Dict[str, Tuple[float, str]]):
        target_value, target_string = metrics[self.monitor_metric]
        target_path = (
            self.checkpoint_dir /
            "ckpt_{}_{}_{}_{}.h5".format(
                bot.name, target_string, bot.step,
                datetime.now().strftime("%m%d%H%M"))
        )
        bot.logger.debug("Saving checkpoint %s...", target_path)
        if (
            len(self.best_performers) < self.keep_n_checkpoints or
            target_value < self.best_performers[-1][0]
        ):
            self.best_performers.append((target_value, target_path, bot.step))
            self.remove_checkpoints(keep=self.keep_n_checkpoints)
            bot.model.save_weights(str(target_path))
            assert target_path.exists()

    def remove_checkpoints(self, keep):
        self.best_performers = sorted(self.best_performers, key=lambda x: x[0])
        for checkpoint in np.unique([
                x[1] for x in self.best_performers[keep:]]):
            Path(checkpoint).unlink()
        self.best_performers = self.best_performers[:keep]

    def reset(self, ignore_previous=False):
        if ignore_previous:
            self.best_performers = []
        else:
            self.remove_checkpoints(0)
