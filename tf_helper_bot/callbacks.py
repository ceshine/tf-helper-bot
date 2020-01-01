import socket
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, Tuple, List, Optional
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
    "Callback", "MovingAverageStatsTrackerCallback"
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

    def on_step_ends(self, bot: BaseBot, train_loss, train_weight):
        self.train_losses.append(train_loss)
        self.train_weights.append(train_weight)
        if bot.step % self.log_interval == 0:
            train_loss_avg = np.average(
                self.train_losses, weights=self.train_weights)
            lr = (
                bot.optimizer.lr(bot.step) if callable(bot.optimizer.lr)
                else bot.optimizer.lr
            )
            bot.logger.info(
                f"Step %s: loss {bot.loss_format} lr: %.3e",
                bot.step, train_loss_avg, lr)
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
