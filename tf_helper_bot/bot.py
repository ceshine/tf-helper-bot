import logging
from pathlib import Path
from typing import Callable, Sequence, Union, Optional

import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from tqdm.autonotebook import tqdm

from .logger import Logger


@dataclass
class BaseBot:
    """Base Interface to Model Training and Inference"""
    train_dataset: tf.data.Dataset
    valid_dataset: tf.data.Dataset
    steps_per_epoch: int
    criterion: Callable
    model: tf.keras.Model
    optimizer: tf.keras.optimizers.Optimizer
    name: str = "basebot"
    log_dir: Union[Path, str] = "./logs"
    log_level: int = logging.INFO
    loss_format: str = "%.4f"
    echo: bool = True
    pbar: bool = True
    step: int = 0
    total_steps: int = 0
    valid_steps: Optional[int] = None
    gradient_accumulation_steps: int = 1
    metrics: Sequence = ()
    callbacks: Sequence = ()

    def __post_init__(self):
        self._gradients = []
        self.logger = Logger(
            self.name, Path(self.log_dir), self.log_level,
            echo=self.echo
        )

        @tf.function
        def get_gradient(input_tensors, target):
            with tf.GradientTape() as tape:
                output = self.model(
                    input_tensors, training=True)
                loss_ = self.criterion(
                    target, self.extract_prediction(output)
                )
            gradients_ = tape.gradient(
                loss_, self.model.trainable_variables)
            return loss_, gradients_

        @tf.function
        def step_optimizer(gradients):
            self.optimizer.apply_gradients(
                zip(
                    gradients,
                    self.model.trainable_variables
                )
            )

        @tf.function
        def predict_batch(input_tensors):
            return self.model(input_tensors, training=False)

        self._get_gradient = get_gradient
        self._step_optimizer = step_optimizer
        self._predict_batch = predict_batch

    def train_one_step(self, input_tensor_list, target):
        loss, gradients = self._get_gradient(
            input_tensor_list[0], target)
        if self.gradient_accumulation_steps > 1:
            loss, gradients = self._get_gradient(
                input_tensor_list[0], target)
            for i in range(1, self.gradient_accumulation_steps):
                loss_, gradients_ = self._get_gradient(
                    input_tensor_list[i], target)
                gradients = [
                    grad_1 + grad_2
                    for grad_1, grad_2 in zip(gradients, gradients_)
                ]
                loss = loss + loss_
            gradients = [
                grad / tf.constant(
                    self.gradient_accumulation_steps,
                    dtype=tf.float32
                )
                for grad in gradients
            ]
            loss = loss / tf.constant(
                self.gradient_accumulation_steps,
                dtype=tf.float32
            )
        self._step_optimizer(gradients)
        return loss

    @staticmethod
    def extract_prediction(output):
        return output

    def train(self, *, checkpoint_interval, n_steps=None, total_steps=None):
        if total_steps:
            self.total_steps = total_steps
        if n_steps is None:
            if self.total_steps is None:
                raise ValueError("n_steps and total_steps cannot both be None")
            n_steps = self.total_steps - self.step
        elif self.total_steps is None:
            self.total_steps = n_steps
        target_step = self.step + n_steps
        input_tensor_list, cnt = [], 0
        # Train starts
        self.run_train_starts_callbacks()
        try:
            while self.step < target_step:
                for input_tensors, targets in self.train_dataset:
                    self.step += 1
                    input_tensors, targets = self.run_batch_inputs_callbacks(
                        input_tensors, targets)
                    input_tensor_list.append(input_tensors)
                    cnt += input_tensors.shape[0]
                    if len(input_tensor_list) == self.gradient_accumulation_steps:
                        loss = self.train_one_step(
                            input_tensor_list, targets
                        )
                        # Step ends
                        self.run_step_ends_callbacks(loss, cnt)
                        input_tensor_list, cnt = [], 0
                    if (
                        (callable(checkpoint_interval) and checkpoint_interval(self.step)) or
                        (
                            not callable(checkpoint_interval) and
                            self.step % checkpoint_interval == 0
                        )
                    ):
                        # Eval starts
                        metrics = self.eval(self.valid_dataset)
                        # Eval ends
                        self.run_eval_ends_callbacks(metrics)
                    if self.step >= target_step:
                        break
                    # Epoch ends
                    if self.step % self.steps_per_epoch == 0:
                        self.run_epoch_ends_callbacks(
                            self.step // self.steps_per_epoch)
        except (KeyboardInterrupt):
            pass
        finally:
            # Train ends
            self.run_train_ends_callbacks()

    def predict(self, dataset, *, return_y=False):
        self.model.eval()
        outputs, y_global = [], []
        for *input_tensors, y_local in tqdm(dataset, disable=not self.pbar):
            outputs.append(self._predict_batch(input_tensors).numpy())
            if return_y:
                y_global.append(y_local.numpy())
        outputs = np.concatenate(outputs, axis=0)
        if return_y:
            y_global = np.concatenate(y_global, axis=0)
            return outputs, y_global
        return outputs

    def eval(self, dataset):
        """Warning: Only support datasets whose predictions and labels together fit in memory."""
        preds, ys = [], []
        losses, weights = [], []
        self.logger.debug("Evaluating...")
        for *input_tensors, y_local in tqdm(dataset, disable=not self.pbar, total=self.valid_steps):
            output = self.extract_prediction(
                self._predict_batch(input_tensors))
            batch_loss = self.criterion(y_local, output)
            losses.append(batch_loss.numpy())
            weights.append(y_local.shape[0])
            # Save batch labels and predictions
            preds.append(output.numpy())
            ys.append(y_local.numpy())
        loss = np.average(losses, weights=weights)
        metrics = {"loss": (loss, self.loss_format % loss)}
        global_ys, global_preds = np.concatenate(ys), np.concatenate(preds)
        for metric in self.metrics:
            metric_loss, metric_string = metric(global_ys, global_preds)
            metrics[metric.name] = (metric_loss, metric_string)
        return metrics

    def run_batch_inputs_callbacks(self, input_tensors, targets):
        for callback in self.callbacks:
            input_tensors, targets = callback.on_batch_inputs(
                self, input_tensors, targets)
        return input_tensors, targets

    def run_step_ends_callbacks(self, train_loss, train_weight):
        for callback in self.callbacks:
            callback.on_step_ends(self, train_loss, train_weight)

    def run_train_starts_callbacks(self):
        for callback in self.callbacks:
            callback.on_train_starts(self)

    def run_train_ends_callbacks(self):
        for callback in self.callbacks:
            callback.on_train_ends(self)

    def run_epoch_ends_callbacks(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_ends(self, epoch)

    def run_eval_ends_callbacks(self, metrics):
        for callback in self.callbacks:
            callback.on_eval_ends(self, metrics)


@dataclass
class BaseDistributedBot(BaseBot):
    """Base Interface to Model Training and Inference"""
    strategy: tf.distribute.Strategy

    def __post_init__(self):
        assert self.gradient_accumulation_steps == 1, (
            "Distribution mode doesn't suppoprt gradient accumulation"
        )
        super().__post_init__()
        @tf.function
        def train_one_step(input_tensor_list, target):
            loss, gradients = self._get_gradient(
                input_tensor_list[0], target)
            self._step_optimizer(gradients)
            return loss
        self._train_one_step = train_one_step

    def train_one_step(self, input_tensors, target):
        loss = self.strategy.experimental_run_v2(
            self._train_one_step,
            args=(input_tensors, target)
        )
        return self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, loss, axis=None
        )
