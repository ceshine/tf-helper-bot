from typing import Callable, Sequence

import tensorflow as tf
from dataclasses import dataclass, field


@dataclass
class BaseBot:
    """Base Interface to Model Training and Inference"""
    train_dataset: tf.data.Dataset
    valid_dataset: tf.data.Dataset
    criterion: Callable
    model: tf.keras.Model
    optimizer: tf.keras.optimizers.Optimizer
    strategy: tf.distribute.Strategy
    step: int = 0
    total_steps: int = 0
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        self._gradients = []

        @tf.function
        def get_gradient(input_tensor, target):
            with tf.GradientTape() as tape:
                output = self.model(
                    input_tensor, training=True)
                loss_ = self.criterion(
                    target, self.extract_prediction(output)
                )
            gradients_ = tape.gradient(
                loss_, self.model.trainable_variables)
            return loss_, gradients_

        def train_one_step(input_tensor_list, target):
            if self.gradient_accumulation_steps == 1:
                loss, gradients = get_gradient(
                    input_tensor_list[0], target)
            else:
                loss, gradients = get_gradient(
                    input_tensor_list[0], target)
                for i in range(1, self.gradient_accumulation_steps):
                    loss_, gradients_ = get_gradient(
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
                # grads = [
                #     tf.TensorArray(
                #         tf.float32, size=self.gradient_accumulation_steps
                #     )
                #     for v in self.model.trainable_variables
                # ]
                # loss_tmp = tf.TensorArray(
                #     tf.float32, size=self.gradient_accumulation_steps
                # )
                # for i in range(self.gradient_accumulation_steps):
                #     loss_, gradients_ = get_gradient(
                #         input_tensor_list[i], target)
                #     loss_tmp = loss_tmp.write(tf.constant(i), loss_)
                #     for j, g in enumerate(gradients_):
                #         grads[j] = grads[j].write(tf.constant(i), g)
                # loss = tf.reduce_mean(loss_tmp.stack())
                # gradients = [
                #     tf.reduce_mean(grad_arr.stack(), axis=0) for grad_arr in grads
                # ]
                # tf.print(gradients[0])
            self.optimizer.apply_gradients(
                zip(
                    gradients,
                    self.model.trainable_variables
                )
            )
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
        epoch = 0
        input_tensor_list, cnt = [], 0
        # Train starts
        try:
            while self.step < target_step:
                epoch += 1
                for input_tensors, targets in self.train_dataset:
                    self.step += 1
                    input_tensor_list.append(input_tensors)
                    cnt += input_tensors.shape[0]
                    if len(input_tensor_list) == self.gradient_accumulation_steps:
                        loss = self.train_one_step(
                            input_tensor_list, targets
                        )
                        print("%.4f" % loss.numpy())
                        input_tensor_list, cnt = [], 0
                    if self.step >= target_step:
                        break
        except (KeyboardInterrupt):
            pass
