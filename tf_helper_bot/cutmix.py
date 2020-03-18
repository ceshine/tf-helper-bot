import tensorflow as tf
import tensorflow_probability as tfp

DEBUG = False


def cutmix_augment(alpha: float):
    """ Adapted from Cutout implementation of tensorflow/models

    Reference: https://github.com/tensorflow/models/blob/831281cedfc8a4a0ad7c0c37173963fafb99da37/research/object_detection/utils/autoaugment_utils.py#L194
    """
    dist = tfp.distributions.Beta(alpha, alpha)

    def cutmix_(images, labels):
        batch_size = tf.shape(images)[0]
        image_height = tf.shape(images)[1]
        image_width = tf.shape(images)[2]
        index = tf.random.shuffle(tf.range(batch_size))

        def sample_func(lambda_):
            cutout_center_height = tf.random.uniform(
                shape=[], minval=0, maxval=image_height,
                dtype=tf.int32)

            cutout_center_width = tf.random.uniform(
                shape=[], minval=0, maxval=image_width,
                dtype=tf.int32)

            mask_width = tf.math.round(
                tf.cast(image_width, tf.float32) * tf.math.sqrt(1 - lambda_)
            )
            mask_height = tf.math.round(
                tf.cast(image_height, tf.float32) * tf.math.sqrt(1 - lambda_)
            )

            lower_pad = tf.maximum(
                0, cutout_center_height -
                tf.cast(tf.math.floor(mask_height / 2.), tf.int32)
            )
            upper_pad = tf.maximum(
                0, image_height - cutout_center_height -
                tf.cast(tf.math.ceil(mask_height / 2.), tf.int32)
            )
            left_pad = tf.maximum(
                0, cutout_center_width -
                tf.cast(tf.math.floor(mask_width / 2.), tf.int32)
            )
            right_pad = tf.maximum(
                0, image_width - cutout_center_width -
                tf.cast(tf.math.ceil(mask_width / 2.), tf.int32)
            )

            cutout_shape = [image_height - (lower_pad + upper_pad),
                            image_width - (left_pad + right_pad)]
            padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
            mask = tf.pad(
                tf.zeros(cutout_shape, dtype=images.dtype),
                padding_dims, constant_values=1
            )
            mask = tf.expand_dims(mask, -1)
            lambda_adj = (
                1 -
                tf.cast(cutout_shape[0] * cutout_shape[1], tf.float32) /
                tf.cast(image_height * image_width, tf.float32)
            )
            return {"mask": mask, "lambda": lambda_adj}

        lambdas = dist.sample([batch_size])
        lambdas = tf.math.reduce_max(
            tf.stack([lambdas, 1-lambdas]), axis=0
        )
        masks = tf.map_fn(
            sample_func, lambdas,
            dtype={"mask": images.dtype, "lambda": tf.float32}
        )
        if DEBUG:
            return (
                tf.zeros_like(images) * masks["mask"] +
                tf.ones_like(images) * (1-masks["mask"]) * 0.5,
                {
                    "labels_1": labels,
                    "labels_2": tf.gather(labels, index),
                    "lambd": masks["lambda"]
                }
            )
        return (
            images * masks["mask"] +
            tf.gather(images, index) * (1-masks["mask"]),
            {
                "labels_1": labels,
                "labels_2": tf.gather(labels, index),
                "lambd": masks["lambda"]
            }
        )
    return cutmix_
