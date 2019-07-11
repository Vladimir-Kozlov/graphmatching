import tensorflow as tf
from tensorflow import keras
import numpy as np


def idxtransform(idx, transform=lambda x: x / 32):
    # Transform coordinates of image points to index feature vectors in high layers
    # Input: idx: point coordinates of shape [batch size, number of points, 2]
    #        transform: index transformation function; by default, we simply divide them by 32 (corresponding to feature map size in CNN)
    # Output: transformed indices with added leading batch number
    x = transform(idx)
    r = tf.reshape(tf.range(tf.shape(idx)[0]), [-1, 1, 1]) #create range of batch numbers
    r = tf.tile(r, tf.concat([[1], tf.shape(idx)[1:-1], [1]], axis=0))
    return tf.concat([r, x], axis=-1)


class IndexTransformationLayer(keras.layers.Layer):
    def __init__(self, scale=2**5, **kwargs):
        self.scale = scale
        super(IndexTransformationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(IndexTransformationLayer, self).build(input_shape)

    def call(self, x):
        return idxtransform(x, scale=self.scale)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + [3]

        