from __future__ import division
import tensorflow as tf
keras = tf.keras


def cosine_similarity(u, v):
    x = tf.math.l2_normalize(u, axis=-1)
    y = tf.math.l2_normalize(v, axis=-1)
    return tf.linalg.matmul(x, y, transpose_b=True)


class AffinityLayer(keras.layers.Layer):
    # Calculates pairwise similarity of two sets of vectors
    def __init__(self, sim_func=lambda u, v: (cosine_similarity(u, v) + 1.) / 2., **kwargs):
        self.sim_func = sim_func
        super(AffinityLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        super(AffinityLayer, self).build(input_shape)

    def call(self, x):
        # Input: V_l, V_r: matrices of object features of shape [n1, k] and [n2, k] respectively
        # Output: Mp: pairwise similarities between object pairs, matrix of shape [n1, n2]
        assert isinstance(x, (list, tuple))
        return self.sim_func(x[0], x[1])

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        return (input_shape[0][0], input_shape[1][0])

