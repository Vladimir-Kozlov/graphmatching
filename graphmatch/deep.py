import tensorflow as tf
from tensorflow import keras
import numpy as np


def sinkhorn_loop(x, max_iter=100, eps_iter=1e-6):
    # Sinkhorn-Knopp algorithm for transforming matrix with nonnegative elements into doubly stochastic one
    # Input: x: matrix to be processed
    #        max_iter: maximum number of iterations in a loop
    #        eps_iter: acceptable precision, looping stops when change is less than eps_iter
    # Output: doubly stochastic matrix
    def sinkhorn_iter(x):
        u = tf.math.reduce_sum(x, axis=-2, keepdims=True)
        h = tf.div_no_nan(x, u) #x / tf.math.maximum(u, eps_divzero)
        v = tf.math.reduce_sum(h, axis=-1, keepdims=True)
        return tf.div_no_nan(h, v) #h / tf.math.maximum(v, eps_divzero)
    def cond(x, i, d):
        return tf.math.logical_and(i < max_iter, d)
    def body(x, i, d):
        y = sinkhorn_iter(x)
        return y, i + 1, tf.math.reduce_any(tf.linalg.norm(y - x, axis=(-2, -1)) >= eps_iter)
    
    i = tf.constant(0)
    d = tf.constant(True)
    return tf.while_loop(cond=cond, body=body, loop_vars=(x, i, d), maximum_iterations=max_iter)[0]