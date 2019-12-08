import numpy as np
import tensorflow as tf
keras = tf.keras


def loss_wrapper(*args, **kwargs):
    def decorator(loss_to_decorate):
        def wrapper(y_true, y_pred):
            return loss_to_decorate(y_true, y_pred, *args, **kwargs)
        return wrapper
    return decorator


def loss_vertex_match_strict(y_true, y_pred, c=1.):
    """
    Simple loss function
    :param y_true: true matching, expected to be a matrix of 0 (should not match) and 1 (should match)
    :param y_pred: actual match, expected to be a matrix of values between 0 and 1 (confidence of match)
    :param C: weight parameter
    :return: sum(1 - elements of y_pred that should match) + C * sum(elements of y_pred that should not match)
    """
    return tf.reduce_mean(y_true * (1. - y_pred) + c * (1. - y_true) * y_pred, axis=[-2, -1], keepdims=False)


def loss_vertex_similarity(y_true, y_pred, c_sim=1., c_dissim=1., c_rel=1., scale=1.):
    weights = np.array([c_sim, c_dissim, c_rel])
    weights /= np.sum(weights)
    y = - tf.math.log(tf.clip_by_value(y_pred, 1e-9, 1.))
    y1 = y - tf.expand_dims(y[:, -1, :], axis=-2)
    y2 = y - tf.expand_dims(y[:, :, -1], axis=-1)
    zeros = tf.zeros(tf.shape(y))
    z = weights[0] * tf.where(tf.equal(y_true, 1.), y, zeros) + \
        weights[1] * tf.where(tf.equal(y_true, 0.), tf.maximum(0., scale - y), zeros) + \
        weights[2] * tf.where(tf.equal(y_true, 0.), tf.maximum(0., scale - y1) + tf.maximum(0., scale - y2), zeros)
    return tf.reduce_sum(z, axis=[-2, -1], keepdims=False) / \
           tf.maximum(tf.reduce_sum(tf.where(tf.logical_or(tf.equal(y_true, 1.), tf.equal(y_true, 0.)),
                                             tf.ones(tf.shape(y)), zeros), axis=[-2, -1]), 1.)
