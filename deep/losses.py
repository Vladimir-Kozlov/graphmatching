import tensorflow as tf
keras = tf.keras
from __future__ import division


def set_loss_params(*args, **kwargs):
	"""
	Decorator intended for setting parameters of loss function
	"""
	return lambda loss: lambda y_true, y_pred: loss(y_true, y_pred, *args, **kwargs)


def loss_vertex_match_strict(y_true, y_pred, C=1.):
	"""
	Vertex similarity loss.
	Input: y_true: true matching, expected to be a matrix of 0 (should not match) and 1 (should match)
	       y_pred: actual match, expected to be a matrix of values between 0 and 1 (confidence of match)
	       C: weight parameter
	Output: sum(1 - elements of y_pred that should match) + C * sum(elements of y_pred that should not match)
	"""
	return tf.reduce_sum(y_true * (1. - y_pred) + C * (1. - y_true) * y_pred, axis=[-2, -1], keepdims=False)
