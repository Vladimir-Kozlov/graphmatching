from __future__ import division
import tensorflow as tf
keras = tf.keras


def idxtransform(idx, transform=lambda x: x // 32):
    # Transform coordinates of image points to index feature vectors in high layers
    # Input: idx: point coordinates of shape [batch size, number of points, 2]
    #        transform: index transformation function; by default, we simply divide them by 32 (corresponding to feature map size in CNN)
    # Output: transformed indices with added leading batch number
    x = transform(idx)
    r = tf.reshape(tf.range(tf.shape(idx)[0]), [-1, 1, 1]) #create range of batch numbers
    r = tf.tile(r, tf.concat([[1], tf.shape(idx)[1:-1], [1]], axis=0))
    return tf.concat([r, x], axis=-1)


class FMapIndexLayer(keras.layers.Layer):
    # Performs feature indexing from feature map
    def __init__(self, transform=lambda x: x // 32, **kwargs):
        self.transform = transform
        super(FMapIndexLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        super(FMapIndexLayer, self).build(input_shape)

    def call(self, x):
        # Input: img: images, [batch size, height, width, channel]
        #        idx: keypoint coordinates, [batch size, number of points, 2]
        # Output: keypoint features, [batch size, number of points, channel]
        idx = idxtransform(x[1], transform=self.transform)
        return tf.gather_nd(x[0], idx)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        return (input_shape[1][0], input_shape[0][-1])


class EdgeAttributeLayer(keras.layers.Lambda):
	# General layer for calculating graph edge attributes
	# Specific form of lambda layer
	__alias = {'concat': lambda u, v: tf.concat([u, v], axis=-1), 
	           'l2dist': lambda u, v: tf.norm(tf.to_float(u - v), ord='euclidean', axis=-1, keepdims=True),
	           'l2dist_squared': lambda u, v: tf.reduce_sum(tf.to_float(u - v)**2, axis=-1, keepdims=True)}
	def __init__(self, attr_func='concat', **kwargs):
		if isinstance(attr_func, str):
			attr_func = self.__alias[attr_func]
		super(EdgeAttributeLayer, self).__init__(function=attr_func, **kwargs)

