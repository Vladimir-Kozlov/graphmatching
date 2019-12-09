import tensorflow as tf
keras = tf.keras


# def idxtransform(idx, transform=lambda x: x // 32):
#     # Transform coordinates of image points to index feature vectors in high layers
#     # Input: idx: point coordinates of shape [batch size, number of points, 2]
#     #        transform: index transformation function; by default, we simply divide them by 32 (corresponding to feature map size in CNN)
#     # Output: transformed indices with added leading batch number
#     x = transform(idx)
#     r = tf.reshape(tf.range(tf.shape(idx)[0]), [-1, 1, 1]) #create range of batch numbers
#     r = tf.tile(r, tf.concat([[1], tf.shape(idx)[1:-1], [1]], axis=0))
#     return tf.concat([r, x], axis=-1)


# class FMapIndexLayer(keras.layers.Layer):
#     # Performs feature indexing from feature map
#     def __init__(self, transform=lambda x: x // 32, **kwargs):
#         self.transform = transform
#         super(FMapIndexLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         assert isinstance(input_shape, (list, tuple))
#         super(FMapIndexLayer, self).build(input_shape)

#     def call(self, x):
#         # Input: img: images, [batch size, height, width, channel]
#         #        idx: keypoint coordinates, [batch size, number of points, 2]
#         # Output: keypoint features, [batch size, number of points, channel]
#         idx = idxtransform(x[1], transform=self.transform)
#         return tf.gather_nd(x[0], idx)

#     def compute_output_shape(self, input_shape):
#         assert isinstance(input_shape, (list, tuple))
#         return (input_shape[1][0], input_shape[0][-1])


class FMapIndexLayer(keras.layers.Lambda):
    """
    Performs feature indexing from feature map
    Input: img: images, [batch size, height, width, channel]
           idx: keypoint coordinates, [batch size, number of points, 2]
    Output: keypoint features, [batch size, number of points, channel]
    """
    def __init__(self, **kwargs):
        super(FMapIndexLayer, self).__init__(function=lambda x: self._mapidx2list(x[0], x[1]), **kwargs)

    @staticmethod
    def _mapidx2list(img, idx):
        def idxtransform(idx):
            # Appends batch number at the beginning of index
            r = tf.reshape(tf.range(tf.shape(idx)[0]), [-1, 1, 1])  # create range of batch numbers
            r = tf.tile(r, tf.concat([[1], tf.shape(idx)[1:-1], [1]], axis=0))
            return tf.concat([r, idx], axis=-1)

        return tf.gather_nd(img, idxtransform(idx))


class EdgeFeatExtract(keras.layers.Lambda):
    def __init__(self, **kwargs):
        super(EdgeFeatExtract, self).__init__(function=lambda x: self._extract_both(x[0], x[1], x[2]), **kwargs)

    @staticmethod
    def _extract_both(x, G, H):
        """
        Calculates features on both ends of edges
        :param x: feature list, [n_vertex, num_of_feat]; extracted from conv-map by FMapIndexLayer, or coordinates
        :param G: incidence matrix, [n_vertex, n_edges]: G[i, k] = 1 iff edge k starts in vertex i
        :param H: incidence matrix, [n_vertex, n_edges]: H[j, k] = 1 iff edge k ends in vertex j
        :return: features on both ends of edges: tuple of matrices of shape [n_edges, num_of_feat] both
        """
        def extract_at_end(x, M):
            return tf.linalg.matmul(M, tf.cast(x, tf.float32), transpose_a=True)

        return extract_at_end(x, G), extract_at_end(x, H)


class EdgeAttributeLayer(keras.layers.Lambda):
    """
    General layer for calculating graph edge attributes. Specific form of lambda layer.
    Designed to be used with EdgeFeatExtract method.
    Example:
        x = FMapIndexLayer()([img, idx])
        x = EdgeFeatExtract()([x, G, H])
        x = EdgeAttributeLayer(attr_func='concat')(x)
    Or:
        x = EdgeFeatExtract([idx, G, H])
        x = EdgeAttributeLayer(attr_func='l2_dist')(x)
    """
    def __init__(self, attr_func='concat', **kwargs):
        if isinstance(attr_func, str):
            if attr_func == 'concat':
                attr_func = lambda x: tf.concat(x, axis=-1)
            elif attr_func == 'l2_dist':
                attr_func = lambda x: tf.norm(tf.cast(x[0] - x[1], tf.float32), ord='euclidean', axis=-1, keepdims=True)
            elif attr_func == 'l2_dist_squared':
                attr_func = lambda x: tf.reduce_sum(tf.cast(x[0] - x[1], tf.float32)**2, axis=-1, keepdims=True)
            else:
                raise ValueError('Only concat, l2_dist, l2_dist_squared options are available right now')
        super(EdgeAttributeLayer, self).__init__(function=attr_func, **kwargs)


class DummyFeaturesLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DummyFeaturesLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dummy_features = self.add_weight(name='dummy_feature_vector', shape=(input_shape[-1], ),
                                              initializer='uniform', trainable=True)
        super(DummyFeaturesLayer, self).build(input_shape)

    def call(self, vertex_features):
        s = tf.shape(vertex_features)
        s = tf.concat([s[:-2], [1, s[-1]]], axis=0)
        return tf.concat([vertex_features, tf.ones(s) * self.dummy_features], axis=-2)

    @staticmethod
    def compute_output_shape(input_shape):
        return input_shape[-2] + 1, input_shape[-1]


class DummyEdgeLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DummyEdgeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DummyEdgeLayer, self).build(input_shape)

    def call(self, edges):
        s = tf.shape(edges)
        s = tf.concat([s[:-2], [1, s[-1]]], axis=0)
        return tf.concat([edges, tf.fill(s, 0.)], axis=-2)

    @staticmethod
    def compute_output_shape(input_shape):
        return [input_shape[-2] + 1, input_shape[-1]]
