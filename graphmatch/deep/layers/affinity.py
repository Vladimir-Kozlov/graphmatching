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
        assert isinstance(input_shape, list)
        super(AffinityLayer, self).build(input_shape)

    def call(self, x):
        # Input: V_l, V_r: matrices of object features of shape [n1, k] and [n2, k] respectively
        # Output: Mp: pairwise similarities between object pairs, matrix of shape [n1, n2]
        assert isinstance(x, list)
        return self.sim_func(x[0], x[1])

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (input_shape[0][0], input_shape[1][0])


class VertexAffinityLayer(keras.layers.Layer):
    # This layer performs calculations of vertex affinity matrix
    def __init__(self, transform_dim=1, sim_func=lambda u, v: (cosine_similarity(u, v) + 1.) / 2., **kwargs):
        self.transform_dim = transform_dim
        self.sim_func = sim_func
        super(VertexAffinityLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.transform_matrix = self.add_weight(name='transform_matrix',
                                                shape=(input_shape[0][-1].value, self.transform_dim),
                                                initializer='orthogonal', trainable=True)
        super(VertexAffinityLayer, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        # Input: V_l, V_r: matrices of vertex features of shape [n1, k] and [n2, k] respectively
        # Output: Mp: cosine similarities between V_l @ transform_matrix and V_r @ transform_matrix,
        #             normalized to [0, 1].
        
        U_l = tf.tensordot(x[0], self.transform_matrix, axes=1)
        U_r = tf.tensordot(x[1], self.transform_matrix, axes=1)

        return self.sim_func(U_l, U_r)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (input_shape[0][0], input_shape[1][0])


class EdgeAffinityLayer(keras.layers.Layer):
    # Layer that calculates edge affinity matrix from edge feature vectors
    def __init__(self, transform_dim=1, sim_func=lambda u, v: (cosine_similarity(u, v) + 1.) / 2., **kwargs):
        self.transform_dim = transform_dim
        self.sim_func = sim_func
        super(EdgeAffinityLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.transform_matrix = self.add_weight(name='transform_matrix',
                                                shape=(input_shape[0][-1].value, self.transform_dim),
                                                initializer='orthogonal', trainable=True)
        super(EdgeAffinityLayer, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        # Input: E_l, E_r: matrices of edge features
        #        of shape [m1, edge feature vector length] and [m2, EFVL] respectively
        #        Masking is done by incidence matrices
        # Output: Mq: cosine similarities between E_l @ transform_matrix and E_r @ transform_matrix, normalized to [0, 1]
        
        E_l = tf.tensordot(x[0], self.transform_matrix, axes=1)
        E_r = tf.tensordot(x[1], self.transform_matrix, axes=1)

        return self.sim_func(E_l, E_r)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (input_shape[0][0], input_shape[1][0])

