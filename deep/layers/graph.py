import tensorflow as tf
keras = tf.keras


class AdjacencyMatrixLayer(keras.layers.Layer):
    """
    Takes edge incidence matrices G and H and returns adjacency matrix A and in- and out-degree vectors D_in and D_out.
    This is used to calculate the average over adjacent vertices and incident edges in convolution layer.
    """
    def __init__(self, **kwargs):
        super(AdjacencyMatrixLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        super(AdjacencyMatrixLayer, self).build(input_shape)

    @staticmethod
    def call(x):
        """
        Calculates A, D_in and D_out
        :param x: G, H - incidence matrices, of shape [n_vertex, n_edges] each
        :return: A - adjacency matrix, A = G H^T; D_in, D_out - in- and out-degree vectors, [n_vertex] and [n_vertex]
        """
        assert isinstance(x, (list, tuple))
        G, H = x
        A = tf.matmul(G, H, transpose_b=True)
        D_in = tf.reduce_sum(A, axis=-2, keepdims=False)
        D_out = tf.reduce_sum(A, axis=-1, keepdims=False)
        return A, D_in, D_out

    @staticmethod
    def compute_output_shape(input_shape):
        isinstance(input_shape, (list, tuple))
        G_shape, H_shape = input_shape
        return (G_shape[-2], H_shape[-2]), (H_shape[-2], ), (G_shape[-2], )


class GraphConvVertexLayer(keras.layers.Layer):
    def __init__(self, units: int, activation=None, **kwargs):
        self.units = units
        self.activation = activation if activation is not None else lambda x: x
        super(GraphConvVertexLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        VF_shape, EF_shape = input_shape[0], input_shape[1]
        self.kernel_self = self.add_weight(name='vertex_kernel_self', shape=(VF_shape[-1], self.units),
                                           initializer='uniform', trainable=True)
        self.kernel_in = self.add_weight(name='vertex_kernel_in', shape=(VF_shape[-1], self.units),
                                         initializer='uniform', trainable=True)
        self.kernel_out = self.add_weight(name='vertex_kernel_out', shape=(VF_shape[-1], self.units),
                                          initializer='uniform', trainable=True)
        self.kernel_edge_in = self.add_weight(name='edge_kernel_in', shape=(EF_shape[-1], self.units),
                                              initializer='uniform', trainable=True)
        self.kernel_edge_out = self.add_weight(name='edge_kernel_out', shape=(EF_shape[-1], self.units),
                                               initializer='uniform', trainable=True)
        self.bias = self.add_weight(name='vertex_bias', shape=(self.units, ), initializer='zeros', trainable=True)
        super(GraphConvVertexLayer, self).build(input_shape)
    
    def call(self, x):
        """
        Call result
        :param x: vertex features [n_vertex, nvfeats], edge_features [n_edge, nefeats],
                  G [n_vertex, n_features], H [n_vertex, n_features], -- edge incidence matrices
                  A [n_vertex, n_vertex] -- adjacency matrix,
                  D_in [1, n_vertex], D_out [n_vertex, 1] -- in- and out-degree
        :return:
        """
        assert isinstance(x, (list, tuple))
        VF, EF, G, H, A, D_in, D_out = x
        z = tf.matmul(VF, self.kernel_self) + \
            tf.matmul(tf.matmul(tf.div_no_nan(A, tf.expand_dims(D_in, axis=-2)), VF, transpose_a=True), self.kernel_in) + \
            tf.matmul(tf.matmul(tf.div_no_nan(A, tf.expand_dims(D_out, axis=-1)), VF), self.kernel_out) + \
            tf.matmul(tf.matmul(tf.div_no_nan(G, tf.expand_dims(D_in, axis=-1)), EF), self.kernel_edge_in) + \
            tf.matmul(tf.matmul(tf.div_no_nan(H, tf.expand_dims(D_out, axis=-1)), EF), self.kernel_edge_out) + \
            self.bias
        return self.activation(z)
    
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        VF_shape = input_shape[0]
        return VF_shape[-2], self.units


class GraphConvEdgeLayer(keras.layers.Layer):
    def __init__(self, units: int, activation=None, **kwargs):
        """
        Convolution layer for edges in graph
        :param units: number of cells (output features)
        :param kwargs: layer parameters
        """
        self.activation = activation if activation is not None else lambda x: x
        self.units = units
        super(GraphConvEdgeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        VF_shape, EF_shape = input_shape[0], input_shape[1]
        self.kernel_self = self.add_weight(name='edge_kernel_self', shape=(EF_shape[-1], self.units),
                                           initializer='uniform', trainable=True)
        self.kernel_in = self.add_weight(name='vertex_kernel_in', shape=(VF_shape[-1], self.units),
                                         initializer='uniform', trainable=True)
        self.kernel_out = self.add_weight(name='vertex_kernel_out', shape=(VF_shape[-1], self.units),
                                          initializer='uniform', trainable=True)
        self.bias = self.add_weight(name='edge_bias', shape=(self.units,), initializer='zeros', trainable=True)
        super(GraphConvEdgeLayer, self).build(input_shape)

    def call(self, x):
        """
        Call result
        :param x: vertex features [n_vertex, nvfeats], edge_features [n_edge, nefeats],
                  G [n_vertex, n_features], H [n_vertex, n_features], -- edge incidence matrices
                  A [n_vertex, n_vertex] -- adjacency matrix,
                  D_in [1, n_vertex], D_out [n_vertex, 1] -- in- and out-degree
        :return:
        """
        assert isinstance(x, (list, tuple))
        VF, EF, G, H = x[0], x[1], x[2], x[3]
        z = tf.matmul(EF, self.kernel_self) + \
            tf.matmul(tf.matmul(G, VF, transpose_a=True), self.kernel_in) + \
            tf.matmul(tf.matmul(H, VF, transpose_a=True), self.kernel_out) + \
            self.bias
        return self.activation(z)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        EF_shape = input_shape[1]
        return EF_shape[-2], self.units
