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
    def __init__(self, units: int, activation=None, use_bias=True, attention=None, use_edge_attr=False, **kwargs):
        self.units = units
        self.activation = activation if activation is not None else lambda x: x
        self.use_bias = use_bias
        self.attention = attention
        self.use_edge_attr = use_edge_attr

        self.kernel_vertex_self = None
        self.kernel_vertex_value_in = None
        self.kernel_vertex_value_out = None
        self.kernel_vertex_key_in = None
        self.kernel_vertex_key_out = None
        self.kernel_vertex_query_in = None
        self.kernel_vertex_query_out = None

        self.kernel_edge_value_in = None
        self.kernel_edge_value_out = None
        self.kernel_edge_score_in = None
        self.kernel_edge_score_out = None

        self.bias = None
        super(GraphConvVertexLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        VF_shape = input_shape[0]
        if self.use_edge_attr:
            EF_shape = input_shape[1]
        self.kernel_vertex_self = self.add_weight(name='kernel_vertex_self', shape=(VF_shape[-1], self.units),
                                                  initializer='uniform', trainable=True)
        self.kernel_vertex_value_in = self.add_weight(name='kernel_vertex_value_in', shape=(VF_shape[-1], self.units),
                                                      initializer='uniform', trainable=True)
        self.kernel_vertex_value_out = self.add_weight(name='kernel_vertex_value_out', shape=(VF_shape[-1], self.units),
                                                       initializer='uniform', trainable=True)
        if self.use_edge_attr:
            self.kernel_edge_value_in = self.add_weight(name='kernel_edge_value_in', shape=(EF_shape[-1], self.units),
                                                        initializer='uniform', trainable=True)
            self.kernel_edge_value_out = self.add_weight(name='kernel_edge_value_out', shape=(EF_shape[-1], self.units),
                                                         initializer='uniform', trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias_vertex', shape=(self.units,), initializer='zeros', trainable=True)
        if self.attention:
            # Attention over neighbors; here, our goal is to calculate weight with which message will be averaged
            # full score = vertex score + edge score
            # vertex score: score(v_i, v_j) - depends on both vertices
            self.kernel_vertex_query_in = self.add_weight(name='kernel_vertex_query_in',
                                                          shape=(VF_shape[-1], self.attention),
                                                          initializer='uniform', trainable=True)
            self.kernel_vertex_query_out = self.add_weight(name='kernel_vertex_query_out',
                                                           shape=(VF_shape[-1], self.attention),
                                                           initializer='uniform', trainable=True)
            self.kernel_vertex_key_in = self.add_weight(name='kernel_vertex_key_in',
                                                        shape=(VF_shape[-1], self.attention),
                                                        initializer='uniform', trainable=True)
            self.kernel_vertex_key_out = self.add_weight(name='kernel_vertex_key_out',
                                                         shape=(VF_shape[-1], self.attention),
                                                         initializer='uniform', trainable=True)
            # edge score: score(e_ij) - depends on the edge only
            if self.use_edge_attr:
                self.kernel_edge_score_in = self.add_weight(name='kernel_edge_score_in', shape=(EF_shape[-1], 1),
                                                            initializer='uniform', trainable=True)
                self.kernel_edge_score_out = self.add_weight(name='kernel_edge_score_out', shape=(EF_shape[-1], 1),
                                                             initializer='uniform', trainable=True)
        super(GraphConvVertexLayer, self).build(input_shape)

    def call(self, x):
        """
        Call result
        :param x: vertex features [n_vertex, nvfeats], edge_features [n_edge, nefeats],
                  G [n_vertex, n_edges], H [n_vertex, n_edges], -- edge incidence matrices
                  A [n_vertex, n_vertex] -- adjacency matrix,
                  D_in [1, n_vertex], D_out [n_vertex, 1] -- in- and out-degree
        :return:
        """
        assert isinstance(x, (list, tuple))
        if self.use_edge_attr:
            VF, EF, G, H, A = x
        else:
            VF, G, H, A = x
        z = tf.matmul(VF, self.kernel_vertex_self)  # simple part
        if self.use_bias:
            z += self.bias
        if self.attention:
            vertex_query_in = tf.matmul(VF, self.kernel_vertex_query_in)
            vertex_query_out = tf.matmul(VF, self.kernel_vertex_query_out)
            vertex_key_in = tf.matmul(VF, self.kernel_vertex_key_in)
            vertex_key_out = tf.matmul(VF, self.kernel_vertex_key_out)
            # Scores as dot products of query and key
            vertex_score_in = tf.matmul(vertex_query_in, vertex_key_in, transpose_b=True)  # [n_vertex, n_vertex]
            vertex_score_out = tf.matmul(vertex_query_out, vertex_key_out, transpose_b=True)  # [n_vertex, n_vertex]
            # Edge scores as functions of edge (dot product of constant query and key)
            if self.use_edge_attr:
                edge_score_in_e = tf.matmul(self.kernel_edge_score_in, EF, transpose_a=True,
                                            transpose_b=True)  # [1, n_edges]
                edge_score_out_e = tf.matmul(self.kernel_edge_score_out, EF, transpose_a=True,
                                             transpose_b=True)  # [1, n_edges]
                edge_score_in = tf.matmul(G, H * edge_score_in_e, transpose_b=True)  # [n_vertex, n_vertex]
                edge_score_out = tf.matmul(G * edge_score_out_e, H, transpose_b=True)  # [n_vertex, n_vertex]
            # Adjacency matrix A gives us a mask
            s = vertex_score_in
            if self.use_edge_attr:
                s += edge_score_in
            s = tf.where(A > 0., s,
                         tf.math.reduce_min(s) * tf.ones_like(s))  # set minimal value for scores that will not matter
            s -= tf.math.reduce_max(s, axis=-2,
                                    keepdims=True)  # for numerical stability; this is why we did previous line
            s = tf.exp(s) * A  # here some will be zeroed
            # softmax over columns - upstream neighbors
            weight_in = tf.math.divide_no_nan(s, tf.reduce_sum(s, axis=-2, keepdims=True))
            s = vertex_score_out
            if self.use_edge_attr:
                s += edge_score_out
            s = tf.where(A > 0., s,
                         tf.math.reduce_min(s) * tf.ones_like(s))  # set minimal value for scores that will not matter
            s -= tf.math.reduce_max(s, axis=-1,
                                    keepdims=True)  # for numerical stability; this is why we did previous line
            s = tf.exp(s) * A  # here some will be zeroed
            # softmax over rows - downstream neighbors
            weight_out = tf.math.divide_no_nan(s, tf.reduce_sum(s, axis=-1, keepdims=True))
        else:
            weight_in = tf.math.divide_no_nan(A, tf.reduce_sum(A, axis=-2, keepdims=True))
            weight_out = tf.math.divide_no_nan(A, tf.reduce_sum(A, axis=-1, keepdims=True))

        z += tf.matmul(weight_in, tf.matmul(VF, self.kernel_vertex_value_in), transpose_a=True)
        z += tf.matmul(weight_out, tf.matmul(VF, self.kernel_vertex_value_out))
        if self.use_edge_attr:
            weight_edge_in = tf.matmul(weight_in, G) * H
            weight_edge_out = tf.matmul(weight_out, H) * G
            z += tf.matmul(weight_edge_in, tf.matmul(EF, self.kernel_edge_value_in))
            z += tf.matmul(weight_edge_out, tf.matmul(EF, self.kernel_edge_value_out))
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
