import tensorflow as tf
from tensorflow import keras
import numpy as np


def idxtransform(idx, scale=2**5):
    # Transform coordinates of image points to index feature vectors in high layers
    # Input: idx: point coordinates of shape [batch size, number of points, 2]
    #        scale: scaling parameter of feature extractor
    #               for instance, we use MobileNetV2 which halves input image 5 times, so default scaling parameter is 32
    # Output: scaled indices with added leading batch number
    assert isinstance(scale, (int, list, tuple))
    if isinstance(scale, int):
        x = idx / scale
    if isinstance(scale, (list, tuple)):
        assert len(scale) == 2
        x = idx / np.reshape(np.array(scale, dtype=np.int32), (1, 2))
    r = tf.reshape(tf.range(tf.shape(idx)[0]), [-1, 1, 1])
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


class VertexAffinityLayer(keras.layers.Layer):
    # Layer that calculates vertex affinity matrix from vertex feature vectors
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(VertexAffinityLayer, self).build(input_shape)
        
    def call(self, x):
        assert isinstance(x, list)
        # Input: U_l, U_r: matrices of vertex features
        #        of shape [n1, vertex feature vector length] and [n2, VFVL] respectively
        # Output: Mp = U_l * U_r^T
        U_l = tf.math.l2_normalize(x[0], axis=-1)
        U_r = tf.math.l2_normalize(x[1], axis=-1)
        return tf.linalg.matmul(U_l, U_r, transpose_b=True)
    
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_u1, shape_u2 = input_shape
        return (shape_u1[0], shape_u2[0])


class EdgeAffinityLayer(keras.layers.Layer):
    # Layer that calculates edge affinity matrix from edge feature vectors and incidence matrices
    def __init__(self, eps=0., **kwargs):
        self.eps = eps
        super(EdgeAffinityLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.w1 = self.add_weight(name='weight1',
                                  shape=(input_shape[0][-1].value, input_shape[1][-1].value),
                                  initializer=keras.initializers.RandomUniform(minval=0., maxval=.5),
                                  trainable=True)
        self.w2 = self.add_weight(name='weight2',
                                  shape=(input_shape[0][-1].value, input_shape[1][-1].value),
                                  initializer=keras.initializers.RandomUniform(minval=0., maxval=.5),
                                  trainable=True)
        # kernel should be block-symmetric matrix with positive elements
        # this ensures symmetry strict positivity
        self.L1 = tf.maximum(self.w1 + tf.linalg.transpose(self.w1), self.eps)
        self.L2 = tf.maximum(self.w2 + tf.linalg.transpose(self.w2), self.eps)
        super(EdgeAffinityLayer, self).build(input_shape)
        
    def call(self, x):
        assert isinstance(x, list)
        # Input: P_l, P_r: matrices of vertex features (to combine into edge features)
        #                  of shape [n1, edge feature vector length] and [n2, EFVL] respectively
        #        G_l, G_r: matrices of edge incidence: G[i, j] = [edge j starts in vertex i]
        #                  of shape [n1, m1] and [n2, m2] respectively
        #        H_l, H_r: matrices of edge incidence: H[i, j] = [edge j ends in vertex i]
        #                  of shape [n1, m1] and [n2, m2] respectively
        # Output: Mq: matrix of edge affinities
        _, _, G_l, G_r, H_l, H_r = x
        P_l = tf.math.l2_normalize(x[0], axis=-1)
        P_r = tf.math.l2_normalize(x[1], axis=-1)
        FG_l = tf.linalg.matmul(G_l, P_l, transpose_a=True)
        FH_l = tf.linalg.matmul(H_l, P_l, transpose_a=True)
        FG_r = tf.linalg.matmul(G_r, P_r, transpose_a=True)
        FH_r = tf.linalg.matmul(H_r, P_r, transpose_a=True)
        # quadratic form with block-symmetric matrix
        def m(x, l, y):
            # x * l * y^T
            return tf.linalg.matmul(tf.tensordot(x, l, axes=1), y, transpose_b=True)
        return m(FG_l, self.L1, FG_r) + m(FH_l, self.L1, FH_r) + m(FG_l, self.L2, FH_r) + m(FH_l, self.L2, FG_r)
    
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (input_shape[0][0], input_shape[1][0])


class VertexAffinityCosineLayer(keras.layers.Layer):
    # Layer that calculates vertex affinity matrix from vertex feature vectors
    def __init__(self, transform_dim=1, **kwargs):
        self.transform_dim = transform_dim
        super(VertexAffinityCosineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.transform_matrix = self.add_weight(name='transform_matrix',
                                                shape=(input_shape[0][-1].value, self.transform_dim),
                                                initializer='orthogonal', trainable=True)
        super(VertexAffinityCosineLayer, self).build(input_shape)
        
    def call(self, x):
        assert isinstance(x, list)
        # Input: V_l, Vmask_l, V_r, Vmask_r
        #        V_l, V_r: matrices of vertex features of shape [n1, k] and [n2, k] respectively
        #        V_mask: vector of 0 and 1 to indicate whether the vertex is meaningful or is a dummy
        # Output: Mp: cosine similarities between V_l @ transform_matrix and V_r @ transform_matrix,
        #             normalized to [0, 1]; if vertex is dummy, then similarities are 0.
        
        U_l = tf.math.l2_normalize(tf.tensordot(x[0], self.transform_matrix, axes=1), axis=-1)
        U_r = tf.math.l2_normalize(tf.tensordot(x[2], self.transform_matrix, axes=1), axis=-1)

        Vmask_l = tf.expand_dims(x[1], axis=-1)
        Vmask_r = tf.expand_dims(x[3], axis=-1)
        U_mask = tf.linalg.matmul(Vmask_l, Vmask_r, transpose_b=True)

        return U_mask * (tf.linalg.matmul(U_l, U_r, transpose_b=True) + 1.) / 2.
    
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (input_shape[0][0], input_shape[2][0])


class EdgeAffinityCosineLayer(keras.layers.Layer):
    # Layer that calculates edge affinity matrix from edge feature vectors
    def __init__(self, transform_dim=1, **kwargs):
        self.transform_dim = transform_dim
        super(EdgeAffinityCosineLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.transform_matrix = self.add_weight(name='transform_matrix',
                                                shape=(input_shape[0][-1].value, self.transform_dim),
                                                initializer='orthogonal', trainable=True)
        super(EdgeAffinityCosineLayer, self).build(input_shape)
        
    def call(self, x):
        assert isinstance(x, list)
        # Input: E_l, E_r: matrices of edge features
        #        of shape [m1, edge feature vector length] and [m2, EFVL] respectively
        #        Masking is done by incidence matrices
        # Output: Mq: cosine similarities between E_l @ transform_matrix and E_r @ transform_matrix, normalized to [0, 1]
        
        E_l = tf.math.l2_normalize(tf.tensordot(x[0], self.transform_matrix, axes=1), axis=-1)
        E_r = tf.math.l2_normalize(tf.tensordot(x[1], self.transform_matrix, axes=1), axis=-1)

        return (tf.linalg.matmul(E_l, E_r, transpose_b=True) + 1.) / 2.
    
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (input_shape[0][0], input_shape[1][0])


def power_iter_factorized(Mp, Mq, G1, G2, H1, H2, max_iter=100, eps_iter=1e-6):
    # Power iteration for affinity matrix
    # Here we take advantage of affinity matrix factorization
    # Input: Mp: node affinity matrix of shape [n1, n2]
    #        Mq: edge affinity matrix of shape [m1, m2]
    #        G1, H1: incidence matrices for graph 1: 
    #                G[i, j] = [edge j starts at vertex i], of shape [n, m]
    #                H[i, j] = [edge j ends at vertex i], of shape [n, m]
    #        G2, H2: incidence matrices for graph 2
    #        max_iter: maximum number of iterations in a loop
    #        eps_iter: acceptable precision, looping stops when change is less than eps_iter
    # Output: leading eigenvector of affinity matrix M, with shape [n1, n2]
    def mtr(v, P1, Q1, P2, Q2):
        # Q1^T * v * Q2
        x1 = tf.linalg.matmul(Q1, v, transpose_a=True)
        x2 = tf.linalg.matmul(x1, Q2)
        
        y = Mq * x2
        
        # P1 * y * P2^T
        z1 = tf.linalg.matmul(P1, y)
        z2 = tf.linalg.matmul(z1, P2, transpose_b=True)
        return z2
    def power_iter(v):
        z = (mtr(v, G1, H1, G2, H2) + mtr(v, H1, G1, H2, G2)) / 2.
        t = z + Mp * v
        return tf.math.l2_normalize(t, axis=[-2, -1])
    def cond(v, i, d):
        return tf.math.logical_and(i < max_iter, d)
    def body(v, i, d):
        u = power_iter(v)
        return u, i + 1, tf.math.reduce_any(tf.linalg.norm(u - v, axis=(-2, -1)) >= eps_iter)
    
    i = tf.constant(0)
    d = tf.constant(True)
    v = tf.ones(tf.shape(Mp))
    return tf.while_loop(cond=cond, body=body, loop_vars=(v, i, d), maximum_iterations=max_iter, swap_memory=True)[0]


class PowerIterationLayer(keras.layers.Layer):
    def __init__(self, max_iter=100, eps_iter=1e-6, **kwargs):
        self.max_iter = max_iter
        self.eps_iter = eps_iter
        super(PowerIterationLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(PowerIterationLayer, self).build(input_shape)
    
    def call(self, x):
        assert isinstance(x, list)
        return power_iter_factorized(*x, max_iter=self.max_iter, eps_iter=self.eps_iter)
    
    def compute_output_shape(self, input_shape):
        # output shape is the same as Mp
        assert isinstance(input_shape, list)
        return (input_shape[0][0], input_shape[0][1])
    
    
def sinkhorn_loop(x, max_iter=100, eps_iter=1e-6):
    # Sinkhorn-Knopp algorithm for transforming matrix with nonnegative elements into doubly stochastic one
    # Input: x: matrix to be processed
    #        max_iter: maximum number of iterations in a loop
    #        eps_iter: acceptable precision, looping stops when change is less than eps_iter
    # Output: doubly stochastic matrix
    def sinkhorn_iter(x):
        u = tf.math.reduce_sum(x, axis=-2, keepdims=True)
        h = tf.div_no_nan(x, u)
        v = tf.math.reduce_sum(h, axis=-1, keepdims=True)
        return tf.div_no_nan(h, v)
    def cond(x, i, d):
        return tf.math.logical_and(i < max_iter, d)
    def body(x, i, d):
        y = sinkhorn_iter(x)
        return y, i + 1, tf.math.reduce_any(tf.linalg.norm(y - x, axis=(-2, -1)) >= eps_iter)
    
    i = tf.constant(0)
    d = tf.constant(True)
    return tf.while_loop(cond=cond, body=body, loop_vars=(x, i, d), maximum_iterations=max_iter, swap_memory=True)[0]


class SinkhornIterationLayer(keras.layers.Layer):
    def __init__(self, max_iter=100, eps_iter=1e-6, **kwargs):
        self.max_iter = max_iter
        self.eps_iter = eps_iter
        super(SinkhornIterationLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(SinkhornIterationLayer, self).build(input_shape)
    
    def call(self, x):
        return sinkhorn_loop(x, max_iter=self.max_iter, eps_iter=self.eps_iter)
    
    def compute_output_shape(self, input_shape):
        # does not change shape of input
        return input_shape

