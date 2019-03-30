import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np


def idxtransform(idx, scale=2**5):
    # Transform coordinates of image points to index feature vectors in high layers
    # Input: idx: point coordinates of shape [batch size, number of points, 2]
    #        scale: scaling parameter of feature extractor
    #               for instance, we use MobileNetV2 which halves input image 5 times, so default scaling parameter is 32
    # Output: scaled indices with added leading batch number
    x = idx / scale
    r = tf.reshape(tf.range(tf.shape(idx)[0]), [-1, 1, 1])
    r = tf.tile(r, tf.concat([[1], tf.shape(idx)[1:-1], [1]], axis=0))
    return tf.concat([r, x], axis=-1)


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
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.w1 = self.add_weight(name='weight1',
                                  shape=(input_shape[0][-1].value, input_shape[1][-1].value),
                                  initializer=keras.initializers.RandomUniform(minval=0., maxval=.5),
                                  trainable=True,
                                  constraint=keras.constraints.NonNeg())
        self.w2 = self.add_weight(name='weight2',
                                  shape=(input_shape[0][-1].value, input_shape[1][-1].value),
                                  initializer=keras.initializers.RandomUniform(minval=0., maxval=.5),
                                  trainable=True,
                                  constraint=keras.constraints.NonNeg())
        # kernel should be block-symmetric matrix with positive elements
        # this ensures symmetry and the fact that all weights are accessible in backprop
        self.L1 = self.w1 + tf.linalg.transpose(self.w1)
        self.L2 = self.w2 + tf.linalg.transpose(self.w2)
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
        return t / tf.linalg.norm(t, axis=(-2, -1), keepdims=True)
    def cond(v, i, d):
        return tf.math.logical_and(i < max_iter, d)
    def body(v, i, d):
        u = power_iter(v)
        return u, i + 1, tf.math.reduce_any(tf.linalg.norm(u - v, axis=(-2, -1)) >= eps_iter)
    
    i = tf.constant(0)
    d = tf.constant(True)
    v = tf.ones(tf.shape(Mp))
    return tf.while_loop(cond=cond, body=body, loop_vars=(v, i, d), maximum_iterations=max_iter)[0]


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
    return tf.while_loop(cond=cond, body=body, loop_vars=(x, i, d), maximum_iterations=max_iter)[0]


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


def deep_graph_matching_model():
    img1_input = keras.layers.Input(shape=(None, None, 3), dtype='float32')
    img2_input = keras.layers.Input(shape=(None, None, 3), dtype='float32')
    idx1_input = keras.layers.Input(shape=(None, 2), dtype='int32')
    idx2_input = keras.layers.Input(shape=(None, 2), dtype='int32')
    g1_input = keras.layers.Input(shape=(None, None), dtype='float32')
    g2_input = keras.layers.Input(shape=(None, None), dtype='float32')
    h1_input = keras.layers.Input(shape=(None, None), dtype='float32')
    h2_input = keras.layers.Input(shape=(None, None), dtype='float32')

    mnv2 = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(None, None, 3),
                                                       alpha=1.,
                                                       depth_multiplier=1, 
                                                       include_top=False, 
                                                       weights='imagenet', 
                                                       pooling=None)
    mnv2_1 = mnv2(img1_input)
    mnv2_2 = mnv2(img1_input)

    idx_vert = keras.layers.Lambda(idxtransform, arguments={'scale': 2**5})
    idx_edge = keras.layers.Lambda(idxtransform, arguments={'scale': 2**5})
    idxv_1 = idx_vert(idx1_input)
    idxe_1 = idx_edge(idx1_input)
    idxv_2 = idx_vert(idx2_input)
    idxe_2 = idx_edge(idx2_input)

    fmap_idx = keras.layers.Lambda(lambda x: tf.gather_nd(*x))
    featv_1 = fmap_idx([mnv2_1, idxv_1])
    feate_1 = fmap_idx([mnv2_1, idxe_1])
    featv_2 = fmap_idx([mnv2_2, idxv_2])
    feate_2 = fmap_idx([mnv2_2, idxe_2])

    # ReLU added in order to produce nonnegative affinity
    Mp = keras.layers.ReLU()(VertexAffinityLayer()([featv_1, featv_2]))
    Mq = keras.layers.ReLU()(EdgeAffinityLayer()([feate_1, feate_2, g1_input, g2_input, h1_input, h2_input]))
    pi = PowerIterationLayer()([Mp, Mq, g1_input, g2_input, h1_input, h2_input])
    sl = SinkhornIterationLayer()(pi)

    return keras.models.Model(inputs=[img1_input, idx1_input, g1_input, h1_input, img2_input, idx2_input, g2_input, h2_input], outputs=[sl])
