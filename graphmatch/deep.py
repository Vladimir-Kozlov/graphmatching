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


def power_iter_factorized(Mp, Mq, G1, H1, G2, H2, max_iter=100, eps_iter=1e-6):
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


class AffinityVertex(keras.layers.Layer):
    # Layer that calculates vertex affinity matrix from vertex feature vectors
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(AffinityVertex, self).build(input_shape)
        
    def call(self, x):
        assert isinstance(x, list)
        # Input: U_l, U_r: matrices of vertex features
        #        of shape [n1, vertex feature vector length] and [n2, VFVL] respectively
        # Output: Mp = U_l * U_r^T
        U_l, U_r = x
        return tf.linalg.matmul(U_l, U_r, transpose_b=True)
    
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_u1, shape_u2 = input_shape
        return (shape_u1[0], shape_u2[0])
    
    
class AffinityEdge(keras.layers.Layer):
    # Layer that calculates edge affinity matrix from edge feature vectors and incidence matrices
    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.w1 = self.add_weight(name='weight1',
                                  shape=(input_shape[0][-1].value, input_shape[1][-1].value),
                                  initializer='uniform',
                                  trainable=True)
        self.w2 = self.add_weight(name='weight2',
                                  shape=(input_shape[0][-1].value, input_shape[1][-1].value),
                                  initializer='uniform',
                                  trainable=True)
        # kernel should be block-symmetric matrix with positive elements
        # the less weights to change the better, right?
        # diagonal elements are calculated twice
        L1 = tf.linalg.band_part(self.w1, 0, -1)
        L2 = tf.linalg.band_part(self.w2, 0, -1)
        self.L1 = tf.math.maximum(0., L1 + tf.linalg.transpose(L1))
        self.L2 = tf.math.maximum(0., L2 + tf.linalg.transpose(L2))
        super(AffinityEdge, self).build(input_shape)
        
    def call(self, x):
        assert isinstance(x, list)
        # Input: P_l, P_r: matrices of vertex features
        #                  of shape [m1, edge feature vector length] and [m2, EFVL] respectively
        #        G_l, G_r: matrices of edge incidence: G[i, j] = [edge j starts in vertex i]
        #                  of shape [n1, m1] and [n2, m2] respectively
        #        H_l, H_r: matrices of edge incidence: H[i, j] = [edge j ends in vertex i]
        #                  of shape [n1, m1] and [n2, m2] respectively
        # Output: Mq: matrix of edge affinities
        P_l, P_r, G_l, G_r, H_l, H_r = x
        FG_l = tf.linalg.matmul(G_l, P_l, transpose_a=True)
        FH_l = tf.linalg.matmul(H_l, P_l, transpose_a=True)
        FG_r = tf.linalg.matmul(G_r, P_r, transpose_a=True)
        FH_r = tf.linalg.matmul(H_r, P_r, transpose_a=True)
        # quadratic form with block-symmetric matrix
        def m(x, l, y):
            # x * l * y^T
            # due to broadcasting issues with tf.matmul, we use Keras dot
            return tf.linalg.matmul(keras.backend.dot(x, l), y, transpose_b=True)
        return m(FG_l, self.L1, FG_r) + m(FH_l, self.L1, FH_r) + m(FG_l, self.L2, FH_r) + m(FH_l, self.L2, FG_r)
    
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return (input_shape[0][0], input_shape[1][0])


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

    # need to reorder arguments either in AffinityEdge call or in power_iter_factorized; the latter is preferrable
    Mp = AffinityVertex()([featv_1, featv_2])
    Mq = AffinityEdge()([feate_1, feate_2, g1_input, g2_input, h1_input, h2_input])
    pi = keras.layers.Lambda(lambda x: power_iter_factorized(*x))([Mp, Mq, g1_input, h1_input, g2_input, h2_input])
    sl = keras.layers.Lambda(sinkhorn_loop)(pi)

    return keras.models.Model(inputs=[img1_input, idx1_input, g1_input, h1_input, img2_input, idx2_input, g2_input, h2_input], outputs=[sl])