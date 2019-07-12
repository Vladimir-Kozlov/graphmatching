import tensorflow as tf
keras = tensorflow.keras


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

