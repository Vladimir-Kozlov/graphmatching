import tensorflow as tf
keras = tf.keras


class SiameseOutputLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SiameseOutputLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SiameseOutputLayer, self).build(input_shape)

    def call(self, vertex_features):
        i1, i2 = vertex_features
        i1 = tf.expand_dims(i1, axis=-2)
        i2 = tf.expand_dims(i2, axis=-3)
        return tf.exp(-tf.reduce_sum((i1 - i2)**2, axis=-1, keepdims=False) / 2.)

    @staticmethod
    def compute_output_shape(input_shape):
        i1, i2 = input_shape
        return i1[-2], i2[-2]


def power_iter_factorized(Mp, Mq, G1, G2, H1, H2, ord=2, max_iter=100, eps_iter=1e-6):
    """
    Power iteration for affinity matrix which takes advantage of affinity matrix factorization
    :param Mp: node affinity matrix of shape [n1, n2]
    :param Mq: edge affinity matrix of shape [m1, m2]
    :param G1: G1[i, j] = [in graph 1, edge j starts at vertex i], of shape [n1, m1]
    :param G2: G2[i, j] = [in graph 2, edge j starts at vertex i], of shape [n2, m2]
    :param H1: H1[i, j] = [in graph 1, edge j ends at vertex i], of shape [n1, m1]
    :param H2: H2[i, j] = [in graph 2, edge j ends at vertex i], of shape [n2, m2]
    :param ord: norm order, accepts any value that tf.linalg.norm accepts
    :param max_iter: maximum number of iterations in a loop
    :param eps_iter: acceptable precision, looping stops when change is less than eps_iter
    :return: leading eigenvector of affinity matrix M, with shape [n1, n2]
    """
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
        # here, different norms can be used
        return tf.div_no_nan(t, tf.linalg.norm(t, ord=ord, axis=(-2, -1), keepdims=True))

    def cond(v, i, d):
        return tf.math.logical_and(i < max_iter, d)

    def body(v, i, d):
        u = power_iter(v)
        # here, a euclidean norm is used to calculate the difference before and after iteration
        return u, i + 1, tf.math.reduce_any(tf.linalg.norm(u - v, ord=2, axis=(-2, -1)) >= eps_iter)
    
    i = tf.constant(0)
    d = tf.constant(True)
    v = tf.ones(tf.shape(Mp))
    return tf.while_loop(cond=cond, body=body, loop_vars=(v, i, d), maximum_iterations=max_iter, swap_memory=True)[0]


class PowerIterationLayer(keras.layers.Layer):
    """
    input: Mp, Mq, G1, G2, H1, H2
    """
    def __init__(self, max_iter=100, eps_iter=1e-6, norm=2, **kwargs):
        self.max_iter = max_iter
        self.eps_iter = eps_iter
        self.norm = norm
        super(PowerIterationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, (list, tuple))
        super(PowerIterationLayer, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, (list, tuple))
        return power_iter_factorized(*x, ord=self.norm, max_iter=self.max_iter, eps_iter=self.eps_iter)

    @staticmethod
    def compute_output_shape(input_shape):
        # output shape is the same as Mp
        assert isinstance(input_shape, (list, tuple))
        return input_shape[0][0], input_shape[0][1]


def sinkhorn_loop(x, max_iter=100, eps_iter=1e-6):
    # Sinkhorn-Knopp algorithm for transforming matrix with non-negative elements into doubly stochastic one
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

    @staticmethod
    def compute_output_shape(input_shape):
        # does not change shape of input
        return input_shape
