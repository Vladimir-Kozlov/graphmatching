import tensorflow as tf
from tensorflow import keras
import numpy as np


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

