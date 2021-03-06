import numpy as np
import scipy.optimize as opt

def principal_eig(M, num_iter=100):
    # Computes principal eigenvector v of matrix M by power iteration
    # Input: M: square matrix (with preferrably positive elements)
    #        num_iter: number of power iterations
    # Output: v: normalized principal eigenvector
    h, w = M.shape
    assert w == h
    # normalized random vector with positive elements
    v = np.random.uniform(low=0., high=1., size=(w, 1)) / 2. + .5
    v /= np.linalg.norm(v)
    # power iteration
    for k in range(num_iter):
        v = np.dot(M, v)
        v /= np.linalg.norm(v)

    return v


def edge_decompose(A):
    # Constructs incidence matrices G and H from adjacency matrix A
    # Input: A: adjacency matrix of a directed graph (square, of 0s and 1s)
    # Output: G, H: incidence matrices: {0, 1}^{n x c}, where c is the number of edges
    #         G[i, k] = 1 if edge k starts in vertex i, 0 otherwise
    #         H[i, k] = 1 if edge k ends in vertex i, 0 otherwise
    #         G x H^T = A
    # G and H are full matrices, but can be made sparce
    h, w = A.shape
    assert w == h
    assert np.all(np.logical_or(A == 0, A == 1))
    c = np.int(np.sum(A)) # number of edges
    G = np.zeros(shape=(w, c))
    H = np.zeros(shape=(h, c))

    k = 0
    for i in range(h):
        for j in range(w):
            if A[i, j] == 1:
                G[i, k] = 1
                H[j, k] = 1
                k += 1

    return G, H


def edge_compose(G, H):
    # Constructs adjacency matrix A from incidence matrices G and H
    h, c1 = G.shape
    w, c2 = H.shape
    assert h == w
    assert c1 == c2
    return np.matmul(G, H.T)


def symm(M):
    # Make square matrix symmetric
    h, w = M.shape
    assert h == w
    return (M + M.T) / 2.


def hungarian(M, goal='min'):
    # Wrapper for Hungarian algorithm from scipy.optimize
    if goal == 'max':
        M = -M

    ri, ci = opt.linear_sum_assignment(M)
    X = np.zeros(shape=M.shape)
    X[ri, ci] = 1
    return X

def sinkhorn(M, pad=0, num_iter=100):
    # Sinkhorn-Knopp algorithm for turning positive matrix into doubly stochastic
    # If matrix is square, it becomes doubly stochastic
    h, w = M.shape
    for k in range(num_iter):
        u = np.sum(M, axis=0, keepdims=True)
        M = ((max(h, w) + pad) * M) / (u * h)
        v = np.sum(M, axis=1, keepdims=True)
        M = ((max(h, w) + pad) * M) / (v * w)
    return M


def construct_affinity_matrix(Kp, Kq, A1, A2):
    G1, H1 = edge_decompose(A1)
    G2, H2 = edge_decompose(A2)
    u = np.squeeze(Kp.reshape((-1, 1), order='F'))
    v = np.squeeze(Kq.reshape((-1, 1), order='F'))
    X = np.kron(G2, G1)
    Y = np.kron(H2, H1)
    return np.diag(u) + np.dot(X, np.dot(np.diag(v), Y.T))