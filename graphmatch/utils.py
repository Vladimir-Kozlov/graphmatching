import numpy as np

def principal_eig(M, num_iter=100):
    # Computes principal eigenvector v of matrix M by power iteration
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


def edge_decompozition(A):
    # Constructs incidence matrices G and H from adjacency matrix A
    # Input: A: adjacency matrix of a directed graph (square, of 0s and 1s)
    # Output: G, H: incidence matrices: {0, 1}^{n x c}, where c is the number of edges
    #         G[i, k] = 1 if edge k starts in vertex i, 0 otherwise
    #         H[i, k] = 1 if edge k ends in vertex i, 0 otherwise
    #         G x H^T = A
    # G and H are full matrices, but can be made sparce
    h, w = A.shape
    assert w == h
    assert np.all(A == 0 || A == 1)
    c = np.sum(A) # number of edges
    G = np.zeros(shape=(w, c))
    H = np.zeros(shape=(h, c))

    k = 0
    for i in range(h):
        for j in range(w):
            if A[i, j] == 1:
                G[i, k] = 1
                H[j, k] = 1
                k++

    return G, H


def edge_composition(G, H):
    # Constructs adjacency matrix A from incidence matrices G and H
    h, c1 = G.shape
    w, c2 = H.shape
    assert h == w
    assert c1 == c2
    return np.matmul(G, H.T)