import numpy as np
import utils


def sm(W, N1, N2, num_iter=100):
    # Spectral Matching
    # Leordeanu, Hebert "A Spectral Technique for Correspondence Problems Using Pairwise Constraints"
    # Input: W: pairwise affinity matrix with size [N1*N2, N1*N2]
    #        N1, N2: number of vertices in graphs
    #        num_iter: number of iterations in power iteration method
    # Output: X: continuous 'soft assignment' with size [N1, N2]
    h, w = W.shape
    assert h == N1 * N2
    assert w == N1 * N2
    v = utils.principal_eig(W, num_iter=num_iter)
    return v.reshape((N1, N2), order='F')

def smac(W, N1, N2, C=None, d=None, num_iter=100):
    # Spectral Graph Matching with Affine Constraints
    # Cour, Srinivasan, Shi "Balanced Graph Matching"
    # Input: W: pairwise affinity matrix
    #        N1, N2: number of vertices in graphs
    #        C, d: equality constraints C * vec(X) = d
    #              if not passed, then defaults to doubly stochastic, requiring N1 == N2
    #              C: full-rank with shape [k, N1 * N2], k < N1 * N2
    #              d: vector of length k
    #        num_iter: number of iterations for power principal eigenvector computation
    # Output: X: continuous 'soft assignment' with size [N1, N2], satisfying passed constraints
    h, w = W.shape
    assert h == N1 * N2
    assert w == N1 * N2

    # if no constraints are provided, we construct our own (doubly stochastic)
    # in that case, we assume the assignment matrix is square, otherwise no such matrix exist
    if (C is None) or (d is None):
        assert N1 == N2
        # C sums over rows and columns of assignment matrix
        # dropping one row ensures that C is full-rank
        C = np.zeros((N1 + N2 - 1, N1 * N2))
        for i in range(N2 - 1):
            C[i, range(i * N1, (i + 1) * N1)] = 1
        for j in range(N1):
            C[range(N2 - 1, N1 + N2 - 1), range(j * N1, (j + 1) * N1)] = 1
        d = np.ones(N1 + N2 - 1)

    assert C.shape[0] == len(d)
    assert C.shape[1] == h
    k = C.shape[0]

    if np.all(d == 0):
        Ceq = C
    else:
        # place nonzero constraint last
        i = np.argmax(d != 0)
        u = np.copy(C[i, :])
        C[i, :] = C[-1, :]
        C[-1, :] = u
        u = np.copy(d[i])
        d[i] = d[-1]
        d[-1] = u
        Ceq = C[:-1, :] - np.expand_dims(d[:-1], axis=1) * C[-1, :] / d[-1]
    P = np.eye(N1 * N2) - np.dot(Ceq.T, np.dot(np.linalg.inv(np.dot(Ceq, Ceq.T)), Ceq))
    H = np.dot(P.T, np.dot(W, P))

    v = utils.principal_eig(H, num_iter=num_iter)
    if d[-1] != 0:
        # if np.dot(C[-1, :], v) == 0, then the problem is ill-posed and has no solution (refer to paper for reason why) 
        v /= np.dot(C[-1, :], v) * d[-1]
    return v.reshape((N1, N2), order='F')


