import numpy as np
from . import utils


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
    if (C is None):
        # C sums over rows and columns of assignment matrix
        # dropping one row ensures that C is full-rank
        C = np.zeros((N1 + N2 - 1, N1 * N2))
        for i in range(N2 - 1):
            C[i, range(i * N1, (i + 1) * N1)] = 1
        for j in range(N1):
            C[range(N2 - 1, N1 + N2 - 1), range(j * N1, (j + 1) * N1)] = 1
    if (d is None):
        d = np.ones(N1 + N2 - 1)


    assert C.shape[0] == len(d)
    assert C.shape[1] == h
    k = C.shape[0]

    if np.all(d == 0):
        Ceq = C
    else:
        # place nonzero constraint last
        i = np.argmax(d != 0)
        C[-1, :], C[i, :] = np.copy(C[i, :]), np.copy(C[-1, :])
        d[-1], d[i] = np.copy(d[i]), np.copy(d[-1])
        Ceq = C[:-1, :] - np.expand_dims(d[:-1], axis=1) * C[-1, :] / d[-1]

    P = np.eye(N1 * N2) - np.dot(Ceq.T, np.dot(np.linalg.inv(np.dot(Ceq, Ceq.T)), Ceq))
    H = np.dot(P.T, np.dot(W, P))

    v = utils.principal_eig(H, num_iter=num_iter)
    if d[-1] != 0:
        # if np.dot(C[-1, :], v) == 0, then the problem is ill-posed and has no solution (refer to paper for reason why) 
        v /= np.dot(C[-1, :], v) * d[-1]
    return v.reshape((N1, N2), order='F')


def gagm(W, N1, N2, beta_range=np.logspace(start=-5., stop=5., num=11, endpoint=True, base=10.), num_sinkh_iter=100, pad=0):
    # Graduated Assignment Graph Matching
    # Gold, Rangarajan "A Graduated Assignment Algorithm for Graph Matching"
    # Input: W: pairwise affinity matrix
    #        N1, N2: number of vertices in graphs
    h, w = W.shape
    assert h == N1 * N2
    assert w == N1 * N2
    W = utils.symm(W)

    x = np.ones((N1 * N2, 1))
    u = np.zeros((N1 + 1, N2 + 1))

    for beta in beta_range:
        u[:N1, :N2] = np.dot(W, x).reshape((N1, N2), order='F')
        # Softassign - a soft version of Hungarian alogithm
        v = np.exp(beta * (u - np.max(u, axis=0, keepdims=True)))
        for k in range(num_sinkh_iter):
            # Sinkhorn iteration implemented here for numerical stability reasons
            v /= np.sum(v, axis=0, keepdims=True)
            v[:, N2] *= max(N1, N2) - N2 + pad + 1.
            v /= np.sum(v, axis=1, keepdims=True)
            v[N1, :] *= max(N1, N2) - N1 + pad + 1.
        x = v[:N1, :N2].reshape((N1 * N2, 1), order='F')
    return v[:N1, :N2]




def spgm(W, N1, N2, gamma=1., num_iter=100, num_pocs_iter=100, eps=1e-6):
    # Successive Projection Graph Matching
    # van Wyk, van Wyk, "A POCS-Based Graph Matching Algorithm"
    # Input: W: pairwise affinity matrix
    #        N1, N2: number of vertices in graphs
    #        eps: solution precision
    # Ouput: X: continuous doubly stochastic 'soft assignment' with size [N1, N2]

    h, w = W.shape
    assert h == N1 * N2
    assert w == N1 * N2
    assert N1 == N2 # due to equality constraints
    W = utils.symm(W)
    # projection of matrix X onto set of row-wise stochastic matrices
    def pocs1(X):
        for i in range(N1):
            p = np.copy(X[i, :])
            f = p.sum()
            s = N2
            idx = np.argsort(p)
            for j in range(N2):
                p[idx[j]] += (1. - f) / s
                if p[idx[j]] < 0:
                    p[idx[j]] = 0
                    f -= X[i, idx[j]]
                    s-= 1
            X[i, :] = p
        return X
    # projection of matrix X onto set of column-wise stochastic matrices
    def pocs2(X):
        for j in range(N2):
            p = np.copy(X[:, j])
            f = p.sum()
            s = N1
            idx = np.argsort(p)
            for i in range(N1):
                p[idx[i]] += (1. - f) / s
                if p[idx[i]] < 0:
                    p[idx[i]] = 0
                    f -= X[idx[i], j]
                    s-= 1
            X[:, j] = p
        return X

    x = np.random.uniform(low=0., high=1., size=(w, 1)) / 2. + .5
    for k in range(num_iter):
        x0 = np.copy(x)
        u = np.dot(W, x)
        x += gamma * u / np.linalg.norm(u)
        v = x.reshape((N1, N2), order='F')
        for p in range(num_pocs_iter):
            v = pocs1(v)
            v = pocs2(v)
            if np.all(v.sum(axis=0) == 1.) and np.all(v.sum(axis=1) == 1.):
                break
        x = v.reshape((N1 * N2, 1), order='F')
        if np.linalg.norm(x - x0) < eps:
            break
    return x.reshape((N1, N2), order='F')


def ipfp(W, N1, N2, num_iter=100, eps=1e-6):
    # Integer Projected Fixed Point
    # Leordeanu, Hebert, Sukthankar "An Integer Projected Fixed Point Method for Graph Matching and MAP Inference"
    # Input: W: pairwise affinity matrix
    #        N1, N2: number of vertices in graphs
    # Ouput: X: continuous doubly stochastic 'soft assignment' with size [N1, N2]
    h, w = W.shape
    assert h == N1 * N2
    assert w == N1 * N2
    assert N1 == N2 # due to equality constraints
    W = utils.symm(W)

    x = np.ones((N1, N2))
    x = utils.sinkhorn(x).reshape((N1 * N2, 1), order='F')
    x0 = np.copy(x)
    s = np.dot(x0.T, np.dot(W, x0))

    for k in range(num_iter):
        u = np.dot(W, x)
        b = utils.hungarian(u.reshape((N1, N2), order='F'), goal='max').reshape((N1*N2, 1), order='F');
        c = np.dot(b.T - x.T, u)
        d = np.dot(b.T - x.T, np.dot(W, b)) - c
        xp = np.copy(x)
        if d >= 0:
            x = b
        else:
            r = min(-c / d, 1)
            x += r * (b - x)
        if np.dot(b.T, np.dot(W, b)) >= s:
            s = np.dot(b.T, np.dot(W, b))
            x0 = b
        if np.linalg.norm(x - xp) < eps:
            break
    return x0.reshape((N1, N2), order='F')

