import numpy as np
import utils


def spectral_matching(W, N1, N2, num_iter=100):
	# Spectral Matching algorithm
	# Leordeanu, Hebert "A Spectral Technique for Correspondence Problems Using Pairwise Constraints"
	# Input: W: pairwise affinity matrix with size [N1*N2, N1*N2]
	#        N1, N2: number of vertices in graphs
	#        num_iter: number of iterations in power iteration method
	# Output: X: assignment matrix with size [N1, N2]
	h, w = W.shape
	assert h == N1 * N2
	assert w == N1 * N2
	v = utils.principal_eig(W, num_iter=num_iter)
	return utils.hungarian(v.reshape((N1, N2)), goal='max')

	