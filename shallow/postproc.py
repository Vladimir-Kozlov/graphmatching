import numpy as np
from . import utils

def discr_assignment(X):
	# Matcher output discretization by linear assignment maximization
	# Input: X: soft matching matrix produced by matcher
	# Output: V: {0, 1} matching matrix, obtained by solving 
	#            linear assignment problem (maximization) with X as weights
	return utils.hungarian(X, goal='max')


def discr_max(X, n=None):
	# Matcher output discretization by maximum value
	# Input: X: soft matching matrix produced by matcher
	# Output: V: {0, 1} matching matrix, with ones corresponding to maximum values in X
	if n is None:
		n = min(X.shape)
	assert n <= max(X.shape)
	Y = X.copy()
	Z = np.zeros(Y.shape)
	replace = np.min(Y) - 1
	for i in range(n):
		z = np.unravel_index(np.argmax(Y), Y.shape)
		Z[z] = 1
		Y[z[0], :] = replace
		Y[:, z[1]] = replace
	return Z
