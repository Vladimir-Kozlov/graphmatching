import numpy as np
import utils

def discr_assignment(X):
	# Matcher output discretization by linear assignment maximization
	# Input: X: soft matching matrix produced by matcher
	# Output: V: {0, 1} matching matrix, obtained by solving 
	#            linear assignment problem (maximization) with X as weights
	return utils.hungarian(X, goal='max')