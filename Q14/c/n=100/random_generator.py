import random
import numpy as np


def generateRandMat(N,M):
	t = 0
	while (t != N):
		t = 0
		A = x = (1.0/(M*15))*np.random.randn(N, M)+(1.0/M)
		for i in range (0,N):
			A[i][M-1] = 1-(sum(A[1])-A[1][M-1])
			if (A[i][M-1] > 0.0):
				t += 1
			else:
				t -= 1
	return A
