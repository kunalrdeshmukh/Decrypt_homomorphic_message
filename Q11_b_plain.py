import nltk
import unicodedata
import re
import numpy as np
import math
import sys
import datetime
from nltk.corpus import brown
import random_generator


text_file = open("PlainData.txt", "r")
data = text_file.read()
text_file.close()

V = "abcdefghijklmnopqrstuvwxyz"
V = list(V)
M = len(V)
N = 2
pi = [0.5786,0.4214]
A = [[0.47467,0.52533],[0.51656,0.48344]]
B = random_generator.generateRandMat(N,M)
O = list(data) 
minIters = 1000 # Min number of re-estimation iterations
epsi = 0.00001
T = len(data)
iters = 0
oldLogProb = -sys.maxint - 1
c = [None]*T
a = np.zeros((T,N))
b = np.zeros((T,N))
y = np.zeros((T,N))
y_di = np.zeros((T,N,N))
print("Data imported and values initialyzed.")



def aPass():
	# Computer a0(i)
	# print("in a pass")
	c[0] = 0
	for i in range (0,N):
		a[0][i]=pi[i]*B[i][V.index(O[0])]
		c[0] = c[0] + a[0][i]
	# scale the a0(i)
	c[0] = 1/c[0]
	for i in range (0,N):
		a[0][i] = c[0]*a[0][i]
	# compute at(i)
	for t in range (1,T):
		c[t]=0
		for i in range (0,N):
			a[t][i] = 0
			for j in range (0,N):
				a[t][i] = a[t][i] + a[t-1][j] * A[j][i]
			a[t][i] = a[t][i]*B[i][V.index(O[t])]
			c[t] = c[t] + a[t][i]
		# scale at(i)
		c[t] = 1/c[t]
		for i in range (0,N):
			a[t][i] = c[t]*a[t][i]
	# print("a pass ends")

def bPass():
	# Let b(t-1)(i) = 1 scaled by c(t-1)
	# print("in b pass")
	for i in range (0,N):
		b[T-1][i] = c[T-1]
	# print(b[T-1])
	# beta pass
	for t in range (T-2,-1,-1):
		for i in range(0,N):
			b[t][i] = 0
			for j in range(0,N):
				b[t][i] = b[t][i] + A[i][j]*B[j][V.index(O[t+1])]*b[t+1][j]
			# scale B(t)(i) with same scale factor as a(t)(i))
			b[t][i] = c[t]*b[t][i]
	# print("b pass ends")

def yPass():
	# print("in y pass")
	for t in range(0,T-1):
		denom = 0
		for i in range(0,N):
			for j in range(0,N):
				denom = denom + a[t][i]*A[i][j]*B[j][V.index(O[t+1])]*b[t+1][j]
		for i in range(0,N):
			y[t][i] = 0
			for j in range(0,N):
				y_di[t][i][j] = (a[t][i]*A[i][j]*B[j][V.index(O[t+1])]*b[t+1][j])/denom 
				y[t][i] = y[t][i] + y_di[t][i][j]
	# Special case for y(t-1)(i)
	denom = 0
	for i in range(0,N):
		denom = denom + a[T-1][i]
	for i in range(0,N):
		y[T-1][i] = a[T-1][i]/denom
	# print("y pass ends")


def reEstimate():
	# print("reestimating")
	#reestimate pi
	for i in range(0,N):
		pi[i] = y[0][i]

	#reestimate A
	for i in range(0,N):
		for j in range(0,N):
			numer = 0
			denom = 0
			for t in range(0,T-1):
				numer += y_di[t][i][j]
				denom += y[t][i]
			# print("numer :",numer)
			# print("denom :",denom)
			A[i][j] = numer/denom

	#re-estimate B
	for i in range(0,N):
		for j in range(0,M):
			numer = 0
			denom = 0
			for t in range(0,T):
				if (V.index(O[t])==j):
					numer += y[t][i]
				denom += y[t][i]
			B[i][j] = numer/denom
	# print("end re-estimate")

def logP():
	logProb = 0.0
	for i in range(0,T):
		logProb=logProb*1.0+math.log(c[i],10)
	logProb = -logProb
	return logProb

def printabpi():
	print("Iteration number%d",iters)
	print("Matrix A:")
	print(A)
	print('')
	print("Matrix B:")
	print((np.array(B)).transpose())
	print('')
	print("Matrix pi:")
	print(pi)
	print('')
	print("sum of B matrix:")
	print(sum(B[1])*1.0)
	print("At time:")
	print(datetime.datetime.time(datetime.datetime.now()))

# oldLogProb = 0
# logProb = logP()
iters = 0
printabpi()
# print(datetime.datetime.time(datetime.datetime.now()))
iters += 1
# delta = abs(logProb-oldLogProb)
while (iters < minIters ):
	# oldLogProb = -logProb
	aPass()
	bPass()
	yPass()
	reEstimate()
	iters+=1
	# logP()
	printabpi()
