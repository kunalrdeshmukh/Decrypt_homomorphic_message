import nltk
import unicodedata
import re
import numpy as np
import math
from nltk.corpus import brown
import random
import sys
import random_generator

data = "".join(brown.words())
data = data.lower() 
data =  re.sub('[^A-Za-z ]+', '',data)
data = re.sub('  +','',data)
data = data.encode("ascii")
data = data[:1000]
T = len(data)
V = "abcdefghijklmnopqrstuvwxyz"
N = 26

############### A matrix generated in 14a######

A = np.zeros((26,26))
for i in range(0,T-1):
	A[V.index(data[i]),V.index(data[i+1])] +=1

A += 5
for i in range(0,26):     #normalize
	row_sum = sum(A[i])
	for j in range(0,26):
		A[i][j] /= row_sum

#################################################################

text_file = open("CryptDataSimple.txt", "r")
data = text_file.read()
text_file.close()
data = data[:400]
O = list(data) 
T = len(data)
V = "abcdefghijklmnopqrstuvwxyz"
N = 26
M = len(V) #26
B = random_generator.generateRandMat(N,M)
pi= [ 0.04417064 , 0.03628645 , 0.03939599,  0.03930837 , 0.02829051 , 0.0485148
 , 0.04272197 , 0.04262368 , 0.03336933 , 0.03993394 , 0.04363741 , 0.03101957
 , 0.0215331 ,  0.03704817 , 0.04195192 , 0.03852417 , 0.01191433 , 0.05821716
 , 0.04037587 , 0.04518354 , 0.03060974 , 0.03364003 , 0.03225684 , 0.03481442
 , 0.04079882 , 0.06385925]
minIters = 200 # Min number of re-estimation iterations
epsi = 0.00001


oldLogProb = -sys.maxint - 1
c = [None]*T
a = np.zeros((T,N))
b = np.zeros((T,N))
y = np.zeros((T,N))
y_di = np.zeros((T,N,N))


#####################Q14 b/c/d)################################
def aPass():
	# Computer a0(i)
	# print("in a pass")
	c[0] = 0
	for i in range (0,N):
		a[0][i]=pi[i]* B[i][V.index(O[0])]
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
	# for i in range(0,N):
	# 	for j in range(0,N):
	# 		numer = 0
	# 		denom = 0
	# 		for t in range(0,T-1):
	# 			numer += y_di[t][i][j]
	# 			denom += y[t][i]
	# 		# print("numer :",numer)
	# 		# print("denom :",denom)
	# 		# A[i][j] = numer/denom

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
	print("Matrix B:")
	print((np.array(B)).transpose())
	print('')
	print("sum of B matrix row1:")
	print(sum(B[1])*1.0)
	print(datetime.datetime.time(datetime.datetime.now()))


def hmm():	
	iters = 0
	while (iters <  minIters):
		# oldLogProb = -logProb
		aPass()
		bPass()
		yPass()
		reEstimate()
		iters+=1
		# logP()
		# printabpi()


	key = [104, 103, 105, 113, 98, 116, 120, 118, 110, 97, 102, 107, 117, 121, 111, 106, 114, 122, 101, 100, 109, 119, 112, 115, 108, 99]
	key = key = np.array(key) - 97
	keyGenerated = np.zeros(26)
	count = 0
	for i in range(0,26):	
		keyGenerated[i] = (B[i].tolist()).index(max(B[i]))
		if (keyGenerated[i] == key[i]):
			count +=1

	print("Fraction :")
	print(count/26.0)
	return (count/26.0)

#################################################################
n = 1000

result = []
for j in range(0,n):
	result.append(hmm())

print("for nHMM where n ="+str(n)+" optimum value found at HMM no."+str(result.index(max(result)))+" and max fraction for the putative key that is correct :"+str(max(result)))