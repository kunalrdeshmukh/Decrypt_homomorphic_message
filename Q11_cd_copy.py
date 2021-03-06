import nltk
import unicodedata
import re
import numpy as np
import math
import sys
import datetime
from nltk.corpus import brown
import random_generator

text_file = open("CryptDataSimple.txt", "r")
data = text_file.read()
text_file.close()
data = data[:1000]
O = list(data) 
data = "".join(brown.words())
data = data.lower() 
data =  re.sub('[^A-Za-z ]+', '',data)
data = re.sub('  +','',data)
data = data.encode("ascii")
data = data[:1000000]
T = len(data)
A = np.zeros((26,26))
V = "abcdefghijklmnopqrstuvwxyz"
N = 26
M = len(V) #26
B = random_generator.generateRandMat(N,M)
pi= [ 0.04417064 , 0.03628645 , 0.03939599,  0.03930837 , 0.02829051 , 0.0485148
 , 0.04272197 , 0.04262368 , 0.03336933 , 0.03993394 , 0.04363741 , 0.03101957
 , 0.0215331 ,  0.03704817 , 0.04195192 , 0.03852417 , 0.01191433 , 0.05821716
 , 0.04037587 , 0.04518354 , 0.03060974 , 0.03364003 , 0.03225684 , 0.03481442
 , 0.04079882 , 0.06385925]
 # pi =  np.random.normal(1/26,0.01,26)
 # pi[25] = 1 - (sum(pi) - pi[25])
minIters = 200 # Min number of re-estimation iterations
epsi = 0.00001
iters = 0
oldLogProb = -sys.maxint - 1
c = [None]*T
a = np.zeros((T,N))
b = np.zeros((T,N))
y = np.zeros((T,N))
y_di = np.zeros((T,N,N))

########################Q11 c)###########################
print("Q11 c:")
for i in range(0,T-1):
	A[V.index(data[i]),V.index(data[i+1])] +=1

A += 5
for i in range(0,26):     #normalize
	row_sum = sum(A[i])
	for j in range(0,26):
		A[i][j] /= row_sum

for i in range(0,26):
	print(sum(A[i]))
print("Matrix A:")
print(A)

#####################Q11 d)################################
print("####################################################")
print("Q11 d:")
T = len(O)

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

while (iters <  minIters):
	# oldLogProb = -logProb
	aPass()
	bPass()
	yPass()
	reEstimate()
	iters+=1
	# logP()
	printabpi()

key = np.zeros(N)
for i in range(0,N):
	key[i] = (B[i].tolist()).index(max(B[i]))
print (key)
countr = 0
offset = [99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116
,117,118,119,120,121,122,97,98] # generated key
for i in range(0,N):
	if(key[i] ==  offset[i]):
		countr += 1
