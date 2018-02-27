import nltk
import unicodedata
import re
import numpy as np
import math
import sys
import datetime
from nltk.corpus import brown
import random_generator


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

########################Q15 a)###########################
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



###########################################################

# insert ciphertext in O
O = [1,2,3,4,5,4,6,7,2,8,9,10,11,12,13,11,7,14,15,16,17,18,19,20,21,1,22,3,23,24,25,26,19,17,27,28,19,29,6,30,8,31,26,32,33,34,35,19,36,37,38,39,40,4,1,2,7,3,9,10,41,6,2,42,10,43,26,44,8,29,45,27,5,28,46,47,48,12,20,22,15,14,17,31,19,23,16,26,18,36,1,24,30,38,21,26,13,49,37,50,39,40,10,34,33,30,19,44,43,9,1,26,18,7,32,21,39,2,7,45,46,4,3,2,7,23,13,26,44,22,27,6,29,10,10,8,51,5,24,26,12,30,38,14,26,25,49,37,45,27,47,1,52,7,3,36,10,16,28,11,21,48,34,40,17,44,6,22,8,20,5,51,12,9,15,14,30,37,16,33,45,38,43,29,10,21,22,30,1,36,10,53,32,19,47,48,46,17,4,23,13,28,35,41,3,37,27,49,10,6,33,2,45,38,34,15,44,24,22,11,18,47,30,25,28,8,37,1,49,45,27,43,34,41,38,5,40,3,50,6,12,8,41,1,52,7,15,14,48,16,15,32,33,9,3,29,11,39,47,43,42,6,17,21,31,36,50,18,2,2,25,27,34,8,38,39,51,44,4,1,2,2,5,42,41,3,52,7,15,12,17,13,26,14,26,53,20,52,49,51,16,23,1,41,1,7,2,9,32,37,10,6,51,16,53,46,19,26,53,29,39,26,14,15,5,17,18,19,24,44,53,32,19,41,1,2,52,45,33,53,22,25,20,7,13,1,50,13,41,36,46,48,31,45,25,11,26,53,17,46,52,52,21,17,37,3,9,10,13,35,20,2,18,51,5,23,28,32,33,26,53,49,28,30,16,47,7,3,35,14,21,15,44,13,47,1,14,30,21,26,44,22,27,38,11,19,30,8]

T = len(O)
V = []
for i in range(1,54):
	V.append(i)
N = 26
M = len(V) #54
B = random_generator.generateRandMat(N,M)
pi= [ 0.04417064 , 0.03628645 , 0.03939599,  0.03930837 , 0.02829051 , 0.0485148
 , 0.04272197 , 0.04262368 , 0.03336933 , 0.03993394 , 0.04363741 , 0.03101957
 , 0.0215331 ,  0.03704817 , 0.04195192 , 0.03852417 , 0.01191433 , 0.05821716
 , 0.04037587 , 0.04518354 , 0.03060974 , 0.03364003 , 0.03225684 , 0.03481442
 , 0.04079882 , 0.06385923]
minIters = 200 # Min number of re-estimation iterations
epsi = 0.00001

result = []

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



def hmm(i):	
	iters = 0
	while (iters <  minIters):
		# oldLogProb = -logProb
		aPass()
		bPass()
		yPass()
		reEstimate()
		iters+=1
		# logP()
		print(iters)


	temp=[]
	for i in range(0,54):
		temp.append(B[:,i].tolist().index(max(B[:,i])))

	if i%100 == 0:
		print(B)

	return temp


#################################################################
n = 1000

result = []
for j in range(0,n):
	result.append(hmm(j))


with open('csvfile.csv','wb') as file:
    for line in result:
        file.write(str(line))
        file.write('\n')