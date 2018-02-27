import nltk
import unicodedata
import re
import numpy as np
import math
from nltk.corpus import brown
import random
import datetime
import sys
import random_generator


data = "".join(brown.words())
data = data.lower() 
data =  re.sub('[^A-Za-z ]+', '',data)
data = re.sub('  +','',data)
data = data.encode("ascii")
data = data[:1000000]
T = len(data)
V = "abcdefghijklmnopqrstuvwxyz"



############### A matrix generated in 14a######

A = np.zeros((26,26))
for i in range(0,T-1):
	A[V.index(data[i]),V.index(data[i+1])] +=1

A += 5
for i in range(0,26):     #normalize
	row_sum = sum(A[i])
	for j in range(0,26):
		A[i][j] /= row_sum
# print(A)
# print("Matrix A is ready.")

###################################################3
data = data[:1000]
offset_value = 0
T = len(data)
cryptString = ""


def cryptData(cryptString):
	offset = []
	offset = random.sample(range(97,123),26)
	for i in range(0,len(data)):
		cryptString += chr(offset[ord(data[i])-97])
		# print chr(offset[ord(data[i])-97]).encode("ascii",'replace')
	return cryptString

def cryptDataSimple(cryptString):
	offset_value = random.randrange(5,20)
	print(offset_value)
	for i in range(0,len(data)):
		if (ord(data[i])+offset_value > 122):
			cryptString += chr(ord(data[i]) + offset_value-26)
		else:
			cryptString += chr(ord(data[i])+offset_value)
		# print chr(offset[ord(data[i])-97]).encode("ascii",'replace')
	return cryptString

cryptString = cryptDataSimple(cryptString)

text_file1 = open("PlainData.txt", "w")
text_file1.write(data)
text_file1.close()

text_file2 = open("CryptData.txt", "w")
text_file2.write(cryptString)
text_file2.close()

text_file3 = open("Key.txt", "w")
text_file3.write(str(offset_value))
text_file3.close()


#################################################################

text_file = open("CryptData.txt", "r")
data = text_file.read()
text_file.close()
data = data[:300]                    # number of charactor in inpur string
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


	temp=[]
	for i in range(0,26):
		temp.append(str((B[i].tolist()).index(max(B[i]))-i))
		
	return temp


#################################################################
n = 10

result = []
for j in range(0,n):
	result.append(hmm(j))


with open('csvfile.csv','wb') as file:
    for line in result:
        file.write(str(line))
        file.write('\n')
