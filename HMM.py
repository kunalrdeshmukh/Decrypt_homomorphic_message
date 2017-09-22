import nltk
import unicodedata
import re
import numpy as np
import math
from nltk.corpus import brown
import sys
import random
import datetime

data = "".join(brown.words())
data = data.lower() 
data =  re.sub('[^A-Za-z ]+', '',data)
data = re.sub('  +','',data)
data = data.encode("ascii")
data = data[:50]
V = "abcdefghijklmnopqrstuvwxyz"
V = list(V)
M = len(V)
N = 2
pi = [0.5786,0.4214]
A = [[0.47467,0.52533],[0.51656,0.48344]]
B = np.random.rand(N,M)
B =  B/B.sum(axis=1)[:,None]  # from https://stackoverflow.com/questions/
								#31364744/creating-a-matrix-of-arbitrary-size-where-rows-sum-to-1

B= [[ 0.03752125 , 0.03737753 , 0.03711691 , 0.03715558 , 0.0369791 ,  0.03698675,
   0.03681541 , 0.03701147 , 0.03721152 , 0.03688235 , 0.03716021 , 0.03695056,
   0.03688205 , 0.03726031 , 0.0369619  , 0.03702636 , 0.03699626 , 0.03709803,
   0.0368785  , 0.03672573 , 0.03704332 , 0.03705526 , 0.03727107 , 0.03715967,
   0.03678304 , 0.07368986]
 ,[ 0.03728717 , 0.03695286 , 0.03712069 , 0.03689174 , 0.03721092 , 0.03731084,
   0.03689683 , 0.03731148 , 0.03715983 , 0.03704478 , 0.03666429  ,0.03702744,
   0.0369244  , 0.03702157 , 0.03741971 , 0.03720426 , 0.0372748  , 0.03736758,
   0.0367429  , 0.03719349 , 0.03687142 , 0.03696339 , 0.03694234 , 0.03694469,
   0.03713507 ,0.070969642]]

B = np.array(B)
O = list(data) 
minIters = 400 # Min number of re-estimation iterations
epsi = 0.00001
T = len(data)
iters = 0
oldLogProb = -sys.maxint - 1
c = [0] * T
a = np.zeros((T,N))
b = np.zeros((T,N))
y = np.zeros((T,N))
y_di = np.zeros((T,N,N))
c = [None]*T


def printParam(i):
	print("Iteration :",i)
	print(B)
	print(P())





############################################################
#forward algorithm:


def sumA(i,t): # to take of summation part of recurrence equation 
	sum = 0
	for j in range(0,N):
		print(i,j,t)
		sum += a[t-1][j] *A[j][i]
	return sum


def forward(o): 
	for i in range(0,N):
		a[0][i] = pi[i] * B[i][V.index(o[0])]
	for t in range(1,T):
		temp = 0
		for i in range(0,N):
			for j in range(0,N):
				temp += a[t-1][j]* A[i][j]
		a[t][i] = temp*B[i][V.index(o[t])]

 
#######################################################
# Backward Algorithm

def sumB(t,i):
	sumb = 0
	for j in range(0,N):
		sumb += A[i][j]*B[j][V.index(O[t+1])]*b[t+1][j]
	return sumb

def backward():
	for i in range(0,N):
		b[0][i] = 1
	for t in range (T-1,0,-1):
		for i in range(0,N):
			temp = 0
			for j in range(0,N):
				temp += b[1][j] * A[i][j]*B[j][V.index(O[t])]
			b[0][i] = temp
			

#######################################################
# gaama and di-gamma calculation:
def P():
	result = 0
	for i in range(0,N):
		result += a[T-1][i]
	return result 

def calculateY():
	denom = 0.0
	for t in range(0,T):
		for i in range(0,N):
			# print(i,t)
			y[t][i] = (a[t][i]*	b[t][i])/P()
			denom += y[t][i]
	for i in range(0,N):
		y[T-1][i] /= denom


def calculateYdi():
	for t in range(0,T-1):
		for i in range(0,N):
			for j in range(0,N):
				y_di[t][i][j] = (a[t][i]*A[i][j]*B[j][V.index(O[t+1])]*b[t+1][j])/P()

###########################################################
a_temp_sum = [0.0]*(M+1)
def fwdbkwd():
	forward(O)
	backward()
	# print(B.shape)
	y = []
	for t in range(0,T):
		temp_sum = 0
		y.append({})
		for i in range(0,N):
			temp=a[t][i]*b[t][i]
			y[t][i] = temp
			temp_sum += temp
		if temp_sum != 0:
			for i in range(0,N):
				y[t][i] /= temp_sum

	y_di=[]
	for t in range(0,T-1):
		y_di.append({})
		temp_sum = 0
		for i in range(0,N):
			y_di[t][i] = {}
			for j in range(0,N):
				temp = a[t][i]*b[t+1][j]*A[i][j]* \
						B[j][V.index(O[t+1])]
				y_di[t][i][j] = temp
				temp_sum += temp
		if temp_sum != 0:
			for i in range(0,N):
				for j in range(0,N):
					y_di[t][i][j] /= temp_sum

	for i in range(0,N):
		pi[i] = y[0][i] / (1 + N)

		temp_sum = 0
		for j in range(N-1):
			temp_sum += y[i][j]

		if temp_sum > 0:
			denom = temp_sum + N
			for j in range(0,N):
				temp_sum = 0
				for t in range(0,T-1):
					temp_sum += y_di[t][i][j]
				A[i][j] = temp_sum/denom
		else:
			for j in range(0,N):
				A[i][j] = 0

		temp_sum += y[N-1][i]
		
		for j in range(0,M):
			a_temp_sum[j] = 0

		for j in range(0,N):
			# print(V.index(O[t+1]))
			a_temp_sum[V.index(O[t+1])] += y[j][i]
			
		if temp_sum > 0:
			denom = temp_sum + M
			for j in range(0,M):
				B[i][j] = a_temp_sum[j]/denom
		else:
			for j in range(0,M):
				B[i][j]=0



########################################################
# Baum-Welch re-estimation

def sumy(T,i):
	sumy = 0
	for t in range(0,T):
		sumy += y[i][i]
	return sumy

def sumy_di(T,i,j):
	sumy = 0
	for t in (0,T):
		sumy += y_di[t][i][j]
	return sumy


def reEstimation():
	for i in range(0,N):
		pi[i] = y[0][i]
	for i in range(0,N):
		for j in range(0,N):
			A[i][j] = np.divide(sumy_di(T-1,i,j),sumy(T-1,i))
	for j in range(0,N):
		for k in range(0,M):
			B[j][k] = np.divide(sumy(T,j),sumy(T,j))


###########################################################
print(B.transpose())
for i in range(0,5):
	# print(datetime.datetime.time(datetime.datetime.now()))
	fwdbkwd()
	print(B.transpose())
