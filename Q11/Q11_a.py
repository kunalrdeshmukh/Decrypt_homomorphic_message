import nltk
import unicodedata
import re
import numpy as np
import math
from nltk.corpus import brown
import random
import sys

data = "".join(brown.words())
data = data.lower() 
data =  re.sub('[^A-Za-z ]+', '',data)
data = re.sub('  +','',data)
data = data.encode("ascii")
data = data[:50000]


cryptString = ""

offset = []
def cryptData(cryptString):
	offset = random.sample(range(97,123),26)
	for i in range(0,50000):
		cryptString += chr(offset[ord(data[i])-97])
		# print chr(offset[ord(data[i])-97]).encode("ascii",'replace')
	return cryptString

def cryptDataSimple(cryptString):
	offset = [99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116
,117,118,119,120,121,122,97,98]
	for i in range(0,50000):
		cryptString += chr(offset[ord(data[i])-97])
		# print chr(offset[ord(data[i])-97]).encode("ascii",'replace')
	return cryptString

cryptString = cryptDataSimple(cryptString)

text_file1 = open("PlainData.txt", "w")
text_file1.write(data)
text_file1.close()

text_file2 = open("CryptDataSimple.txt", "w")
text_file2.write(cryptString)
text_file2.close()

text_file3 = open("KeyS.txt", "w")
text_file3.write(str(offset))
text_file3.close()

