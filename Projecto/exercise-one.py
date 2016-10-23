from __future__ import division, unicode_literals
from textblob import TextBlob
from sklearn.datasets import fetch_20newsgroups
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

import math
import os


listaDeFicheiros = []

for file in os.listdir("C:/Users/Fernando/Desktop/fao30/documents"):
    if file.endswith(".txt"):
        listaDeFicheiros += [file]

print listaDeFicheiros

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


blob1 = TextBlob(train.data[1])
blob2 = TextBlob(train.data[2])
blob3 = TextBlob(train.data[3])


bloblist = [blob1, blob2, blob3]

# print "imprimes ?"
#print test.noun_phrases


# print tf("guy",blob1)
#
# print idf("guy", bloblist)
#
# print blob1.ngrams(n=2)
