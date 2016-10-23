from __future__ import division, unicode_literals
from textblob import TextBlob

from nltk.corpus import stopwords

from sklearn.datasets import fetch_20newsgroups
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

import math
import os

i = 1
fileList = []
documentList = []

for file in os.listdir("C:/Users/Fernando/Desktop/PRI/PRI-labs/Projecto/fao30/documents"):
    if file.endswith(".txt"):
        f = open("C:/Users/Fernando/Desktop/PRI/PRI-labs/Projecto/fao30/documents/" + file, 'r')
        content = f.read()
        fileList += [content]
        f.close()



for file in fileList:
    docContent = TextBlob(file.decode('unicode_escape'))
    documentList += [docContent]



def tf(word, docContent):
    return docContent.words.count(word) / len(docContent.words)

def n_containing(word, documentList):
    return sum(1 for docContent in documentList if word in docContent.words)

def idf(word, documentList):
    return math.log(len(documentList) / (1 + n_containing(word, documentList)))

def tfidf(word, docContent, documentList):
    return tf(word, docContent) * idf(word, documentList)



doc1 = TextBlob("ola, adeus,!!! the")
doc2 = TextBlob(train.data[2])
doc3 = TextBlob(train.data[3])


documentList = [doc1, doc2, doc3]

# stop_words = set(stopwords.words('english'))
# doc1 = set(doc1.words) - stop_words
#
# print doc1
#
# print idf("guy", documentFist)
#
# print doc1.ngrams(n=2)

