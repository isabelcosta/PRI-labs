#from __future__ import unicode_literals
from __future__ import division
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
from nltk.corpus import stopwords
from math import log
from nltk import ngrams
import itertools
import re
import operator


stopWords = list(stopwords.words('english'))

# def word_grams(words):
#     s = []
#     s += words
#     for tuple in ngrams(words, 2):
#         s.append(tuple)
#     for tuple in ngrams(words, 3):
#         s.append(tuple)
#
#     return s

def wordToNgrams(text, n, exact=True):
    return [" ".join(j) for j in zip(*[text[i:] for i in range(n)])]


def tok(text):

    finalCandidates = []

    text = "".join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()]).lower()
    tokensWithoutStopWords = [x for x in text.split() if x not in stopWords]
    finalCandidates += tokensWithoutStopWords
    finalCandidates += wordToNgrams(tokensWithoutStopWords, 2)
    finalCandidates += wordToNgrams(tokensWithoutStopWords, 3)

    return finalCandidates

def extractKeyphrases(test):



    print "\nFiltering document ... "
    candidates = tok(test)






    print "\n print candidates ..."
    print candidates









# --------------------------------------------------------------------#



print "\nGetting document ..."
#input doc that we want to extraxt keyphrases from
document = open("input.txt", 'r')
doc = document.read()
doc = doc.decode("unicode_escape")

top5 = extractKeyphrases(doc)





