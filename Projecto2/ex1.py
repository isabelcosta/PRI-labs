#from __future__ import unicode_literals
from __future__ import division
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
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


def ngrams(text):

    finalCandidates = []

    text = "".join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()]).lower()
    tokensWithoutStopWords = [x for x in text.split() if x not in stopWords]

    finalCandidates += tokensWithoutStopWords
    finalCandidates += wordToNgrams(tokensWithoutStopWords, 2)
    finalCandidates += wordToNgrams(tokensWithoutStopWords, 3)

    return finalCandidates

def tok_sent(text):

    finalCandidates = []
    text = ''.join([c for c in text]).lower()
    text = [x for x in text.split() if x not in stopWords]
    finalCandidates += [" ".join(text)]

    ### [0] porque o sent tokenize tem de receber uma string ###
    sentenceList = sent_tokenize(finalCandidates[0])

    # for sent in sentList:
    #      print sent

    return sentenceList

def extractKeyphrases(text):


    print "\nFiltering document sentences ... "
    sentenceList = tok_sent(text)
    # print sentList



    for i, sent in enumerate(sentenceList):
        sentenceList[i] = "".join([c for c in sent if c not in string.punctuation])


    print "\nGetting ngrams ... "
    candidates = ngrams(text)
    # print candidates

    print "\nCreating graph ... "

    graph = {}

    for ngram in candidates:
        elementList = []
        for sentence in sentenceList:
            # print sentence
            # print ngram
            # print str(sentence.count(ngram)) + "\n"
            elementList.append(sentence.count(ngram))
        graph[ngram] = elementList

    for element in graph:
        print element
        print graph[element]











# --------------------------------------------------------------------#



print "\nGetting document ..."
#input doc that we want to extraxt keyphrases from
document = open("input.txt", 'r')
doc = document.read()
doc = doc.decode("unicode_escape")

keyphrases = extractKeyphrases(doc)





