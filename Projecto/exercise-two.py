from __future__ import division, unicode_literals
from sklearn.datasets import fetch_20newsgroups
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk import ngrams
import math
import operator
import os



# ---------------""" Functions for WORDS """--------------------- #

def tf(word, docContent):
    # word -> candidate (string)
    # docContent -> list of document words without stop words(list)
    return docContent.count(word) / len(docContent)

def n_containing(word, documentList):
    return sum(1 for docContent in documentList if word in docContent)

def idf(word, documentList):
    return math.log((len(documentList) +1) / (1 + n_containing(word, documentList)))

def tfidf(word, docContent, documentList):
    return tf(word, docContent) * idf(word, documentList)




# Get relative path to documents
currentPath = os.path.dirname(os.path.abspath(__file__)) + "\documents\\";

fileList = []
# Get all documents in "documents" directory into fileList
# Import documents into fileList
for fileX in os.listdir(currentPath):
    if fileX.endswith(".txt"):
        f = open(currentPath + fileX, 'r')
        content = f.read()
        fileList += [content]
        f.close()


f = open("candidates.txt", 'r')
fileContent = f.read().decode('unicode_escape')
f.close()

stopWords = list(stopwords.words('english'))

docContentBlob = TextBlob(fileContent)
# remove stop words from each document
# [word1, word2, ...] without stop words
docWithoutStopWords = [x for x in docContentBlob.words if x not in stopWords]

# joins all words from a document separated by spaces, without the stopwords, to crate a TextBlob
documentSentence = " ".join(docWithoutStopWords)

# creates TextBlob of all words without stop words to get bi-grams
docContentBlobWithoutStopWords = TextBlob(documentSentence)

# turn bi-grams WordList into strings separated by spaces
documentBiGramList = []
for word in docContentBlobWithoutStopWords.ngrams(n=2):
    documentBiGramList += [" ".join(word)]

# print documentBiGramList

# joins all words and bi-grams from a document
docFinalCandidates = docWithoutStopWords + documentBiGramList

finalDocWordList = []
finalDocList = []

for trainDoc in fileList:
    trainBlob = TextBlob(trainDoc.decode('utf-8'))
    trainBlobWithoutStopWords = [x for x in trainBlob.words if x not in stopWords]
    finalDocWordList += [trainBlobWithoutStopWords]

finalDocList += finalDocWordList
finalDocBigramList = []
for num,trainDoc in enumerate(finalDocWordList):
    docContentAux = []
    DocBigramList = []
    docContentAux = ngrams(trainDoc, 2)
    for gram in docContentAux:
        DocBigramList += [" ".join(list(gram))]
    finalDocList[num] += DocBigramList


docCandidatesScored = []


#FIXME tf-idf for words seems to be working, WIP in bigrams
#create dictionary with words/bi-grams and it's score
for pos, trainDoc in enumerate(finalDocList):
    # dictionary containing word/bi-grams and its scores of tf_idf{ word: tf_idf } for a certain document
    keyphrasesDict = {}
    # /// Create 2 diferent lists (words, bigrams) for each document and the candidates doc ////

    print "\nDoc " + str(pos) + " ------------------------------\n"
#    print "finalCandidates: " + str(docFinalCandidates)
#    print " \n "

    for word in docFinalCandidates:
        #trainBlob = TextBlob(trainDoc.decode('utf-8'))
        keyphrasesDict[word] = tfidf(word, trainDoc, finalDocList)
#       print "Word: " + word + " ---- " + str(keyphrasesDict[word])

    docCandidatesScored += [keyphrasesDict]

print " \n "


for doc,candidateScored in enumerate(docCandidatesScored):
    print sorted(candidateScored.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]
    #print "Document " + str(doc) + " --- " + str(newdic) + "\n"