from __future__ import division, unicode_literals
from sklearn.datasets import fetch_20newsgroups
from textblob import TextBlob
from nltk.corpus import stopwords
import math
import os

""" Functions """

def tf(word, docContent):
    return docContent.words.count(word) / len(docContent.words)

def n_containing(word, documentList):
    return sum(1 for docContent in documentList if word in TextBlob(docContent).words)

def idf(word, documentList):
    return math.log(len(documentList) / (1 + n_containing(word, documentList)))

# docContent is TextBlob
# documentList is a list with all documents
def tfidf(word, docContent, documentList):
    return tf(word, docContent) * idf(word, documentList)


""" Script """

fileList = []
documentWordList = []
documentBiGramList = []
documentBiGramWordList = []
docCandidatesScored = []

# List of lists for each document candidates
# [[candidates of doc1],[candidates of doc2],[candidates of doc3], ...]
docCandidates = []

train = fetch_20newsgroups(subset='train');
trainData = train.data[:10]

# Use english text document to apply keyphrase extraction method
f = open("candidates.txt", 'r')
fileContent = f.read().decode('unicode_escape')
f.close()

print fileContent

docContentBlob = TextBlob(fileContent)

stopWords = set(stopwords.words('english'))

# remove stop words from each document
# [word1, word2, ...] without stop words
docWithoutStopWords = list(set(docContentBlob.words) - stopWords)

# joins all words from a document separated by spaces, without the stopwords, to crate a TextBlob
documentSentence = " ".join(docWithoutStopWords)

# creates TextBlob of all words without stop words to get bi-grams
docContentBlobWithoutStopWords = TextBlob(documentSentence)

# turn bi-grams WordList into strings separated by spaces
documentBiGramList = []
for word in docContentBlobWithoutStopWords.ngrams(n=2):
    documentBiGramList += [" ".join(word)]

# joins all words and bi-grams from a document
docCandidates += docWithoutStopWords + documentBiGramList

print docCandidates

# ----------------------------------------------------------------------------------
# Get relative path to documents
# currentPath = os.path.dirname(os.path.abspath(__file__)) + "\docTest\\";
#
#
# # Get all documents in "documents" directory into fileList
# # Import documents into fileList
# for fileX in os.listdir(currentPath):
#     if fileX.endswith(".txt"):
#         f = open(currentPath + fileX, 'r')
#         content = f.read()
#         fileList += [content]
#         f.close()
#

# # create dictionary with words/bi-grams and it's score
# for doc in fileList:
#     # dictionary containing word/bi-grams and its scores of tf_idf{ word: tf_idf } for a certain document
#     keyphrasesDict = {}
#     for word in docCandidates:
#         keyphrasesDict[word] = tfidf(word, TextBlob(doc.decode('unicode_escape')), fileList)
#         print keyphrasesDict[word]
#         print word
#     docCandidatesScored += [keyphrasesDict]
#
# print docCandidatesScored

# ----------------------------------------------------------------------------------

# FIXME tf-idf calculations not working
# create dictionary with words/bi-grams and it's score
for pos, trainDoc in enumerate(trainData):
    # dictionary containing word/bi-grams and its scores of tf_idf{ word: tf_idf } for a certain document
    keyphrasesDict = {}

    print "\nDoc " + str(pos) + " ------------------------------\n"

    for word in docCandidates:
        trainBlob = TextBlob(trainDoc.decode('unicode_escape'))
        keyphrasesDict[word] = tfidf(word, trainBlob, trainData)
        # print "-----------////////////------------"
        # print trainDoc
        print "Word: " + word + "     TF-IDF: " + str(keyphrasesDict[word])
        # print "-----------////////////------------"
    docCandidatesScored += [keyphrasesDict]

#print docCandidatesScored
