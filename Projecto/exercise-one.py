from __future__ import division, unicode_literals
from textblob import TextBlob
from nltk.corpus import stopwords
import math
import os


""" Functions """

def tf(word, docContent):
    return docContent.words.count(word) / len(docContent.words)

def n_containing(word, documentList):
    return sum(1 for docContent in documentList if word in TextBlob(docContent.decode('unicode_escape')).words)

def idf(word, documentList):
    return math.log(len(documentList) / (1 + n_containing(word, documentList)))

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

# Get relative path to documents
currentPath = os.path.dirname(os.path.abspath(__file__)) + "\documents\\";


# Get all documents in "documents" directory into fileList
# Import documents into fileList
for fileX in os.listdir(currentPath):
    if fileX.endswith(".txt"):
        f = open(currentPath + fileX, 'r')
        content = f.read()
        fileList += [content]
        f.close()


# For each document in fileList, removes stop words and punctuation and gathers all words and bi-grams
for fileX in fileList:

    docContentBlob = TextBlob(fileX.decode('unicode_escape'))

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
    docCandidates += [docWithoutStopWords + documentBiGramList]

    # print docCandidates
    # print "\n"


print "This is about to be slow AF"
print "cat"
# create dictionary with words/bi-grams and it's score
for pos, fileX in enumerate(fileList):
    # dictionary containing word/bi-grams and its scores of tf_idf{ word: tf_idf } for a certain document
    keyphrasesDict = {}
    print "shark"
    for word in docCandidates[pos]:
        if word not in keyphrasesDict:
            keyphrasesDict[word] = tfidf(word, TextBlob(fileX.decode('unicode_escape')), fileList)
            print "lion"
    docCandidatesScored += [keyphrasesDict]

print "dog"

# for i, blob in enumerate(fileList):
#     print("Top words in document {}".format(i + 1))
#     scores = {word: tfidf(word, blob, fileList) for word in blob.words}
#     sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     for word, score in sorted_words[:3]:
#         print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

# for document in documentList:
#     print document.words

# doc1 = TextBlob("ola!!!, awsegdas, fewasdga!!! adeus.........the............")
# # print doc1.ngrams(n=2)
# for word in doc1.ngrams(n=2):
#     print " ".join(word)

    # rain.data[2])
# doc3 = TextBlob(train.data[3])
#
#
# documentList = [doc1, doc2, doc3]
# stop_words = set(stopwords.words('english'))
# doc1 = set(doc1.words) - stop_words
# print list(doc1)
#
# #
# print idf("guy", documentFist)
#
# print doc1.ngrams(n=2)

# list1 = ['1', '2', '3']
# str1 = ' '.join(list1)
#
# print str1