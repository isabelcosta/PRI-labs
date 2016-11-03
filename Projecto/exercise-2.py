from __future__ import division, unicode_literals
from sklearn import metrics
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import ngrams
import string
import os
import operator
import numpy as np
import codecs
# __import__("exercise-1")




# ---------------------------------------------
# --------------------------------------------------------------------#
## FUNCTIONS ##

def tok(text):
    text = "".join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()])
    tokens = nltk.word_tokenize(text)
    return tokens


def extractKeyphrases(train, test):

    print "Extracting keyphrases ... \n"

    vectorizer = TfidfVectorizer(use_idf=False, ngram_range=(1,2), stop_words="english", tokenizer=tok)
    trainvec = vectorizer.fit_transform(train)
    testvec = vectorizer.transform([test.decode('utf-8')])

    feature_names = vectorizer.get_feature_names()

    scores = {}
    for i in testvec.nonzero()[1]:
        scores[feature_names[i]] = testvec[0, i]

    topScores = sorted(scores.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]

    top ={}
    for i in topScores:
        top[i[0]] = i[1]

    return top


# --------------------------------------------------------------------#



# print "Getting training collection .."
# # Get relative path to documents
# currentPath = os.path.dirname(os.path.abspath(__file__)) + "\documents\\";
#
# fileList = []
# # Get all documents in "documents" directory into fileList
# # Import documents into fileList
# for fileX in os.listdir(currentPath):
#     if fileX.endswith(".txt"):
#         f = open(currentPath + fileX, 'r')
#         content = f.read()
#         fileList += [content]
#         f.close()
#
# print "Getting document .."
# #input doc that we want to extraxt keyphrases from
# document = open("input.txt", 'r')
# doc = document.read()
#
#
# top5 = extractKeyphrases(fileList, doc)
#
# print "Top 5 Keyphrase Candidates"
# for word in top5:
#     print "\t" + word + " : " + top5[word]
#

# ---------------------------------------------

'''
The keyphrase extraction method should now be applied to documents from one of the datasets
available at https://github.com/zelandiya/keyword-extraction-datasets.
In this case, the IDF scores should be estimated with basis on the entire collection of documents,
in the dataset of your choice, instead of relying on the 20 newsgroups collection.
Using one of the aforementioned datasets, for which we know what are the most relevant
keyphrases that should be associated to each document, your program should print the precision,
recall, and F1 scores achieved by the simple extraction method. Your program should also print
the mean average precision.
'''

'''     Functions           '''

def calc_precision(knownKeys, predictedKeys):
    # tp / (tp + fp)

    # print knownKeys
    # print predictedKeys
    # total = 0
    # for key in knownKeys:
    #     if key in predictedKeys:
    #         print key
    #         total += 1
    # print total

    tp = sum(1 for key in knownKeys if key in predictedKeys)
    fp = len(predictedKeys) - tp

    return tp / (tp + fp)

def calc_recall(knownKeys, predictedKeys):
    # tp / (tp + fn)
    tp = sum(1 for key in knownKeys if key in predictedKeys)
    fn = len(knownKeys) - tp

    return tp / (tp + fn)

def calc_f1(knownKeys, predictedKeys):
    # 2 * (precision * recall) / (precision + recall)
    return 2 * (calc_precision(knownKeys, predictedKeys) * calc_recall(knownKeys, predictedKeys)) / (calc_precision(knownKeys, predictedKeys) + calc_recall(knownKeys, predictedKeys))

def calc_mean_avg_precision(knownKeys, predictedKeys):
    #
    return

def remove_duplicates(keyphrasesList):
    seen = set()
    seen_add = seen.add
    return [x for x in keyphrasesList if not (x in seen or seen_add(x))]

def get_ngrams(keyphrasesList, n):
    return [' '.join(x) for x in ngrams(keyphrasesList, n)]
'''     Global Variables     '''
# contains the list of all dataset, each element represents a document from the dataset
# {docName => [content]}
fileList = {}
# {docName => [known keyphrases]}
knownKeyphrases = []
# {docName => [top keyphrases]}
predictedKeyphrases = []

# Get relative path to documents
datasetPath = os.path.dirname(os.path.abspath(__file__)) + "\documents\\";
# knownKeysPath = os.path.dirname(os.path.abspath(__file__)) + "\keys\\";

# Get all documents in "documents" directory into fileList
# Import documents into fileList
for docName in os.listdir(datasetPath):
    if docName.endswith(".txt"):
        f = open(datasetPath + docName, 'r')
        fileList[docName.partition(".txt")[0]] = f.read()
        f.close()

# Get dataset and known keyphrases

# Get all documents in "documents" directory into fileList
# Import documents into fileList
for docFromDataset in os.listdir(datasetPath):
    docContent = []
    docKeyphrases = []
    if docFromDataset.endswith(".txt"):
        docName = docFromDataset.partition(".txt")[0]
        fDoc = open(datasetPath + docFromDataset, 'r')
        docContent = fDoc.read()
        fDoc.close()
        # Get all document's known keyphrases in "keys" directory into knownKeyphrases
        # Import documents into fileList
        for keyphraseFile in os.listdir(datasetPath):
            if (keyphraseFile.endswith(".key") & (docName in str(keyphraseFile))):
                fKeys = open(datasetPath + keyphraseFile, 'r')
                docKeyphrases = fKeys.read()
                fKeys.close()

        # Use exercise-one to calculate top keyphrases to be associated with each document
        calculatedKeyScores = extractKeyphrases(fileList.values(), docContent)

        docKeyphrasesList = docKeyphrases.split()

        # calculatedKeyScores = extractKeyphrases((fileList.values().remove(fileList[docName])), docKeyphrases)
        knownKeyphrases = get_ngrams(docKeyphrasesList, 2) + remove_duplicates(docKeyphrasesList)
        predictedKeyphrases = calculatedKeyScores.keys()

        # Performance metrics - Calculate precision, recall, F1 scores and mean average precision

        # print "Metrics for document " + docFromDataset
        # print "Precision: \t" + str(metrics.precision_score(kkNumpyArrays, pkNumpyArrays))
        # print "Recall: \t" + str(metrics.recall_score(knownKeyphrases, predictedKeyphrases))
        # print "F1-score: \t" + str(metrics.f1_score(knownKeyphrases, predictedKeyphrases))
        # print "Mean Average Precision: \t" + str(metrics.average_precision_score(knownKeyphrases, predictedKeyphrases))

        print "Metrics for document " + docFromDataset
        print "Precision: \t" + str(calc_precision(knownKeyphrases, predictedKeyphrases))
        print "Recall: \t" + str(calc_recall(knownKeyphrases, predictedKeyphrases))
        print "F1-score: \t" + str(calc_f1(knownKeyphrases, predictedKeyphrases))
        # print "Mean Average Precision: \t" + str(metrics.average_precision_score(knownKeyphrases, predictedKeyphrases))

