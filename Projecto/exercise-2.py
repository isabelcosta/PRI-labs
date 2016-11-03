from __future__ import division, unicode_literals
from sklearn import metrics
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import os
import operator
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
        scores[str(feature_names[i])] = str(testvec[0, i])

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

'''     Global Variables     '''
# contains the list of all dataset, each element represents a document from the dataset
# {docName => [content]}
fileList = {}
# {docName => [known keyphrases]}
knownKeyphrases = {}
# {docName => [top keyphrases]}
predictedKeyphrases = {}

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

        # calculatedKeyScores = extractKeyphrases((fileList.values().remove(fileList[docName])), docKeyphrases)
        knownKeyphrases[docName] = docKeyphrases.split("\n")
        predictedKeyphrases = calculatedKeyScores.keys()

        # Performance metrics - Calculate precision, recall, F1 scores and mean average precision

        print "Metrics for document " + docFromDataset
        print "Precision: \t" + str(metrics.accuracy_score(knownKeyphrases, predictedKeyphrases))
        print "Recall: \t" + str(metrics.precision_score(knownKeyphrases, predictedKeyphrases))
        print "F1-score: \t" + str(metrics.average_precision_score(knownKeyphrases, predictedKeyphrases))
        print "Mean Average Precision: \t" + str(metrics.f1_score(knownKeyphrases, predictedKeyphrases))
