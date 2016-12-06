from __future__ import division
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import operator
import os

# TRAIN -> NEWSGROUP, TEST -> COLECÇAO DE FILES DO GIT
def extractKeyphrases(train, test):

    print "Extracting keyphrases ... \n"
    



# --------------------------------------------------------------------#

fileList = {}
knownKeyphrases = []
predictedKeyphrases = []


# Get relative path to documents
datasetPath = os.path.dirname(os.path.abspath(__file__)) + "\\documents\\";

# Get all documents in "documents" directory into fileList
# Import documents into fileList
for docName in os.listdir(datasetPath):
    if docName.endswith(".txt"):
        f = open(datasetPath + docName, 'r')
        fileList[docName.partition(".txt")[0]] = f.read()
        f.close()


keyphrases = extractKeyphrases(fileList.values(), fileList.values())