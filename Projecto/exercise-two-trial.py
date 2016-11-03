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


'''     Imports     '''
from __future__ import division, unicode_literals
from textblob import TextBlob
from nltk.corpus import stopwords
import math
import os

'''     Global Variables     '''
fileList = []

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




# Get dataset and known keyphrases

# Use exercise-one to calculate top keyphrases to be associated with each document

# Calculate precision, recall, F1 scores and mean average precision
