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
from sklearn import metrics
import os
from "exercise-1.py" import extractKeyphrases

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

# Get dataset and known keyphrases

# Get all documents in "documents" directory into fileList
# Import documents into fileList
for docFromDataset in os.listdir(datasetPath):
    docContent = [];
    docKeyphrases = [];
    if docFromDataset.endswith(".txt"):
        docName = docFromDataset.partition(".txt");
        fDoc = open(datasetPath + docFromDataset, 'r')
        docContent = fDoc.read()
        fDoc.close()
        # Get all document's known keyphrases in "keys" directory into knownKeyphrases
        # Import documents into fileList
        for keyphraseFile in os.listdir(datasetPath):
            if (keyphraseFile.endswith(".key") & keyphraseFile.contains(docName)):
                fKeys = open(datasetPath + keyphraseFile, 'r')
                docKeyphrases = fKeys.read().split("\n");
                fKeys.close()
        fileList[docName] = docContent
        knownKeyphrases[docName] = docKeyphrases
        calculatedKeyScores = {}
        # Use exercise-one to calculate top keyphrases to be associated with each document
        # calculatedKeyScores = automatic_keyphrase_extraction()
        predictedKeyphrases = calculatedKeyScores.keys()

        # Performance metrics - Calculate precision, recall, F1 scores and mean average precision

        print "Metrics for document " + docFromDataset
        print "Precision: \t" + metrics.accuracy_score(knownKeyphrases, predictedKeyphrases)
        print "Recall: \t" + metrics.precision_score(knownKeyphrases, predictedKeyphrases)
        print "F1-score: \t" + metrics.average_precision_score(knownKeyphrases, predictedKeyphrases)
        print "Mean Average Precision: \t" + metrics.f1_score(knownKeyphrases, predictedKeyphrases)
