from __future__ import division, unicode_literals
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import os
import operator
from nltk.corpus import stopwords

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

def tok(text):
    text = "".join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()])
    tokens = nltk.word_tokenize(text)
    return tokens

def extractKeyphrases(train, test):

    print "Extracting keyphrases ... \n"
    vectorizer = TfidfVectorizer(use_idf=False, ngram_range=(1,2), stop_words=list(stopwords.words('english')), tokenizer=tok)
    trainvec = vectorizer.fit_transform(train)
    testvec = vectorizer.transform(test)

    print "tou lento"
    previousDoc = 0

    feature_names = vectorizer.get_feature_names()
    scores = []
    docScores = {}
    row, columns = testvec.nonzero()
    for doc, word in zip(row, columns):

        docScores[feature_names[word]] = str(testvec[doc, word])
        if previousDoc != doc:

            currentSortedScore = sorted(docScores.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]
            # print "Doc: " + str(previousDoc+1) + " " + str(currentSortedScore)

            scores += [currentSortedScore]
            docScores = {}

        previousDoc = doc


    currentSortedScore = sorted(docScores.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]
    # print "Doc: " + str(previousDoc+1) + " " + str(currentSortedScore)
    scores += [currentSortedScore]

    top =[]
    for listT in scores:
        topAux={}
        for tuple in listT:
            topAux[tuple[0]] = tuple[1]
        top += [topAux]

    return top

def calc_precision(tp, fp):
    return float(tp / (tp + fp))

def calc_recall(tp, fn):
    return float(tp / (tp + fn))

def calc_f1(precision, recall):
    if precision == 0 and recall == 0:
        return "N/A"
    return 2 * ((precision * recall) / (precision + recall))

def calc_precision_recall_f1_score(docName, knownKeyphrases, predictedKeyphrases):

    tp = sum(1 for key in knownKeyphrases if key in predictedKeyphrases)
    # tp = sum(1 for key in predictedKeyphrases if key in knownKeyphrases)

    fn = len(knownKeyphrases) - tp
    fp = len(predictedKeyphrases) - tp

    precision = calc_precision(tp, fp)
    recall = calc_recall(tp, fn)

    f1_score = calc_f1(precision, recall)

    print docName + ": " + str(precision) + ", " + str(recall) + ", " + str(f1_score)
    return precision, recall, f1_score

def calc_mean_avg_precision(all_precisions):
    mean_avg_precision =  sum(all_precisions)/len(all_precisions)
    print "Mean Average Precision: \t" + str(mean_avg_precision)
    return mean_avg_precision

'''  -------------------------   Global Variables   ---------------------------  '''
# contains the list of all dataset, each element represents a document from the dataset
# {docName => [content]}
fileList = {}

# {docName => [known keyphrases]}
knownKeyphrases = []

# {docName => [top keyphrases]}
predictedKeyphrases = []

# all precisions to be used in mean average precision
allPrecisions = []

''' ------------------------- Exercise Execution ------------------------- '''

# Get relative path to documents
datasetPath = os.path.dirname(os.path.abspath(__file__)) + "\\documents\\";

# Get all documents in "documents" directory into fileList
# Import documents into fileList
for docName in os.listdir(datasetPath):
    if docName.endswith(".txt"):
        f = open(datasetPath + docName, 'r')
        # fileList[docName.partition(".txt")[0]] = f.read().decode("unicode_escape")
        fileList[docName.partition(".txt")[0]] = f.read()
        # print fileList[docName.partition(".txt")[0]]
        f.close()

# Get dataset and known keyphrases
# Use exercise-one to calculate top keyphrases to be associated with each document
calculatedKeyScores = extractKeyphrases(fileList.values(), fileList.values())

print "Document: Precision, Recall, F1-score"

# Get all documents in "documents" directory into fileList
# Import documents into fileList
nDocs = 0
for docName in fileList:
    docContent = ""
    docKeyphrases = []

    docContent = fileList[docName]
    # Get all document's known keyphrases in "keys" directory into knownKeyphrases
    # Import documents into fileList0
    fKeys = open(datasetPath + docName + ".key", 'r')
    docKeyphrases = fKeys.read()
    fKeys.close()

    knownKeyphrases = docKeyphrases.splitlines()
    knownKeyphrases = [x.decode("unicode_escape") for x in knownKeyphrases]
    predictedKeyphrases = calculatedKeyScores[nDocs].keys()
    # print knownKeyphrases
    # print predictedKeyphrases

    # Performance metrics - Calculate precision, recall, F1 scores and mean average precision
    allPrecisions += [calc_precision_recall_f1_score(docName, knownKeyphrases, predictedKeyphrases)[0]]

    nDocs += 1

calc_mean_avg_precision(allPrecisions)
