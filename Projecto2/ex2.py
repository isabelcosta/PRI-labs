#from __future__ import unicode_literals
from __future__ import division
from sklearn.datasets import fetch_20newsgroups
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import operator
import os


stopWords = list(stopwords.words('english'))

def calc_precision(tp, fp):
    return float(tp / (tp + fp))

def calc_mean_avg_precision(all_precisions):
    mean_avg_precision =  sum(all_precisions)/len(all_precisions)
    print "Mean Average Precision: \t" + str(mean_avg_precision)
    return mean_avg_precision

def pagerank(graph, d, numloops):

    ranks = {}
    npages = len(graph)
    for page in graph:
        ranks[page] = 1.0 / npages

    for i in range(0, numloops):
        # print i
        newranks = {}
        for page in graph:
            newrank = d / npages
            for node in graph:
                if page in graph[node]:
                    newrank = newrank + (1 - d) * (ranks[node] / len(graph[node]))
            newranks[page] = newrank
        ranks = newranks
    return ranks

# def tok(text):
#     text = "".join([c for c in text if c not in string.punctuation])
#     text = ''.join([c for c in text if not c.isdigit()])
#     tokens = nltk.word_tokenize(text)
#     return tokens
def tok(text):
    finalCandidates = []

    text = "".join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()]).lower()
    tokensWithoutStopWords = [x for x in text.split() if x not in stopWords]

    finalCandidates += tokensWithoutStopWords
    finalCandidates += wordToNgrams(tokensWithoutStopWords, 2)
    finalCandidates += wordToNgrams(tokensWithoutStopWords, 3)

    return finalCandidates

def calc_BM25_of_file(file):
    # trainSize = len(train)
    #
    # print "\nFiltering background collection ... "
    # trainFiltered = []
    # for doc in train:
    #     trainFiltered += tok(doc)

    print "\nFiltering document ... "
    candidates = tok(file)

    # treating input document
    inputText = "".join([c for c in file if c not in string.punctuation])
    inputText = ''.join([c for c in inputText if not c.isdigit()])
    tokensInput = nltk.word_tokenize(inputText)
    inputText = ' '.join([x for x in tokensInput if x not in stopWords])
    inputText = inputText.lower()
    inputText = inputText.encode('utf-8')

    ####CALCULATES TF FOR INPUT ####
    inputLen = len(test)
    scoresTF = {}
    for word in candidates:
        scoresTF[word] = inputText.count(word) / inputLen

    ####CALCULATES IDF FOR BACKGROUND COLLECTION ####
    scoresIDF = {}
    for word in candidates:
        nt = sum(1 for docContent in trainFiltered if word in docContent)
        scoresIDF[word] = IDF(trainSize, nt)

    # avg length of the docs
    sumWords = 0
    for doc in train:
        sumWords += len(doc)
    avgDL = sumWords / trainSize

    print "\nCalculating BM25 ... "
    scoresBM25 = {}
    for word in candidates:
        scoresBM25[word] = BM25(scoresIDF[word], scoresTF[word], inputLen, avgDL)

    # TOP 5
    return sorted(scoresBM25.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]


# TRAIN -> NEWSGROUP, TEST -> COLECÇAO DE FILES DO GIT
def extractKeyphrases(train, test):

    print "Extracting keyphrases ... \n"
    vectorizer = TfidfVectorizer(use_idf=False, ngram_range=(1,3), stop_words=list(stopwords.words('english')), tokenizer=tok)
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


    for i, something in enumerate(top):
        print i
        print something




# --------------------------------------------------------------------#

fileList = {}
knownKeyphrases = []
predictedKeyphrases = []

# TRAIN DATA

train = fetch_20newsgroups(subset='train')
trainData = train.data[:100]


# Get relative path to documents
datasetPath = os.path.dirname(os.path.abspath(__file__)) + "\\documents\\";

# Get all documents in "documents" directory into fileList
# Import documents into fileList
for docName in os.listdir(datasetPath):
    if docName.endswith(".txt"):
        f = open(datasetPath + docName, 'r')
        fileList[docName.partition(".txt")[0]] = f.read()
        f.close()


keyphrases = extractKeyphrases(train, fileList.values())





