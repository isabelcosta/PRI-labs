#from __future__ import unicode_literals
from __future__ import division
from sklearn.datasets import fetch_20newsgroups
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import operator
import itertools
import os
from math import*
from math import log
from sklearn.linear_model import Perceptron
from numpy import dot



stopWords = list(stopwords.words('english'))



##############  FUNCTIONS  ##################

### Mean Precision

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


#CALCULO IDF
# N="docs in collection, nt=#docs with term
def IDF(candidatesByDoc, filteredTrain, NDocs):
    scoresIDF = {}
    for doc in candidatesByDoc:
        docScores={}
        for ngram in candidatesByDoc[doc]:
            #print ngram
            nt = sum(1 for doc in filteredTrain if ngram in filteredTrain[doc]) #FALTA FILTERING TRAIN DATASET
            #print NDocs
            #print nt
            docScores[ngram] = log(1+(NDocs-nt+0.5)/(nt+0.5))
        scoresIDF[doc]=docScores
    return scoresIDF



def TF(candidatesbyDoc, filteredTrain, docsLen):
    scoresTF = {}
    for doc in candidatesbyDoc:
        scoresDoc= {}
        for ngram in candidatesbyDoc[doc]:
            scoresDoc[ngram] = filteredTrain[doc].count(ngram)/docsLen[doc]

        scoresTF[doc]=scoresDoc
    return scoresTF

### BM25
#CALCULO BM25
#recebe o idef, o tf, D=doc length=(#words), avgdl=avg D
def BM25( idf, tf, D, avgdl):
    k=1.2
    b= 0.75
    top = tf *(k+1)
    aux = D/avgdl
    bottom = tf + k * (1 - b + b * aux)
    return idf * top/bottom



"""
    Calculates PageRank score for each n-gram
    input:
        graph => dictionary with key = n-gram and value = [n-grams]
        d => dumping factor
        numloops => number of iterations
    output:
        ......
"""
def pagerank(graph, d, numloops):

    ranks = {}
    npages = len(graph)
    for page in graph:
        ranks[page] = 1.0 / npages

    for loop in range(0, numloops):
        # print loop
        newranks = {}
        for page in graph:
            newrank = d / npages
            for node in graph:
                if page in graph[node]:
                    newrank = newrank + (1 - d) * (ranks[node] / len(graph[node]))
            newranks[page] = newrank
        ranks = newranks
    return ranks


### PAGERANK IMPROVED
def pagerankImproved(graph, keywords, Prior, d, numloops):

    ranks = {}

    nCandidates = len(graph)
    for ngram in graph:
        ranks[ngram] = 1.0 / nCandidates


    SumAllPriors=0
    for x in Prior.values():
        SumAllPriors+=x

    b=1-d

    for loop in range(0, numloops):

        newRanks = {}
        for ngram in graph:

            #
            Sum =0

            weights= graph[ngram]

            for co_oc in graph[ngram]:

                # nominator
                nom = ranks[co_oc] * weights[co_oc]

                # denominator
                sumWeight = 0
                for co_ocWeights in graph[co_oc].values():
                    sumWeight+=co_ocWeights

                Sum+= nom/sumWeight

            newRank= d * Prior[ngram]/SumAllPriors + b+Sum

            newRanks[ngram] = newRank


        # for ng in newRanks:
        #     newRanks[ng]=newRanks[ng]/nCandidates
        ranks = newRanks

    ##normalization
    for ng in ranks:
        ranks[ng]=ranks[ng]/nCandidates

    return ranks





def createGraph(DocCandidates, candidatesBySentences):

    graph={}

    for ngram in DocCandidates:
        #print ngram
        co_oc={}

        for sentence in candidatesBySentences:
            if ngram in sentence:
               # print "IIIIIIIIN"
                for ngram2 in sentence:
                    if ngram2!=ngram:
                        #nCo_oc=1
                        if ngram2 not in co_oc:
                           # print "YEEEEEEE CO OC"
                            co_oc[ngram2]= 1
                        else:
                            co_oc[ngram2] = co_oc[ngram2] +1

        #print co_oc
        graph[ngram]= co_oc

    return graph



def extract_candidate_chunks(text, grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'):

    # # tokenize, POS-tag, and chunk using regular expressions
    chunker = nltk.chunk.regexp.RegexpParser(grammar)

    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))

    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    # join constituent chunk words into a single chunked phrase
    candidates = [' '.join(word for word, pos, chunk in group).lower()
                  for key, group in itertools.groupby(all_chunks, lambda (word, pos, chunk): chunk != 'O') if key]

    return [cand for cand in candidates]



def wordToNgrams(text, n, exact=True):
    return [" ".join(j) for j in zip(*[text[i:] for i in range(n)])]


def ngrams(text):

    finalCandidates = []

    text = "".join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()]).lower()
    tokensWithoutStopWords = [x for x in text.split() if x not in stopWords]

    finalCandidates += tokensWithoutStopWords
    finalCandidates += wordToNgrams(tokensWithoutStopWords, 2)
    finalCandidates += wordToNgrams(tokensWithoutStopWords, 3)

   # print finalCandidates

    return finalCandidates


###



def tok(text):

    text = "".join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()])
    tokens = nltk.word_tokenize(text)
    tokensWithoutStopWords = [x for x in tokens if x not in stopWords]

    tags = nltk.pos_tag(tokensWithoutStopWords)
    namedEntities = nltk.ne_chunk(tags, binary=True)

    tokensWithoutStopWords = ' '.join([c for c in tokensWithoutStopWords])

    # Limit to the regular expression
    #print "\nExtracting keyphrases ... "
    result= extract_candidate_chunks(tokensWithoutStopWords.encode('utf-8'))

    #choosing ngrams n=[1,3]
    final = [ngram for ngram in result if len(ngram.split())<=3]

    #print final
    return final


def tok_sent(text):

    sentences = []
    text = ''.join([c for c in text]).lower()
    text = [x for x in text.split() if x not in stopWords]
    sentences += [" ".join(text)]

    ### [0] porque o sent tokenize tem de receber uma string ###
    sentenceList = sent_tokenize(sentences[0])

    for i,sentence in enumerate(sentenceList):
        text="".join([c for c in sentence if c not in string.punctuation])
        text = ''.join([c for c in text if not c.isdigit()])
        sentenceList[i] =text


    # for sent in sentList:
    #      print sent

    return sentenceList


def dot_product(a, b):
    return sum([a[i]*b[i] for i in range(len(a))])


def decision( x, w, theta ):
    return (dot_product(x, w) > theta)


def perceptron(matrix):

    # Weights NEED to have the number of elements = to the number of features of matrix
    weights = [0,0,0]
    x = 0
    theta = 0
    converged = False


    while not converged:
        x += 1
        if x > 50:  # Being lazy for now
            converged = True
        for val in matrix.iteritems():
            # print val
            # print val[1][:2]
            # print val[1][-1]
            d = decision(val[1][:2], weights, theta)
            if d == val[1][-1]:
                continue
            elif d == False and val[1][-1] == True:
                theta -= 1
                for i in range(3):
                    weights[i] += val[1][i]

            elif d == True and val[1][-1] == False:
                theta += 1
                for i in range(3):
                    weights[i] -= val[1][i]


        # for key, val in matrix.iteritems():
        #     # Multiplies both matrices and checks if the value is bigger than theta
        #     # returns True if that happens or False otherwise
        #     d = decision(key, weights, theta)
        #     if d == val:
        #         continue
        #     elif d == False and val == True:
        #         theta -= 1
        #         for i in range(len(key)):
        #             weights[i] += key[i]
        #
        #     elif d == True and val == False:
        #         theta += 1
        #         for i in range(len(key)):
        #             weights[i] -= key[i]

    print "WEIGHTS:"
    print weights
    # Returns the ideal weights
    return weights



##############################################################
##############################################################


def extractKeyphrases( train, dataset, keys):

    NDocs = len(train)
    #ngrams per doc
    candidatesByDoc = {}

    #ngrams and tehir position value sentence relative ??
    candidatesDocSentenceWeight = {}

    #candidates per senteces per doc
    candidatesBySentence = {}

    print "Creating ngrams"
    docSentences = {}
    filteredTrain={}
    docsLen={}  ##number of words
    for doc in dataset:
        print doc

        docContent=""
        candidatesDoc = []

        # GET SENTENCES
        docSentences[doc] = tok_sent(dataset[doc])

        nSentences= len(docSentences[doc])
        ngram_pos={}

        candidatesBySentence[doc] = []
        lengthDoc=0

        # GET NGRAMS
        for i,sentence in enumerate(docSentences[doc]):
            candidates = []
            candidates += tok(sentence)


            #length
            lengthDoc+= len(sentence.split())

            docContent+=sentence

            # FOR EACH SENTENCE
            candidatesBySentence[doc] += [candidates]

            # FOR EACH DOC
            for ngram in candidates:
                if ngram not in candidatesDoc:
                    candidatesDoc.append(ngram)
                    ngram_pos[ngram]=(nSentences - i)/nSentences

        # print candidatesBySentence
        docsLen[doc]= lengthDoc
        candidatesByDoc[doc] = candidatesDoc
        candidatesDocSentenceWeight[doc]= ngram_pos

        # print "print candidates doc" + doc

        filteredTrain[doc]=docContent



    #CREATE GRAPHS
    graphs={}
    for doc in candidatesByDoc:
        graph={}
        graph= createGraph(candidatesByDoc[doc], candidatesBySentence[doc])
        graphs[doc]= graph


    # CALCULATE PRIORS

    print "YEYY TF"
    scoresTF = TF(candidatesByDoc, filteredTrain, docsLen)


    print "YEYY IDF"
    scoresIDF = IDF(candidatesByDoc, filteredTrain, NDocs)



    # avg length of the docs
    sumWords = 0
    for doc in docsLen:
        sumWords += docsLen[doc]
    avgDL = sumWords / NDocs

    print "YEYY BM25"
    ###Calculating BM25 --> PRIOR
    scoresBM25 = {}
    for doc in candidatesByDoc:
        scoreDoc={}
        for ngram in candidatesByDoc[doc]:
            scoreDoc[ngram]= BM25(scoresIDF[doc][ngram], scoresTF[doc][ngram], docsLen[doc], avgDL)
        scoresBM25[doc]=scoreDoc

    Priors = {}
    Priors['SENTENCE'] = candidatesDocSentenceWeight
    Priors['BM25'] = scoresBM25

    ##CALCULATE PAGERANK
    PageRankOriginal = {}
    PageRankImproved = {}
    for doc in candidatesByDoc:
        print doc
        pro = pagerank(graphs[doc], 0.15, 3)
        # print "PAGERANK ORIGINAL"
        # print sorted(pro.iteritems(), key=operator.itemgetter(1), reverse=True)
        PageRankOriginal[doc] = pro

        # print "PAGERANK IMPROVED -  SENTENCE AND CO-OCCURRENCE"
        prISC = pagerankImproved(graphs[doc], candidatesByDoc[doc], Priors['BM25'][doc], 0.15, 3)
        # print sorted(prISC.iteritems(), key=operator.itemgetter(1), reverse=True)
        # PageRankImproved[doc] = pri

        # print "PAGERANK IMPROVED -  BM25 AND CO-OCCURRENCE"
        prIBC = pagerankImproved(graphs[doc], candidatesByDoc[doc], Priors['SENTENCE'][doc], 0.15, 3)
        # print sorted(prIBC.iteritems(), key=operator.itemgetter(1), reverse=True)


    print"\nCreating Test Matrix ...\n"

    # List of keywords for each file in the colection
    keywordListByFile = []

    for keyFile in keys:
        # List of keywords in each file
        keywordList = []
        for line in keys[keyFile].splitlines():
            # Adds a list of the file keywords to the main(keywordListByFile) list
            keywordList += [line]
        keywordListByFile += [keywordList]


    # # Creates a matrix ---> dic { (feature1, feaature2,...): Val }
    # # Features ex: BM25, Pagerank, TF-Idf, ...
    # # Val = True if the ngram belongs to the keys of the doc False otherwise
    # Xmatrix = {}
    #
    # # pagerankImproved(graph, keywords, Prior, d, numloops)
    #
    # for i, doc in enumerate(candidatesByDoc):
    #     docMatrix = {}
    #     for n,ngram in enumerate(candidatesByDoc[doc]):
    #         # Calc Tf-Idf and BM25
    #         tfIdf_Score = scoresIDF[doc][ngram]* scoresTF[doc][ngram]
    #         BM25_Score = BM25(scoresIDF[doc][ngram], scoresTF[doc][ngram], docsLen[doc], avgDL)
    #         docPagerank = pagerankImproved(graphs[doc], candidatesByDoc[doc], Priors['BM25'][doc], 0.15, 10)
    #
    #         # If ngram belongs to the keys of that specific file = True or False Otherwise
    #         if ngram in keywordListByFile[i]:
    #             docMatrix[(tfIdf_Score, BM25_Score, docPagerank[ngram])] = True
    #             Xmatrix[doc] = docMatrix
    #         else:
    #             docMatrix[(tfIdf_Score, BM25_Score, docPagerank[ngram])] = False
    #             Xmatrix[doc] = docMatrix
    #
    #
    # print Xmatrix
    # print "\n\n\n"
    # for doc in Xmatrix:
    #     print doc
    #     print Xmatrix[doc]
    #
    # print "\n\n\n"

    Ymatrix = {}

    # pagerankImproved(graph, keywords, Prior, d, numloops)

    for i, doc in enumerate(candidatesByDoc):
        docMatrix = {}
        for n, ngram in enumerate(candidatesByDoc[doc]):
            # Calc Tf-Idf and BM25
            tfIdf_Score = scoresIDF[doc][ngram] * scoresTF[doc][ngram]
            BM25_Score = BM25(scoresIDF[doc][ngram], scoresTF[doc][ngram], docsLen[doc], avgDL)
            docPagerank = pagerankImproved(graphs[doc], candidatesByDoc[doc], Priors['BM25'][doc], 0.15, 10)
            if ngram in keywordListByFile[i]:
                docMatrix[ngram] = [tfIdf_Score, BM25_Score, docPagerank[ngram], True]
            else:
                docMatrix[ngram] = [tfIdf_Score, BM25_Score, docPagerank[ngram], False]
        Ymatrix[doc] = docMatrix



    print "\nPerceptron ...\n"



    weightsByDoc = {}
    for docMatrix in Ymatrix:
            weightsByDoc[docMatrix] = perceptron(Ymatrix[docMatrix])




    allTop5scores = {}
    for docMatrix in Ymatrix:
        scoresByDoc = {}
        for ngram in Ymatrix[docMatrix]:
            d = dot_product(Ymatrix[docMatrix][ngram][:3], weightsByDoc[docMatrix])
            scoresByDoc[ngram] = d
        allTop5scores[docMatrix] = dict(sorted(scoresByDoc.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])

    print "\nTop 5 keywords by document: ...\n"

    for doc in allTop5scores:
        print "\nDoc: " + doc + "\n"
        for ngram in allTop5scores[doc]:
            print ngram, " --> ", allTop5scores[doc][ngram]

# --------------------------------------------------------------------#

fileList = {}
keyList = {}
knownKeyphrases = []
predictedKeyphrases = []

# TRAIN DATA

train = fetch_20newsgroups(subset='train')
trainData = train.data[:10]


# Get relative path to documents
datasetPath = os.path.dirname(os.path.abspath(__file__)) + "\\maui-semeval2010-train\\";


# Get all documents in "documents" directory into fileList
# Import documents into fileList
for docName in os.listdir(datasetPath):
    if docName.endswith(".txt"):
        f = open(datasetPath + docName, 'r')
        fileList[docName.partition(".txt")[0]] = f.read().decode("utf-8").encode("ascii", "ignore")
        f.close()
    else:
        f = open(datasetPath + docName, 'r')
        keyList[docName.partition(".key")[0]] = f.read().decode("utf-8").encode("ascii", "ignore")
        f.close()


#print fileList.values()

inputDoc =  open("input2.txt", 'r')

inputCont = inputDoc.read()
inputCont = inputCont.decode("unicode_escape")

inputData={}
inputData["doc1"]= inputCont

keyphrases = extractKeyphrases(fileList, fileList, keyList)



