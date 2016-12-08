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



#CALCULO IDF
# N="docs in collection, nt=#docs with term
def IDF(N, nt):
    return log(1+(N-nt+0.5)/(nt+0.5))


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



def createGraph(DocCandidates, candidatesBySentences):

    graph={}

    for ngram in DocCandidates:
        # print ngram
        co_oc={}

        for sentence in candidatesBySentences:
            if ngram in sentence:
                for ngram2 in sentence:
                    if ngram2!=ngram:
                        #nCo_oc=1
                        if ngram2 not in co_oc:
                            co_oc[ngram2]= 1
                        else:
                            co_oc[ngram2] = co_oc[ngram2] +1

        # print co_oc
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
    tokens = word_tokenize(text)
    tokensWithoutStopWords = [x for x in tokens if x not in stopWords]

    tags = nltk.pos_tag(tokensWithoutStopWords)
    namedEntities = nltk.ne_chunk(tags, binary=True)

    tokensWithoutStopWords = ' '.join([c for c in tokensWithoutStopWords])

    # Limit to the regular expression
    #print "\nExtracting keyphrases ... "
    result= extract_candidate_chunks(tokensWithoutStopWords.encode('utf-8'))

    #choosing ngrams n=[1,3]
    final = [ngram for ngram in result if len(ngram.split())<=3]

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
    weights = [0,0]
    x = 0
    theta = 0
    converged = False

    print "\nPERCEPTRON\n"

    while not converged:
        x += 1
        if x > 50:  # Being lazy for now
            converged = True

        for key, val in matrix.iteritems():
            # Multiplies both matrices and checks if the value is bigger than theta
            # returns True if that happens or False otherwise
            d = decision(key, weights, theta)
            if d == val:
                continue
            elif d == False and val == True:
                theta -= 1
                for i in range(len(key)):
                    weights[i] += key[i]

            elif d == True and val == False:
                theta += 1
                for i in range(len(key)):
                    weights[i] -= key[i]
    # Returns the ideal weight and theta
    return weights, theta






# --------------------------------------------------------------------#


def extractKeyphrases( train, dataset, keys):

    NDocs = len(train)
    #ngrams per doc
    candidatesByDoc = {}

    #candidates per senteces per doc
    candidatesBySentence = {}


    docSentences = {}
    filteredTrain={}
    docsLen={}  ##number of words
    for doc in dataset:
#        print doc

        docContent=""
        candidatesDoc = []

        # GET SENTENCES
        docSentences[doc] = tok_sent(dataset[doc])

        candidatesBySentence[doc] = []
        lengthDoc=0

        # GET NGRAMS
        for sentence in docSentences[doc]:
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

        # print candidatesBySentence
        docsLen[doc]= lengthDoc
        candidatesByDoc[doc] = candidatesDoc
        # print "print candidates doc" + doc
        # print candidatesByDoc
        filteredTrain[doc]=docContent
   # print candidatesBySentence

    print "\nCreating Graphs...\n"

    #CREATE GRAPHS
    graphs={}
    for doc in candidatesByDoc:
        graph={}
        graph= createGraph(candidatesByDoc[doc], candidatesBySentence[doc])
        graphs[doc]= graph

    # CALCULATE PRIOR
    print "\nCalculating TF ...\n"
    ####CALCULATES TF FOR INPUT ####
    scoresTF = {}
    for doc in candidatesByDoc:
        scoresDoc= {}
        for ngram in candidatesByDoc[doc]:
            scoresDoc[ngram] = filteredTrain[doc].count(ngram)/docsLen[doc]

        scoresTF[doc]=scoresDoc

    # print scoresTF

    print "\nCalculating IDF ...\n"
    ####CALCULATES IDF FOR BACKGROUND COLLECTION ####
    scoresIDF = {}
    for doc in candidatesByDoc:
        docScores={}
        for ngram in candidatesByDoc[doc]:
            # print ngram
            nt = sum(1 for doc in filteredTrain if ngram in filteredTrain[doc]) #FALTA FILTERING TRAIN DATASET
            # print NDocs
            # print nt
            docScores[ngram] = IDF(NDocs, nt)
        scoresIDF[doc]=docScores

    # print scoresIDF

    # avg length of the docs
    sumWords = 0
    for doc in docsLen:
        sumWords += docsLen[doc]
    avgDL = sumWords / NDocs

    print "\nCalculating BM25 ...\n"
    ###Calculating BM25 --> PRIOR
    scoresBM25 = {}
    for doc in candidatesByDoc:
        scoreDoc={}
        for ngram in candidatesByDoc[doc]:
            scoresDoc[ngram]= BM25(scoresIDF[doc][ngram], scoresTF[doc][ngram], docsLen[doc], avgDL)
        scoresBM25[doc]=scoresDoc

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


    # Creates a matrix ---> dic { (feature1, feaature2,...): Val }
    # Features ex: BM25, Pagerank, TF-Idf, ...
    # Val = True if the ngram belongs to the keys of the doc False otherwise
    Xmatrix = {}

    for i, doc in enumerate(candidatesByDoc):
        for n,ngram in enumerate(candidatesByDoc[doc]):
            # Calc Tf-Idf and BM25
            tfIdf_Score = scoresIDF[doc][ngram]* scoresTF[doc][ngram]
            BM25_Score = BM25(scoresIDF[doc][ngram], scoresTF[doc][ngram], docsLen[doc], avgDL)

            # If ngram belongs to the keys of that specific file = True or False Otherwise
            if ngram in keywordListByFile[i]:
                Xmatrix[(tfIdf_Score, BM25_Score)] = True
            else:
                Xmatrix[(tfIdf_Score, BM25_Score)] = False


    print Xmatrix


    print "\nPerceptron\n"

    # Penso nao haver problema em usar o mesmo dataset como test
    trainMatrix = Xmatrix
    testMatrix = Xmatrix

    weights, theta = perceptron(trainMatrix)

    print "\nIdeal weights and theta: \n"
    print weights, theta

    print "\nPerceptron Result: \n"

    for key, val in testMatrix.iteritems():
        # Multiplies the Test matrix by the ideal weight and theta returned by perceptron
        # checking if the value is bigger than theta
        # returns True if that happens or False otherwise
        d = decision(key, weights, theta)
        answer = "Wrong! :("
        # If d = val it means that we correctly predictec that this specific ngram is a keyword
        # because it is also present in the keys of the file were it appears
        if d == val:
            answer = "Correct! :)"
        print answer, " --> ", key, d, val

    #CALCULATE PAGERANK
    # PageRank={}
    # for doc in candidatesByDoc:
    #     pr=pagerank(graphs[doc], candidatesByDoc[doc], scoresBM25[doc], graphs[doc], 0.15, 3):


# --------------------------------------------------------------------#

fileList = {}
keyList = {}
knownKeyphrases = []
predictedKeyphrases = []

# TRAIN DATA

train = fetch_20newsgroups(subset='train')
trainData = train.data[:10]


# Get relative path to documents
datasetPath = os.path.dirname(os.path.abspath(__file__)) + "\\documents\\";


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



