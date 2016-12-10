#from __future__ import unicode_literals
from __future__ import division
from sklearn.datasets import fetch_20newsgroups
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import operator
import itertools
import os
from math import log


stopWords = list(stopwords.words('english'))


##############  FUNCTIONS  ##################

### Mean Precision

def calc_mean_avg_precision(all_precisions):
    mean_avg_precision =  sum(all_precisions)/len(all_precisions)
    print "Mean Average Precision: \t" + str(mean_avg_precision)
    return mean_avg_precision


def precisonCalc(knownKeyphrases, predictedKeyphrases):
    tp = sum(1 for key in knownKeyphrases if key in predictedKeyphrases)
    fp = len(predictedKeyphrases) - tp

    return float(tp / (tp + fp))



#CALCULO IDF
# N="docs in collection, nt=#docs with term
def IDF(candidatesByDoc, filteredTrain, NDocs):
    print "idf"
    scoresIDF = {}
    for doc in candidatesByDoc:
        print doc
        docScores={}
        for ngram in candidatesByDoc[doc]:
            #print ngram
            nt = sum(1 for doc1 in filteredTrain if ngram in filteredTrain[doc]) #FALTA FILTERING TRAIN DATASET
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


###



def tok(text):

    #text = "".join([c for c in text if c not in string.punctuation])
    #text = ''.join([c for c in text if not c.isdigit()])
    tokensWithoutStopWords = nltk.word_tokenize(text)
    #tokensWithoutStopWords = [x for x in tokens if x not in stopWords]

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

    return sentenceList




##############################################################
##############################################################


def extractKeyphrases( train, dataset):

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
            #print i
            candidates += tok(sentence)


            #length
            lengthDoc+= len(sentence.split())

            docContent+=sentence

            # FOR EACH SENTENCE
            candidatesBySentence[doc] += [candidates]

            # FOR EACH DOC
           # print "ngrama and weight"
            for ngram in candidates:
                if ngram not in candidatesDoc:
                    candidatesDoc.append(ngram)
                    ngram_pos[ngram]=(nSentences - i)/nSentences
                # candidatesDoc.append(ngram)
                # ngram_pos[ngram] = (nSentences - i) / nSentences





        # print candidatesBySentence
        docsLen[doc]= lengthDoc
        candidatesByDoc[doc] = candidatesDoc
        candidatesDocSentenceWeight[doc]= ngram_pos

        # print "print candidates doc" + doc

        filteredTrain[doc]=docContent



    #CREATE GRAPHS
    print "Creating graphs.."
    graphs={}
    for doc in candidatesByDoc:
        print doc
        graph={}
        graph= createGraph(candidatesByDoc[doc], candidatesBySentence[doc])
        graphs[doc]= graph

    # for doc in graphs:
    #     print doc
    #     for ngram in graphs[doc]:
    #         print "    " + ngram + ": " + str(graphs[doc][ngram])




    # CALCULATE PRIORS

    print "TF and IDF calculations"
    scoresTF = TF(candidatesByDoc, filteredTrain, docsLen)
    scoresIDF = IDF(candidatesByDoc, filteredTrain, NDocs)



    # avg length of the docs
    sumWords = 0
    for doc in docsLen:
        sumWords += docsLen[doc]
    avgDL = sumWords / NDocs

    print "BM25 calculation"
    ###Calculating BM25 --> PRIOR
    scoresBM25 = {}
    for doc in candidatesByDoc:
        scoreDoc={}
        for ngram in candidatesByDoc[doc]:
            scoreDoc[ngram]= BM25(scoresIDF[doc][ngram], scoresTF[doc][ngram], docsLen[doc], avgDL)
        scoresBM25[doc]=scoreDoc

    # for doc in scoresBM25:
    #     print doc
    #     for ngram in scoresBM25[doc]:
    #         print "    " + ngram + ": " + str(scoresBM25[doc][ngram])


    Priors={}
    Priors['SENTENCE'] = candidatesDocSentenceWeight
    Priors['BM25']=scoresBM25




    ##CALCULATE PAGERANK
    ## PageRank= [doc][type PageRank][keyphrases]
    PageRank={}
    print "PAGERANKS"
    for doc in candidatesByDoc:
        prDoc={}
        print doc

       # print "PAGERANK ORIGINAL"
        pro=pagerank(graphs[doc],   0.15, 3)
        #print sorted(pro.iteritems(), key=operator.itemgetter(1), reverse=True)
        prDoc['ORIGINAL'] = sorted(pro.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]

       # print "PAGERANK IMPROVED -  SENTENCE AND CO-OCCURRENCE"
        prISC = pagerankImproved(graphs[doc], candidatesByDoc[doc], Priors['BM25'][doc], 0.15, 3)
        #print sorted(prISC.iteritems(), key=operator.itemgetter(1), reverse=True)
        prDoc['IMPROVED - SENTENCE PRIOR AND CO-OCCURENCE WEIGHT'] = sorted(prISC.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]

       # print "PAGERANK IMPROVED -  BM25 AND CO-OCCURRENCE"
        prIBC = pagerankImproved(graphs[doc], candidatesByDoc[doc], Priors['SENTENCE'][doc], 0.15, 3)
       # print sorted(prIBC.iteritems(), key=operator.itemgetter(1), reverse=True)
        prDoc['IMPROVED - BM25 PRIOR AND CO-OCCURENCE WEIGHT'] = sorted(prIBC.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]

        PageRank[doc] = prDoc


    return PageRank


# --------------------------------------------------------------------#

fileList = {}
knownKeyphrases = []
predictedKeyphrases = []

# TRAIN DATA

train = fetch_20newsgroups(subset='train')
trainData = train.data[:10]


# Get relative path to documents
datasetPath = os.path.dirname(os.path.abspath(__file__)) + "\\documents\\";
trainPath = os.path.dirname(os.path.abspath(__file__)) + "\\maui-semeval2010-train\\";

# Get all documents in "documents" directory into fileList
# Import documents into fileList
for docName in os.listdir(datasetPath):
    if docName.endswith(".txt"):
        f = open(datasetPath + docName, 'r')
        fileList[docName.partition(".txt")[0]] = f.read().decode("utf-8").encode("ascii", "ignore")
        #print docName.partition(".txt")[0]
        #print type(fileList[docName.partition(".txt")[0]])
        f.close()


#print fileList.values()

inputDoc =  open("input2.txt", 'r')

inputCont = inputDoc.read()
inputCont = inputCont.decode("unicode_escape")

inputData={}
inputData["Nihilism Wikipedia"]= inputCont

keyphrasesPageRanks = extractKeyphrases(fileList, fileList)




#Get known Kephrases for each doc
KnownKeyphrases = {}
for docName in fileList:

    docKeyphrases = []
    knownKeyphrases=[]

    # Get all document's known keyphrases in "keys" directory into knownKeyphrases
    # Import documents into fileList0
    fKeys = open(datasetPath + docName + ".key", 'r')
    docKeyphrases = fKeys.read()
    fKeys.close()

    knownKeyphrases = docKeyphrases.splitlines()
    knownKeyphrases = [x.decode("unicode_escape") for x in knownKeyphrases]

    #predictedKeyphrases = calculatedKeyScores[nDocs].keys()
    # print knownKeyphrases
    # print predictedKeyphrases

    # Performance metrics - Calculate precision, recall, F1 scores and mean average precision
    #allPrecisions += [calc_precision_recall_f1_score(docName, knownKeyphrases, predictedKeyphrases)[0]]

    KnownKeyphrases[docName]= knownKeyphrases



for doc in keyphrasesPageRanks:
    print "Doc: " + doc
    print " --- Page Rank ---"

    for type in keyphrasesPageRanks[doc]:
        print type

        print keyphrasesPageRanks[doc][type]

    print " --- Known keys ---"
    print KnownKeyphrases[doc]

    print "\n"


## PRECISIONS[TYPE RANK][[list scores][avg]]

Scores={}
Scores['All']=[]
Precisions={}
Precisions['ORIGINAL']= Scores
Precisions['IMPROVED - SENTENCE PRIOR AND CO-OCCURENCE WEIGHT']= Scores
Precisions['IMPROVED - BM25 PRIOR AND CO-OCCURENCE WEIGHT']= Scores

for doc in keyphrasesPageRanks:
    #docScores={}

    for typePageRank in keyphrasesPageRanks[doc]:

        # scores={}
        # scores = calc_precision_recall_f1_score(doc, keyphrasesPageRanks[doc][typePageRank].keys(), KnownKeyphrases[doc] )
        extracted=[]
        for elem in keyphrasesPageRanks[doc][typePageRank]:
            extracted+=[elem[0]]

        #print extracted

        precision= precisonCalc( extracted, KnownKeyphrases[doc] )

        Precisions[typePageRank]['All']+=[precision]



for type in Precisions:
    print type
    print "Values: " + str(Precisions[type]['All'])
    Precisions[type]['Mean Avg Precision'] = calc_mean_avg_precision(Precisions[type]['All'])
    print "Mean Avg Precision: " + str(Precisions[type]['Mean Avg Precision'])

    print "\n"






