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


stopWords = list(stopwords.words('english'))


##############  FUNCTIONS  ##################

### Mean Precision


### TF IDF


### BM25


### PRIOR


### WEIGHT


### PageRank
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
                    # newrank = newrank + (1 - d) * (ranks[node] / len(graph[node]))
                    newrank = d * Prior(node)/AllPriors + (1-d)* (for each co_oc of node ((rank(co_oc)*Weight)/sumAllWeights of co-oc ) )
            newranks[page] = newrank
        ranks = newranks
    return ranks

### Graph
def createGraph2(candidates, sentencesList):

    graph={}

    for docID, docC in enumerate(candidates):
        print "DOC " + str(docID) + " ############################################################################################"

        for i, ngram in enumerate(docC):
            print "NGRAM : " +str(i) + " of DOC " + str(docID) +" ###############################################################"
            #print ngram
            elementList = []
            for j, docS in enumerate(sentencesList):

                for n, sentence in enumerate(docS):
                    print "SENTENCE " + str(n) + " in DOC " + str(j) + " ######################################3"
                    print sentence
                    if sentence.count(ngram) != 0:
                        for ngram2 in candidates[docID]:
                            print "CO OCCURECE NGRAM " + ngram2 + " in DOC " + str(docID) + " ######################################3"
                            if sentence.count(ngram2) != 0 and ngram2 != ngram and ngram2 not in elementList:
                                elementList += [ngram2]

            graph[ngram] = elementList
    return graph



def createGraph(DocCandidates, candidatesBySentences):

    graph={}

    for ngram in DocCandidates:
        print ngram
        co_oc=[]

        for sentence in candidatesBySentences:
            if ngram in sentence:
                print "IIIIIIIIN"
                for ngram2 in sentence:
                    if ngram2!=ngram and ngram2 not in co_oc:
                        print "YEEEEEEE CO OC"
                        co_oc.append(ngram2)

        print co_oc
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

    print final
    return final


def tok_sent(text):

    sentences = []
    text = ''.join([c for c in text]).lower()
    text = [x for x in text.split() if x not in stopWords]
    sentences += [" ".join(text)]

    ### [0] porque o sent tokenize tem de receber uma string ###
    sentenceList = sent_tokenize(sentences[0])

    # for sent in sentList:
    #      print sent

    return sentenceList






### Extract Keyphrases
def extractKeyphrases2(train, dataset):

    sentencesList=[]
    print "\nFiltering documents' sentences ... "
    # for doc in dataset:
    #     print doc
    #     sentencesList += [tok_sent(dataset[doc])]
    sentencesList = tok_sent(dataset)
    print sentencesList


    # for doc in sentencesList:
    #     for i, sentence in enumerate(doc):
    #         doc[i] = "".join([c for c in sentence if c not in string.punctuation])
    for i, sent in enumerate(sentencesList):
        sentencesList[i] = "".join([c for c in sent if c not in string.punctuation])



    print "\nGetting ngrams ... "

    # candidates=[]
    # for doc in dataset:
    #     print doc
    #     candidates +=[tok(dataset[doc])]
   # print candidates
    candidates=tok(dataset)


    print "NUMBER OF NGRAMS ############################3"
    # for i, pos in enumerate(candidates):
    #     print len(candidates[i])
    print len(candidates)

    print "\nCreating graph ... "
    graph = createGraph([candidates], [sentencesList])
    #
    # for element in graph:
    #     print element
    #     print graph[element]



# --------------------------------------------------------------------#


def extractKeyphrases( train, dataset):

    #ngrams per doc
    candidatesByDoc = {}

    #candidates per senteces per doc
    candidatesBySentence = {}

    print dataset

    # docSentences={}
    # for doc in dataset:
    #     print doc
    #
    #     candidatesDoc=[]
    #
    #     # GET SENTENCES
    #     docSentences[doc] = tok_sent(dataset[doc])
    #
    #     candidatesBySentence[doc] = []
    #
    #
    #     #GET NGRAMS
    #     for sentence in docSentences[doc]:
    #         candidates = []
    #         candidates += tok(sentence)
    #
    #         #FOR EACH SENTENCE
    #         candidatesBySentence[doc] += [candidates]
    #
    #         # FOR EACH DOC
    #         for ngram in candidates:
    #             if ngram not in candidatesDoc:
    #                 candidatesDoc.append(ngram)
    #
    #                 if ngram not in allCandidates:
    #                     allCandidates.append(ngram)
    #
    #
    #     #print candidatesBySentence
    #     #print all
    #
    #     candidatesByDoc[doc] = candidatesDoc




    docSentences = {}
    for doc in dataset:
        print doc

        candidatesDoc = []

        # GET SENTENCES
        docSentences[doc] = tok_sent(dataset[doc])

        candidatesBySentence[doc] = []

        # GET NGRAMS
        for sentence in docSentences[doc]:
            candidates = []
            candidates += tok(sentence)

            # FOR EACH SENTENCE
            candidatesBySentence[doc] += [candidates]

            # FOR EACH DOC
            for ngram in candidates:
                if ngram not in candidatesDoc:
                    candidatesDoc.append(ngram)

        # print candidatesBySentence

        candidatesByDoc[doc] = candidatesDoc

    print candidatesBySentence

    #CREATE GRAPHS
    graphs={}
    for doc in candidatesByDoc:
        graph={}
        graph= createGraph(candidatesByDoc[doc], candidatesBySentence[doc])
        graphs[doc]= graph


    for doc in graphs:
        print doc
        for ngram in graphs[doc]:
            print "    " + ngram + ": " + str(graphs[doc][ngram])


    #CALCULATE PAGERANK



























# --------------------------------------------------------------------#

fileList = {}
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
        #print docName.partition(".txt")[0]
        #print type(fileList[docName.partition(".txt")[0]])
        f.close()


#print fileList.values()

inputDoc =  open("input2.txt", 'r')

inputCont = inputDoc.read()
inputCont = inputCont.decode("unicode_escape")

inputData={}
inputData["doc1"]= inputCont

keyphrases = extractKeyphrases(fileList, fileList)






