#from __future__ import unicode_literals
from __future__ import division
from sklearn.datasets import fetch_20newsgroups
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk import ngrams

import operator


stopWords = list(stopwords.words('english'))



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

    return finalCandidates

def tok_sent(text):

    finalCandidates = []
    text = ''.join([c for c in text]).lower()
    text = [x for x in text.split() if x not in stopWords]
    finalCandidates += [" ".join(text)]

    ### [0] porque o sent tokenize tem de receber uma string ###
    sentenceList = sent_tokenize(finalCandidates[0])

    # for sent in sentList:
    #      print sent

    return sentenceList

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

def extractKeyphrases(text):


    print "\nFiltering document sentences ... "
    sentenceList = tok_sent(text)
    # print sentList



    for i, sent in enumerate(sentenceList):
        sentenceList[i] = "".join([c for c in sent if c not in string.punctuation])


    print "\nGetting ngrams ... "
    candidates = ngrams(text)
    # print candidates

    print "\nCreating graph ... \n"

    graph = {}
    graph2 = {}

    ### TEMOS DE TER EM CONTA O NURMERO DE COOCORRENCIAS ? OU BASTA SABER SE EXISTE COOCORRENCIA OU NAO ? ###

    # for ngram in candidates[:15]:
    #     elementList = []
    #     for sentence in sentenceList:
    #         elementList.append(sentence.count(ngram))
    #     graph[ngram] = elementList
    #
    # for element in graph:
    #     print element
    #     print graph[element]

    ### CREATE GRAPH ###

    for i, ngram in enumerate(candidates):
        elementList = []
        for n, sentence in enumerate(sentenceList):
            if sentence.count(ngram) != 0:
                for ngram2 in candidates:
                    if sentence.count(ngram2) != 0 and ngram2 != ngram and ngram2 not in elementList:
                        elementList += [ngram2]



        graph2[ngram] = elementList

    for element in graph2:
        print element
        print graph2[element]

    print "\nCalculating Pagerank ... "

    pagerankDic = pagerank(graph2,0.15,50)

    pagerankDicTop5 = dict(sorted(pagerankDic.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])

    print "\nPagerank Top 5:\n "

    for ngram in pagerankDicTop5:
        print ngram
        print pagerankDicTop5[ngram]


# --------------------------------------------------------------------#



print "\nGetting document ..."
#input doc that we want to extraxt keyphrases from
document = open("input.txt", 'r')
doc = document.read()
doc = doc.decode("unicode_escape")

keyphrases = extractKeyphrases(doc)





