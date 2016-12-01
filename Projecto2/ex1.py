#from __future__ import unicode_literals
from __future__ import division
from sklearn.datasets import fetch_20newsgroups
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk import ngrams

import operator

stopWords = list(stopwords.words('english'))

"""
    input:
        text => sentence without stopwords and punctuation (string)
        n => degree of n-gram
"""
def wordToNgrams(text, n, exact=True):
    return [" ".join(j) for j in zip(*[text[i:] for i in range(n)])]

"""

    text => a single sentence (string)
"""
def ngrams(text):

    finalCandidates = []

    text_without_punctuation = "".join([c for c in text if c not in string.punctuation])
    text_lower_case = ''.join([c for c in text_without_punctuation if not c.isdigit()]).lower()
    tokensWithoutStopWords = [x for x in text_lower_case.split() if x not in stopWords]

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
        print loop
        newranks = {}
        for page in graph:
            newrank = d / npages
            for node in graph:
                if page in graph[node]:
                    newrank = newrank + (1 - d) * (ranks[node] / len(graph[node]))
            newranks[page] = newrank
        ranks = newranks
    return ranks

def extractKeyphrases(document):

    print "\nFiltering document sentences ... "
    sentenceList = tok_sent(document)
    # print sentenceList

    candidates = []
    # stores candidates per phrase
    candidates_per_phrase = []

    for i, sent in enumerate(sentenceList):
        sentenceList[i] = "".join([c for c in sent if c not in string.punctuation])
        print "\nGetting ngrams for phrase " + str(i) + "... "
        candidates += ngrams(sentenceList[i])
        candidates_per_phrase += [ngrams(sentenceList[i])]

    # print candidates

    print "\nCreating graph ... \n"

    graph = {}

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

    for phrase_n_grams in candidates_per_phrase:
        for n_gram in phrase_n_grams:
            if n_gram in graph:
                for other_gram in phrase_n_grams:
                    if n_gram != other_gram and other_gram not in graph[n_gram]:
                        graph[n_gram] += [other_gram]
            else:
                graph[n_gram] = []
                for other_gram in phrase_n_grams:
                    if n_gram != other_gram:
                        graph[n_gram] += [other_gram]


    """
    phrase: era uma vez. abc vez.

    graph = {
                'era': ['uma', 'vez'],
                'uma': ['era', 'vez'],
                'vez': ['abc', 'uma', 'era'],
                'abc': ['vez']
            }

    """
    # for element in graph:
    #     print element
    #     print graph[element]

    print "\nCalculating Pagerank ... "

    pagerankDic = pagerank(graph,0.15,50)

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





