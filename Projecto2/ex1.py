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
    output:
        list of n-grams of a single sentence
"""
def wordToNgrams(text, n, exact=True):
    return [" ".join(j) for j in zip(*[text[i:] for i in range(n)])]

"""
    input:
        text => a single sentence (string)
    output:
        final_candidates => all n-grams from a sentence
"""
def ngrams(text):
    final_candidates = []

    text_without_punctuation = "".join([c for c in text if c not in string.punctuation])
    text_lower_case = ''.join([c for c in text_without_punctuation if not c.isdigit()]).lower()
    tokensWithoutStopWords = [x for x in text_lower_case.split() if x not in stopWords]

    final_candidates += tokensWithoutStopWords
    final_candidates += wordToNgrams(tokensWithoutStopWords, 2)
    final_candidates += wordToNgrams(tokensWithoutStopWords, 3)

    return final_candidates

"""
    Splits a document into a list of sentences
    input:
        text => text (string) from a document
    output:
        sentence_list =>  list of sentences
"""
def tok_sent(text):

    final_candidates = []

    text_lowered = ''.join([c for c in text]).lower()
    text_with_out_stop_words = [x for x in text_lowered.split() if x not in stopWords]

    final_candidates += [" ".join(text_with_out_stop_words)]

    ### [0] porque o sent tokenize tem de receber uma string ###
    sentence_list = sent_tokenize(final_candidates[0])

    # for sent in sentList:
    #      print sent

    return sentence_list

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

def extractKeyphrases(document):

    print "Filtering document sentences ... "
    sentence_list = tok_sent(document)
    # print sentenceList

    #candidates = []
    # stores candidates per phrase
    candidates_per_phrase = []

    for i, sent in enumerate(sentence_list):
        sentence_list[i] = "".join([c for c in sent if c not in string.punctuation])
        print "Getting ngrams for phrase " + str(i) + "... "
        #candidates += ngrams(sentenceList[i])
        candidates_per_phrase += [ngrams(sentence_list[i])]

    # print candidates

    print "Creating graph ..."

    graph = {}
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

    print "Calculating Pagerank ... "

    pagerankDic = pagerank(graph,0.15,50)

    pagerankDicTop5 = dict(sorted(pagerankDic.iteritems(), key=operator.itemgetter(1), reverse=True)[:5])

    print "\nPagerank Top 5:\n"

    for ngram in pagerankDicTop5:
        print ngram + " : " + str(pagerankDicTop5[ngram])

""" Execution """

print "\nGetting document ..."
#input doc that we want to extraxt keyphrases from
document = open("input.txt", 'r')
doc = document.read()
doc = doc.decode("unicode_escape")

keyphrases = extractKeyphrases(doc)
