#from __future__ import unicode_literals
from __future__ import division
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
from nltk.corpus import stopwords
from math import log
from nltk import ngrams
import itertools
import re
import operator


stopWords = list(stopwords.words('english'))

#grammar=r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}'
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

    return final


def extractKeyphrases(train, test):

    trainSize= len(train)

    print "\nFiltering background collection ... "
    trainFiltered = []
    for doc in train:
        trainFiltered += tok(doc)

    print "\nFiltering document ... "
    candidates = tok(test)


    #treating input document
    inputText= "".join([c for c in test if c not in string.punctuation])
    inputText = ''.join([c for c in inputText if not c.isdigit()])
    tokensInput = nltk.word_tokenize(inputText)
    inputText = ' '.join([x for x in tokensInput if x not in stopWords])
    inputText= inputText.lower()
    inputText= inputText.encode('utf-8')


    ####CALCULATES TF FOR INPUT ####
    inputLen = len(test)
    scoresTF = {}
    for word in candidates:
        scoresTF[word]= inputText.count(word)/inputLen


    ####CALCULATES IDF FOR BACKGROUND COLLECTION ####
    scoresIDF = {}
    for word in candidates:
        nt= sum(1 for docContent in trainFiltered if word in docContent)
        scoresIDF[word]= IDF(trainSize, nt)


    # avg length of the docs
    sumWords = 0
    for doc in train:
        sumWords += len(doc)
    avgDL = sumWords / trainSize


    print "\nCalculating BM25 ... "
    scoresBM25= {}
    for word in candidates:
        scoresBM25[word]= BM25(scoresIDF[word], scoresTF[word],inputLen, avgDL)

    #TOP 5
    return sorted(scoresBM25.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]






#CALCULO BM25
#recebe o idef, o tf, D=doc length=(#words), avgdl=avg D
def BM25( idf, tf, D, avgdl):
    k=1.2
    b= 0.75
    top = tf *(k+1)
    aux = D/avgdl
    bottom = tf + k * (1 - b + b * aux)
    return idf * top/bottom



#CALCULO IDF
# N="docs in collection, nt=#docs with term
def IDF(N, nt):
    return log((N-nt+0.5)/(nt+0.5))




# --------------------------------------------------------------------#



print "\nGetting training collection ..."
# Get relative path to documents
#currentPath = os.path.dirname(os.path.abspath(__file__)) + "\documents\\";

# fileList = []
# # Get all documents in "documents" directory into fileList
# # Import documents into fileList
# for fileX in os.listdir(currentPath):
#     if fileX.endswith(".txt"):
#         f = open(currentPath + fileX, 'r')
#         content = f.read()
#         fileList += [str(content)]
#         f.close()

train = fetch_20newsgroups(subset='train')
trainData = train.data[:20]

print "\nGetting document ..."
#input doc that we want to extraxt keyphrases from
document = open("input.txt", 'r')
doc = document.read()
doc= unicode(doc, 'utf-8')

top5 = extractKeyphrases(trainData, doc)

print "\nTop 5 Keyphrase Candidates"
for tuple in top5:
    print "\t" + tuple[0] + " : " + str(tuple[1])


