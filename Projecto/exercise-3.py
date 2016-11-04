from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import string
from nltk.corpus import stopwords
from math import log
import os
import operator

#train = fetch_20newsgroups(subset='train')

# --------------------------------------------------------------------#
## FUNCTIONS ##
stopWords = list(stopwords.words('english'))
def tok(text):
    text = "".join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()])
    tokens = nltk.word_tokenize(text)



    return tokens


def extractKeyphrases(train, test):

    trainSize= len(train)

    print "Extracting keyphrases ... \n"
    vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,2), stop_words=stopWords, tokenizer=tok)


    trainvec = vectorizer.fit_transform(train)




    # transforms de test doc in tokens
    candidates = tok(test)


    ####CALCULATES TF FOR INPUT ####
    inputContent = str(test)
    inputLen = len(inputContent)
    scoresTF = {}
    for word in candidates:
        scoresTF[word]= inputContent.count(word)/inputLen



    # avg length of the docs
    sumWords=0
    for doc in train:
        sumWords+=len(doc)

    avgDL = sumWords / trainSize




    scoresIDF = {}

    for word in candidates:
        nt= sum(1 for docContent in train if word in docContent)
        scoresIDF[word]= IDF(trainSize, nt)



    scoresBM25= {}
    for word in candidates:
        scoresBM25[word]= BM25(scoresIDF[word], scoresTF[word],inputLen, avgDL)





#CALCULO BM25
#recebe o idef, o tf, D=doc length=(#words), avgdl=avg D
def BM25( idf, tf, D, avgdl):

    k=1,2
    b= 0.75

    return (tf*(k+1))/(tf + k*(1-b+b*D/avgdl))



#CALCULO IDF
# N="docs in collection, nt=#docs with term
def IDF(N, nt):
    return log((N-nt+0.5)/(nt+0.5))



# --------------------------------------------------------------------#



print "Getting training collection .."
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

print "Getting document .."
#input doc that we want to extraxt keyphrases from
document = open("input.txt", 'r')
doc = [document.read()]

top5 = extractKeyphrases(trainData, doc)

print "Top 5 Keyphrase Candidates"
for enum, doc in enumerate(top5):
    for word in doc:
        print "\t" + word + " : " + str(doc[word])

