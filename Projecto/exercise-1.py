from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import string
from nltk.corpus import stopwords
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

    print "Extracting keyphrases ... \n"
    vectorizer = TfidfVectorizer(use_idf=False, ngram_range=(1,2), stop_words=stopWords, tokenizer=tok)
    trainvec = vectorizer.fit_transform(train)
    testvec = vectorizer.transform(test)

    previousDoc = 0

    feature_names = vectorizer.get_feature_names()
    scores = []
    docScores = {}
    row, columns = testvec.nonzero()
    for doc, word in zip(row, columns):

        docScores[feature_names[word]] = str(testvec[doc, word])

        if previousDoc != doc:
            currentSortedScore = sorted(docScores.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]
            #print "Doc: " + str(previousDoc) + " " + str(currentSortedScore)

            scores += [currentSortedScore]
            docScores = {}

        previousDoc = doc


    currentSortedScore = sorted(docScores.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]
    scores += [currentSortedScore]
    #print "Doc: " + str(previousDoc) + " " + str(currentSortedScore)

    top =[]
    for list in scores:
        topAux={}
        for tuple in list:
            topAux[tuple[0]] = tuple[1]
        top+= [topAux]

    return top


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

