from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import os
import operator

#train = fetch_20newsgroups(subset="train")

# --------------------------------------------------------------------#
## FUNCTIONS ##

def tok(text):
    text = "".join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()])
    tokens = nltk.word_tokenize(text)
    return tokens


def extractKeyphrases(train, test):

    print "Extracting keyphrases ... \n"

    vectorizer = TfidfVectorizer(use_idf=False, ngram_range=(1,2), stop_words="english", tokenizer=tok)
    trainvec = vectorizer.fit_transform(train)
    testvec = vectorizer.transform([test.decode('utf-8')])

    feature_names = vectorizer.get_feature_names()

    scores = {}
    for i in testvec.nonzero()[1]:
        scores[str( feature_names[i])] = str( testvec[0, i])

    topScores = sorted(scores.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]

    top ={}
    for i in topScores:
        top[i[0]] = i[1]

    return top


# --------------------------------------------------------------------#



print "Getting training collection .."
# Get relative path to documents
currentPath = os.path.dirname(os.path.abspath(__file__)) + "\documents\\";

fileList = []
# Get all documents in "documents" directory into fileList
# Import documents into fileList
for fileX in os.listdir(currentPath):
    if fileX.endswith(".txt"):
        f = open(currentPath + fileX, 'r')
        content = f.read()
        fileList += [content]
        f.close()

print "Getting document .."
#input doc that we want to extraxt keyphrases from
document = open("input.txt", 'r')
doc = document.read()


top5 = extractKeyphrases(fileList, doc)

print "Top 5 Keyphrase Candidates"
for word in top5:
    print "\t" + word + " : " + top5[word]

