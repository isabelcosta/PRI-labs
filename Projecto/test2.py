from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import os
import operator

#train = fetch_20newsgroups(subset="train")
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

document = open("input.txt", 'r')
doc = document.read()

keyList = []
for fileX in os.listdir(currentPath):
    if fileX.endswith(".key"):
        f = open(currentPath + fileX, 'r')
        content = f.read()
        keyList += [content]
        f.close()

def tok(text):
    text = "".join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()])
    tokens = nltk.word_tokenize(text)
    return tokens

# print train.data[:10]
#print train.target[:10]
# print train.target_names

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(use_idf=False, ngram_range=(1,2), stop_words="english", tokenizer=tok)
trainvec = vectorizer.fit_transform(fileList)
testvec = vectorizer.transform([doc.decode('utf-8')])




feature_names = vectorizer.get_feature_names()

scores = {}

for i in testvec.nonzero()[1]:
    scores[str( feature_names[i])] = str( testvec[0, i])

print sorted(scores.iteritems(), key=operator.itemgetter(1), reverse=True)[:5]
