from __future__ import division, unicode_literals
from textblob import TextBlob

from nltk.corpus import stopwords


import math
import os

i = 1
fileList = []
documentWordList = []
documentBiGramList = []



currentPath = os.path.dirname(os.path.abspath(__file__)) + "\documents\\";

for file in os.listdir(currentPath):
    if file.endswith(".txt"):
        f = open(currentPath + file, 'r')
        content = f.read()
        fileList += [content]
        f.close()

#para cada documento da lista de docs
for file in fileList:
    #tornar conteudo num textblob
    docContent = TextBlob(file.decode('unicode_escape'))
    stop_words = set(stopwords.words('english'))
    # retira stopwords do content do documento
    document_without_stop_words = list(set(docContent.words) - stop_words)

    documentWordList += [document_without_stop_words]
#    print document_without_stop_words
#     documentSentence = " ".join(document_without_stop_words)




# for document in documentList:
#     print document.words

def tf(word, docContent):
    return docContent.words.count(word) / len(docContent.words)

def n_containing(word, documentList):
    return sum(1 for docContent in documentList if word in docContent.words)

def idf(word, documentList):
    return math.log(len(documentList) / (1 + n_containing(word, documentList)))

def tfidf(word, docContent, documentList):
    return tf(word, docContent) * idf(word, documentList)


doc1 = TextBlob("ola!!!, awsegdas, fewasdga!!! adeus.........the............")



#doc2 = TextBlob(train.data[2])
# doc3 = TextBlob(train.data[3])
#
#
# documentList = [doc1, doc2, doc3]
stop_words = set(stopwords.words('english'))
doc1 = set(doc1.words) - stop_words
print doc1
#
# #
# print idf("guy", documentFist)
#
# print doc1.ngrams(n=2)

# list1 = ['1', '2', '3']
# str1 = ' '.join(list1)
#
# print str1