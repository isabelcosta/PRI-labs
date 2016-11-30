from urllib2 import urlopen
import urlparse
import re
from bs4 import BeautifulSoup
import robotparser
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk import ngrams
import string

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



###################################################################

# http://rss.nytimes.com/services/xml/rss/nyt/Americas.xml

response = urlopen("http://rss.nytimes.com/services/xml/rss/nyt/Americas.xml")
parsed_html = BeautifulSoup(response, "lxml")
print parsed_html.prettify()

content = ""

print "\n\nTITLES\n"

for title in parsed_html.find_all("title"):
    content += title.text + " "


print "\n\nDISCRIPTIONS\n"

for description in parsed_html.find_all("description"):
    content += description.text + " "


print "\nFiltering document sentences ... "
sentenceList = tok_sent(content)
# print sentList



for i, sent in enumerate(sentenceList):
    #  if c not in string.punctuation
    sentenceList[i] = "".join([c for c in sent])


print "\nGetting ngrams ... "
candidates = ngrams(content)
# print candidates

print "\nCreating graph ... "

graph2 = {}

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

