import os

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import string
from lxml import etree
import operator
import webbrowser
from generate_html import generated_html_with_keyphrases

############### Global Variables ####################

stopWords = list(stopwords.words('english'))

############### Functions ###########################

def wordToNgrams(text, n, exact=True):
    return [" ".join(j) for j in zip(*[text[i:] for i in range(n)])]


def ngrams(text):

    finalCandidates = []

    # if c not in string.punctuation
    text = "".join([c for c in text])
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

    for i in range(0, numloops):
        print "Loop: " + str(i+1)
        newranks = {}
        for page in graph:
            newrank = d / npages
            for node in graph:
                if page in graph[node]:
                    newrank = newrank + (1 - d) * (ranks[node] / len(graph[node]))
            newranks[page] = newrank
        ranks = newranks
    return ranks

###################### Execution ##########################

content = []

root = etree.parse("http://rss.nytimes.com/services/xml/rss/nyt/World.xml")
for title in root.xpath('//title'):
    if title.text != None:
        content.append(title.text)

for description in root.xpath('//description'):
    if description.text != None:
        content.append(description.text)

content = ' '.join(content)

print "\nFiltering document sentences ... "
sentenceList = tok_sent(content)

candidates_per_phrase = []

for i, sent in enumerate(sentenceList):
    sentenceList[i] = "".join([c for c in sent if c not in string.punctuation])
    candidates_per_phrase += [ngrams(sentenceList[i])]

for i, sent in enumerate(sentenceList):
    #  if c not in string.punctuation
    sentenceList[i] = "".join([c for c in sent])


print "\nCreating graph ... "

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

print "\nCalculating PageRank ..."

keywordDic = pagerank(graph,0.15,1)

keywordDicTop50 = sorted(keywordDic.iteritems(), key=operator.itemgetter(1), reverse=True)[:50]

wordList = []
scoreList = []

for keyword in keywordDicTop50:
    wordList += [keyword[0]]
    scoreList += [str(keyword[1])]
    print keyword[0] + " --> " + str(keyword[1])

print "\nPagerank Top 10:\n "

for keyword in keywordDicTop50[:10]:
    print keyword[0] + " --> " + str(keyword[1])

wordListStr = ' '.join(wordList)

print "\nCreating HTML page ... \n"

f = open('Top5keywords.html', 'w')
message = generated_html_with_keyphrases(wordList, scoreList, wordListStr)
f.write(message.encode("utf-8"))
f.close()

filePath = os.path.dirname(os.path.abspath(__file__))

webbrowser.open_new_tab(filePath +'\\Top5keywords.html')