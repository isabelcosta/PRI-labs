from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import string
import itertools
from nltk.corpus import stopwords
import re
import operator

#train = fetch_20newsgroups(subset='train')

# --------------------------------------------------------------------#
## FUNCTIONS ##

stopWords = list(stopwords.words('english'))

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

    print extract_candidate_chunks(text)
    #
    # print "TAGS: " + str(tags) + "\n"

    #FILTRAR APENAS TRIGRAMAS tops
    return extract_candidate_chunks(str(tokensWithoutStopWords))


def extractKeyphrases(train, test):

    trainResults = []
    for doc in train:
        trainResults += tok(doc)

    testResults = tok(test)

    







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
