from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk

train = fetch_20newsgroups(subset='train')

def tok(text):
    text = "".join([c for c in text if c not in string.punctuation])
    text = ''.join([c for c in text if not c.isdigit()])
    tokens = nltk.word_tokenize(text)
    return tokens


vectorizer = TfidfVectorizer( use_idf=False,ngram_range=(1,2), stop_words = 'english', tokenizer=tok  )
trainvec = vectorizer.fit_transform(train.data)

document = open('candidates.txt', 'r')
doc = document.read()

testvec = vectorizer.transform( [doc])

feature_names = vectorizer.get_feature_names()
for i in testvec.nonzero()[1]:
    print feature_names[i] + ' : ' + testvec[0, i]