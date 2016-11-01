from __future__ import division, unicode_literals
from textblob import TextBlob
from nltk.corpus import stopwords
import math
import operator


doc1 = TextBlob(["sdfgfdgbdf"])

print doc1
print doc1.words