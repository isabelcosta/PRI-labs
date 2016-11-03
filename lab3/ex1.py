from sklearn.datasets import fetch_20newsgroups
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

#print train.data[:10]
print train.target[:10]
#print train.target_names

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(use_idf=False)
trainvec = vectorizer.fit_transform(train.data)
testvec = vectorizer.transform(test.data)

# print "-------------vectorizer---------------"
# print vectorizer
# print "-------------trainvec---------------"
# print trainvec
# print "-------------testvec---------------"
# print testvec

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(trainvec, train.target)
classes = classifier.predict(testvec)

print type(test.target)
print type(classes)
print test.target[:10]
print classes[:10]
from sklearn import metrics
print metrics.accuracy_score(test.target, classes)
# print metrics.classification_report(test.target, classes)
