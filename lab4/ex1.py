import nltk


from sklearn.datasets import fetch_20newsgroups
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

#print train.data[:10]
# print train.target[:10]
# print train.target_names

print train.data[1]

listaPalavras = []

for sentence in nltk.sent_tokenize(train.data[1]):
    listaPalavras = nltk.word_tokenize(sentence)
    print nltk.pos_tag(listaPalavras)

