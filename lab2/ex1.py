file = open("docs.txt", 'r')

doc_list = []

# Each line represents a document
# List index can be the id
for line in file:
	doc_list = doc_list + [line]

# {word: [[ docID, occurrencesInDoc ], ...]}
words_in_doc = {}

id_doc = 0
found_doc = False
for doc in doc_list:
	for word in doc.split():
		if word in words_in_doc:
			for docs_with_word in words_in_doc[word]:
				if docs_with_word[0] == id_doc:
					docs_with_word[1] = docs_with_word[1] + 1
					found_doc = True
			if not found_doc:
				words_in_doc[word] += [[id_doc, 1]]
		else:
			words_in_doc[word] = [] + [[id_doc, 1]]
	found_doc = False
	id_doc += 1


for word in words_in_doc:
	print word
	print words_in_doc[word]

""""""
file.close()
