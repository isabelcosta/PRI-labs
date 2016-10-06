file = openfile(doc.txt, 'r')

doc_list = []

for line in file:

	doc_list = doc_list + [line]

words_in_doc = {}

id_doc = 0

for sentece in doc_list:

	for word in sentece.split():

		if word in words_in_doc

			words_in_doc[word][id_doc] + 1

		else 

			words_in_doc[word] = []


	id_doc += 1 





file.close()
