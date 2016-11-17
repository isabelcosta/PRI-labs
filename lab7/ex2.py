
print "Getting document ... \n"
#input doc that we want to extraxt keyphrases from
document = open("links.txt", 'r')
doc = document.read()
doc = doc.splitlines()

dic = {}



for line in doc:
    print line
    key = line.split()[0]
    dic[key] = line.split()[1:]

print dic

