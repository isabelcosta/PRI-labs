# Get relative path to documents
currentPath = os.path.dirname(os.path.abspath(__file__)) + "\documents\\";


# Get all documents in "documents" directory into fileList
# Import documents into fileList
for fileX in os.listdir(currentPath):
    if fileX.endswith(".txt"):
        f = open(currentPath + fileX, 'r')
        content = f.read()
        fileList += [content]
        f.close()

