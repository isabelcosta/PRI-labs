from whoosh.index import create_in
from whoosh.fields import *
from whoosh.index import open_dir
from whoosh.qparser import *

file = open("C:\Users\Fernando\Desktop\PRI\PRI-labs\lab5\pri_cfc.txt", 'r')


idList = []
relevantDocList = []

schema = Schema(id = NUMERIC(stored=True), content=TEXT)
ix = create_in("C:\Users\Fernando\Desktop\PRI\PRI-labs\lab5\IndexDir", schema)
writer = ix.writer()



for pos, line in enumerate(file):
    writer.add_document(id=pos, content=line.decode("utf-8"))


writer.commit()



ix = open_dir("C:\Users\Fernando\Desktop\PRI\PRI-labs\lab5\IndexDir")
with ix.searcher() as searcher:
    query = QueryParser("content", ix.schema, group=OrGroup).parse(u"distinguish between")
    results = searcher.search(query, limit=100)
    for r in results:
        idList += [r.get("id")]
        print r
    print "Number of results:", results.scored_length()

print "lista de ID's: " + str(idList)
file.close()