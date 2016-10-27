from whoosh.index import create_in
from whoosh.fields import *
schema = Schema(id = NUMERIC(stored=True), content=TEXT)
ix = create_in("indexdir", schema)
writer = ix.writer()
writer.add_document(id=1,
content=u"This is the first document we've added!")
writer.add_document(id=2,
content=u"The second one is even more interesting!")
writer.commit()