from urllib2 import urlopen
import urlparse
import re
import time
import robotparser

site = urlopen("http://www.ist.utl.pt")
content = site.read()

rp = robotparser.RobotFileParser("http://www.ist.utl.pt/robots.txt")
rp.read()

linksre = '<a\s.*?href=[\'"](.*?)[\'"].*?</a>'
links = re.findall(linksre, content, re.I)
url = urlparse.urljoin("http://www.ist.utl.pt/", "eventos/")

filteredLinks = []

for link in links:
     #print filteredLinks
     #print rp.can_fetch("*", link)
     if "http" in link and link not in filteredLinks and rp.can_fetch("*", link):
         #print link
         filteredLinks += [link]

site.close()

counter = len(filteredLinks)
print counter

for i, link in enumerate(filteredLinks):
     print i
     if i == counter:
         break

     links2 = []
     #time.sleep(1)
     site = urlopen(link)
     content = site.read()
     links2 = re.findall(linksre, content, re.I)
     site.close()
     for link in links2:
         #print filteredLinks
         if "http" in link and link not in filteredLinks and rp.can_fetch("*", link):
             print link
             filteredLinks += [link]






#print content
#print links
#print url




