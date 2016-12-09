import os

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk import ngrams
import string
from lxml import etree
import operator
import webbrowser
import csv

stopWords = list(stopwords.words('english'))

def wordToNgrams(text, n, exact=True):
    return [" ".join(j) for j in zip(*[text[i:] for i in range(n)])]


def ngrams(text):

    finalCandidates = []

    # if c not in string.punctuation
    text = "".join([c for c in text])
    text = ''.join([c for c in text if not c.isdigit()]).lower()
    tokensWithoutStopWords = [x for x in text.split() if x not in stopWords]

    finalCandidates += tokensWithoutStopWords
    finalCandidates += wordToNgrams(tokensWithoutStopWords, 2)
    finalCandidates += wordToNgrams(tokensWithoutStopWords, 3)

    return finalCandidates

def tok_sent(text):

    finalCandidates = []
    text = ''.join([c for c in text]).lower()
    text = [x for x in text.split() if x not in stopWords]
    finalCandidates += [" ".join(text)]

    ### [0] porque o sent tokenize tem de receber uma string ###
    sentenceList = sent_tokenize(finalCandidates[0])

    # for sent in sentList:
    #      print sent

    return sentenceList

def pagerank(graph, d, numloops):

    ranks = {}
    npages = len(graph)
    for page in graph:
        ranks[page] = 1.0 / npages

    for i in range(0, numloops):
        print "Loop: " + str(i+1)
        newranks = {}
        for page in graph:
            newrank = d / npages
            for node in graph:
                if page in graph[node]:
                    newrank = newrank + (1 - d) * (ranks[node] / len(graph[node]))
            newranks[page] = newrank
        ranks = newranks
    return ranks

###################################################################

content = []

root = etree.parse("http://rss.nytimes.com/services/xml/rss/nyt/World.xml")
for title in root.xpath('//title'):
    # print title.text
    if title.text != None:
        content.append(title.text)

for description in root.xpath('//description'):
    # print description.text
    if description.text != None:
        content.append(description.text)

content = ' '.join(content)
# print content

print "\nFiltering document sentences ... "
sentenceList = tok_sent(content)
# print sentenceList

candidates = []
# stores candidates per phrase
candidates_per_phrase = []

for i, sent in enumerate(sentenceList):
    sentenceList[i] = "".join([c for c in sent if c not in string.punctuation])
    # print "\nGetting ngrams for phrase " + str(i) + "... "
    candidates += ngrams(sentenceList[i])
    candidates_per_phrase += [ngrams(sentenceList[i])]

for i, sent in enumerate(sentenceList):
    #  if c not in string.punctuation
    sentenceList[i] = "".join([c for c in sent])


print "\nCreating graph ... "

graph = {}

for phrase_n_grams in candidates_per_phrase:
    for n_gram in phrase_n_grams:
        if n_gram in graph:
            for other_gram in phrase_n_grams:
                if n_gram != other_gram and other_gram not in graph[n_gram]:
                    graph[n_gram] += [other_gram]
        else:
            graph[n_gram] = []
            for other_gram in phrase_n_grams:
                if n_gram != other_gram:
                    graph[n_gram] += [other_gram]

print "\nCalculating Pagerank ..."

keywordDic = pagerank(graph,0.15,1)

keywordDicTop10 = sorted(keywordDic.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]
keywordDicTop50 = sorted(keywordDic.iteritems(), key=operator.itemgetter(1), reverse=True)[:50]


wordList = []
scoreList = []


for keyword in keywordDicTop50:
    wordList += [keyword[0]]
    scoreList += [str(keyword[1])]
    print keyword[0] + " --> " + str(keyword[1])

print "\nPagerank Top 10:\n "

for keyword in keywordDicTop50[:10]:
    print keyword[0] + " --> " + str(keyword[1])

wordListStr = ' '.join(wordList)

# print wordList
# print scoreList

# print "\nCreating CSV ..."
#
#
# with open('Top10keywords.csv', 'wb') as f:  # Just use 'w' mode in 3.x
#     w = csv.writer(f)
#     w.writerow(('Keyword', 'Score'))
#     w.writerows(keywordDicTop10.items())



# for ngram in teste:
#     print ngram
#     print keywordDicTop10[ngram]

print "\nCreating HTML page ... \n"

f = open('Top5keywords.html','w')

message = """ <!DOCTYPE html>
<meta charset="utf-8">
<style>
div#title {
    font-family: "Lucida Console";
    color: green;
    width: 36%;
    padding-left: 2%;
    float: left;

}
div#title2 {
    font-family: "Lucida Console";
    color: green;
    width: 62%;
    float: right;

}
div#table {
    width: 62%;
    height: 500px;
    float: right;
}

</style>
</head>
<body>
  <div id="title">
  <h3>Top 50 Keywords</h3>
  </div>
  <div id="title2">
  <h3>Top 10 Keywords</h3>
  </div>
  <div id="table">
    <style type="text/css">
      .tg  {border-collapse:collapse;border-spacing:0;border-color:#aaa;}
      .tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#aaa;color:#333;background-color:#fff;}
      .tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:#aaa;color:#fff;background-color:#f38630;}
      .tg .tg-baqh{text-align:center;vertical-align:top}
      .tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
      </style>
      <table class="tg">
        <tr>
          <th class="tg-amwm"> Keywords </th>
          <th class="tg-amwm"> Score </th>
        </tr>
        <tr>
          <td class="tg-baqh">""" + wordList[0] + """</td>
          <td class="tg-baqh">""" + scoreList[0] + """</td>
        </tr>
        <tr>
          <td class="tg-baqh">""" + wordList[1] + """</td>
          <td class="tg-baqh">""" + scoreList[1] + """</td>
        </tr>
        <tr>
          <td class="tg-baqh">""" + wordList[2] + """</td>
          <td class="tg-baqh">""" + scoreList[2] + """</td>
        </tr>
        <tr>
          <td class="tg-baqh">""" + wordList[3] + """</td>
          <td class="tg-baqh">""" + scoreList[3] + """</td>
        </tr>
        <tr>
          <td class="tg-baqh">""" + wordList[4] + """</td>
          <td class="tg-baqh">""" + scoreList[4] + """</td>
        </tr>
          <tr>
          <td class="tg-baqh">""" + wordList[5] + """</td>
          <td class="tg-baqh">""" + scoreList[5] + """</td>
        </tr>
        <tr>
          <td class="tg-baqh">""" + wordList[6] + """</td>
          <td class="tg-baqh">""" + scoreList[6] + """</td>
        </tr>
        <tr>
          <td class="tg-baqh">""" + wordList[7] + """</td>
          <td class="tg-baqh">""" + scoreList[7] + """</td>
        </tr>
        <tr>
          <td class="tg-baqh">""" + wordList[8] + """</td>
          <td class="tg-baqh">""" + scoreList[8] + """</td>
        </tr>
        <tr>
          <td class="tg-baqh">""" + wordList[9] + """</td>
          <td class="tg-baqh">""" + scoreList[9] + """</td>
        </tr>

      </table>

  </div>

  <!-- ################################################################################################################# -->

    <script src="http://d3js.org/d3.v3.min.js"></script>
    <script src="https://rawgit.com/jasondavies/d3-cloud/master/build/d3.layout.cloud.js"></script>
    <script>
    //Simple animated example of d3-cloud - https://github.com/jasondavies/d3-cloud
    //Based on https://github.com/jasondavies/d3-cloud/blob/master/examples/simple.html
    // Encapsulate the word cloud functionality
    function wordCloud(selector) {
        var fill = d3.scale.category20();
        //Construct the word cloud's SVG element
        var svg = d3.select(selector).append("svg")
            .attr("width", 500)
            .attr("height", 500)
            .append("g")
            .attr("transform", "translate(250,250)");
        //Draw the word cloud
        function draw(words) {
            var cloud = svg.selectAll("g text")
                            .data(words, function(d) { return d.text; })
            //Entering words
            cloud.enter()
                .append("text")
                .style("font-family", "Impact")
                .style("fill", function(d, i) { return fill(i); })
                .attr("text-anchor", "middle")
                .attr('font-size', 1)
                .text(function(d) { return d.text; });
            //Entering and existing words
            cloud
                .transition()
                    .duration(600)
                    .style("font-size", function(d) { return d.size + "px"; })
                    .attr("transform", function(d) {
                        return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
                    })
                    .style("fill-opacity", 1);
            //Exiting words
            cloud.exit()
                .transition()
                    .duration(200)
                    .style('fill-opacity', 1e-6)
                    .attr('font-size', 1)
                    .remove();
        }
        //Use the module pattern to encapsulate the visualisation code. We'll
        // expose only the parts that need to be public.
        return {
            //Recompute the word cloud for a new set of words. This method will
            // asycnhronously call draw when the layout has been computed.
            //The outside world will need to call this function, so make it part
            // of the wordCloud return value.
            update: function(words) {
                d3.layout.cloud().size([500, 500])
                    .words(words)
                    .padding(5)
                    .rotate(function() { return ~~(Math.random() * 2) * 90; })
                    .font("Impact")
                    .fontSize(function(d) { return d.size; })
                    .on("end", draw)
                    .start();
            }
        }
        }
        //Some sample data - http://en.wikiquote.org/wiki/Opening_lines
        var words = [ """ + "\"" + wordListStr + "\"" + """]
        //Prepare one of the sample sentences by removing punctuation,
        // creating an array of words and computing a random size attribute.
        function getWords(i) {
            return words[i]
                    .replace(/[!\.,:;\?]/g, '')
                    .split(' ')
                    .map(function(d) {
                        return {text: d, size: 10 + Math.random() * 60};
                    })
        }
        //This method tells the word cloud to redraw with a new set of words.
        //In reality the new words would probably come from a server request,
        // user input or some other source.
        function showNewWords(vis, i) {
            i = i || 0;
            vis.update(getWords(i ++ % words.length))
            setTimeout(function() { showNewWords(vis, i + 1)}, 2000)
        }
        //Create a new instance of the word cloud visualisation.
        var myWordCloud = wordCloud('body');
        //Start cycling through the demo data
        showNewWords(myWordCloud);
        </script>

</body>

"""



f.write(message.encode("utf-8"))
f.close()

filePath = os.path.dirname(os.path.abspath(__file__));

# print message
webbrowser.open_new_tab(filePath +'\\Top5keywords.html')