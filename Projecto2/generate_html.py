def generated_html_with_keyphrases(wordList, scoreList, wordListStr):
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

    return message