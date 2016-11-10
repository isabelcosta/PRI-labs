import json
json_string = '{"first_name": "Bruno", "last_name":"Martns"}'
parsed_json = json.loads(json_string)
print(parsed_json['first_name'])

from lxml import etree
root = etree.parse("AirFlightsData.xml")

# for elem in root.xpath(".//Passenger/name"):
#     print elem, elem.text
#
# for elem in root.xpath(".//date"):
#     print elem, elem.text
#
# for elem in root.xpath(".//Flight[date='2008-12-24']"):
#     print elem, elem.text

for elem in root.xpath(".//Airport[name='Zurich']/following-sibling::Airport"):
    print elem, elem.text

# for elem in root.xpath("/doc/Flight/*[position()<3]"):
#     print elem, elem.text

# doc = etree.parse("AirFlightsData.xml")
# for res in doc.xpathEval("//*"):
#     print res.name
#     print res.content
# doc.freeDoc()

# import simplexquery as sxq
# res = sxq.execute("doc('AirFlightsData.xml')//*")
# print res
