import json
from lxml import etree

json_string = '{"first_name": "Bruno", "last_name":"Martins"}'
parsed_json = json.loads(json_string)
print(parsed_json['first_name'])


root = etree.parse("AirFlightsData.xml")
for elem in root.xpath('//Flight[source = "LHR"]'):
    print elem, elem.text





