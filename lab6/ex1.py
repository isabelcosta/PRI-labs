import json
import xml.etree.ElementTree as ET
from lxml import etree

json_string = '{"first_name": "Bruno", "last_name":"Martns"}'
parsed_json = json.loads(json_string)
print(parsed_json['first_name'])


root = etree.parse("AirFlightsData.xml")
for elem in root.xpath('//date'):
    print elem, elem.text





