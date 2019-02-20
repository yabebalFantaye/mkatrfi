import sys, xml.etree.ElementTree as ET

data = ""
for line in sys.stdin:
    data += line

tree = ET.fromstring(data)

nodeA = tree.find('.//tag')
nodeB = tree.find('.//tag1')

tmp = nodeA.text
nodeA.text = nodeB.text
nodeB.text = tmp 

print ET.tostring(tree)
