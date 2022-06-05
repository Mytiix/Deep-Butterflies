host = "https://research.cytomine.be/"
public_key = "50af36d2-3ccc-4928-a21d-aab32e437fbf"
private_key = "49725276-a001-4d25-9800-76409c8aad9a"
from cytomine import Cytomine
from cytomine.utilities.descriptor_reader import read_descriptor
with Cytomine(host, public_key, private_key) as c:
	read_descriptor("descriptor.json")