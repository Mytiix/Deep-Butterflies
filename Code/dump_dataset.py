import logging
import pickle
import sys
import os

from cytomine import Cytomine
from cytomine.models import AnnotationCollection, TermCollection

from argparse import ArgumentParser
from shapely import wkt

logging.basicConfig()
logger = logging.getLogger("cytomine.client")
logger.setLevel(logging.INFO)


def dump_images(images_list, repository):
	xpos = {}
	ypos = {}
	terms = {}

	terms_collection = TermCollection().fetch_with_filter("project", params.project_id)
	terms_names = {term.id : term.name for term in terms_collection}
	#check_terms = [str(i) + '-v-lm' for i in range(1,15)] + [str(i) + '-d-lm' for i in range(1,19)]
	check_terms = [str(i) + '-v-slm' for i in range(15,30)] + [str(i) + '-d-slm' for i in range(19,45)]

	for image in images_list:
		#image.download(dest_pattern=repository+'/images/%d.tif' % image.id)

		annotations = AnnotationCollection()
		annotations.project = params.project_id
		annotations.showWKT = True
		annotations.showTerm = True
		annotations.image = image.id
		annotations.fetch()

		for annotation in annotations:
			if terms_names[annotation.term[0]] in check_terms:
				point = wkt.loads(annotation.location)
				(cx, cy) = point.xy
				xpos[(annotation.term[0], image.id)] = int(cx[0])
				ypos[(annotation.term[0], image.id)] = image.height - int(cy[0])
				terms[annotation.term[0]] = 1

	if not os.path.exists(repository+'/landmarks_v2/'):
		os.makedirs(repository+'/landmarks_v2/')

	for image in images_list:
		file = open(repository+'/landmarks_v2/%d.txt' % image.id, 'w')
		for t in terms.keys():
			if (t, image.id) in xpos:
				file.write('%d %d %d %f %f\n' % (
					t,
					xpos[(t, image.id)], 
					ypos[(t, image.id)],
					xpos[(t, image.id)] / float(image.width),
					ypos[(t, image.id)] / float(image.height)))
		file.close()



if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--host', dest='host', required=True, help="The Cytomine host")
	parser.add_argument('--public_key', dest='public_key', required=True, help="The Cytomine public key")
	parser.add_argument('--private_key', dest='private_key', required=True, help="The Cytomine private key")
	parser.add_argument('--project_id', dest='project_id', required=True, help="The project from which we want the images")
	params, _ = parser.parse_known_args(sys.argv[1:])

	with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key) as cytomine:

		repository = 'D:/Dataset_TFE/images_v2/'
	
		# Get every images to dump in the repository
		with open('sp.pkl', 'rb') as file:
			sp = pickle.load(file)
			
		images_list_d = list()
		images_list_v = list()
		for k, v in sp.items():
			images_list_d += v[0]
			images_list_v += v[1]

		specie = 'all_slm'
		
		dump_images(images_list_d, repository+specie+'/d')
		dump_images(images_list_v, repository+specie+'/v')