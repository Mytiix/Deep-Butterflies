import logging
import sys
import cv2

import numpy as np
import matplotlib.pyplot as plt

from cytomine import Cytomine
from cytomine.models import AnnotationCollection, TermCollection

from collections import defaultdict
from argparse import ArgumentParser
from shapely import wkt
from tabulate import tabulate

logging.basicConfig()
logger = logging.getLogger("cytomine.client")
logger.setLevel(logging.INFO)

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--host', dest='host', help="The Cytomine host")
	parser.add_argument('--public_key', dest='public_key', help="The Cytomine public key")
	parser.add_argument('--private_key', dest='private_key', help="The Cytomine private key")
	parser.add_argument('--project_id', dest='project_id', required=True, help="The project from which we want the images")
	parser.add_argument('--job_id', dest='job_id', required=True, help="The job from which the images were created")
	parser.add_argument('--image_id', dest='image_id', required=True, help="The image to plot")
	params, _ = parser.parse_known_args(sys.argv[1:])

	with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key) as cytomine:

		# Get terms
		terms_collection = TermCollection().fetch_with_filter("project", params.project_id)
		terms_names = {term.id : term.name for term in terms_collection}
		t = [str(i) + '-v-lm' for i in range(1,15)] + [str(i) + '-d-lm' for i in range(1,19)]

		# Read image
		image_path = 'tif/PB13.0906-d.tif'
		image = cv2.imread(image_path)
		height = image.shape[0]
		
		# Write ground truth annotations
		annotations = AnnotationCollection()
		annotations.project = params.project_id
		annotations.image = params.image_id
		annotations.showTerm = True
		annotations.showWKT = True
		annotations.fetch()
		for annotation in annotations:
			if terms_names[annotation.term[0]] in t:
				point = wkt.loads(annotation.location)
				image = cv2.circle(image, (int(point.x), height-int(point.y)), radius=15, color=(0,0,255), thickness=-1)
				#image = cv2.circle(image, (int(point.x), height-int(point.y)), radius=30, color=(0,255,0), thickness=3)				


		# Write predicted annotations
		annotations = AnnotationCollection()
		annotations.project = params.project_id
		annotations.image = params.image_id
		annotations.job = params.job_id
		annotations.showTerm = True
		annotations.showWKT = True
		annotations.fetch()
		for annotation in annotations:
			point = wkt.loads(annotation.location)
			image = cv2.circle(image, (int(point.x), height-int(point.y)), radius=15, color=(255,0,0), thickness=-1)	

		cv2.imwrite('im.png', image)