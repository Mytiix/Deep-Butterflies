import logging
import random
import pickle
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
	parser.add_argument('--host', dest='host', required=True, help="The Cytomine host")
	parser.add_argument('--public_key', dest='public_key', required=True, help="The Cytomine public key")
	parser.add_argument('--private_key', dest='private_key', required=True, help="The Cytomine private key")
	parser.add_argument('--project_id', dest='project_id', required=True, help="The project from which we want the images")
	parser.add_argument('--side', dest='side', required=True, help="v or d (ventral or dorsal)")
	parser.add_argument('--term', dest='term', required=True, help="The number of the term to plot")
	parser.add_argument('--species', dest='species', required=True, help="The name(s) of the specie(s) (separated by ',')")
	parser.add_argument('--npred', dest='npred', default=30000, help="Number of pixels sampled during prediction", type=int)
	params, _ = parser.parse_known_args(sys.argv[1:])

	with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key) as cytomine:
		random.seed(0)

		# Get terms
		terms_collection = TermCollection().fetch_with_filter("project", params.project_id)
		terms_names = {term.id : term.name for term in terms_collection}
		terms_ids = {term.name : term.id for term in terms_collection}
		t = [str(i) + '-v-lm' for i in range(1,15)] + [str(i) + '-d-lm' for i in range(1,19)]

		# Read image
		image_path = 'tif/PB13.0121-d.tif'
		image_id = 536138331
		image = cv2.imread(image_path)
		width = image.shape[1]
		height = image.shape[0]

		# Get list of images from the specified specie(s)
		with open('sp.pkl', 'rb') as file:
			sp = pickle.load(file)
		idx = 0 if params.side == 'd' else 1
		images_list = list()
		if params.species == 'all':
			for k, v in sp.items():
				images_list += v[0+idx]

		else:
			for specie in params.species.split(','):
				images_list += sp[specie][0+idx]

		# Discard image for the "test set"
		nb_to_swap = len(images_list) // 5
		for i in range(nb_to_swap):
			images_list.pop(random.randint(0, len(images_list)-i))


		# Compute value for Npred
		xr = []
		yr = []
		for im in images_list:
			annotations = AnnotationCollection()
			annotations.project = params.project_id
			annotations.image = im.id
			annotations.term = terms_ids[params.term+'-'+params.side+'-lm']
			annotations.showWKT = True
			annotations.fetch()
			assert len(annotations) == 1
			for annotation in annotations:
				if terms_names[annotation.term[0]] in t:
					point = wkt.loads(annotation.location)
					xr.append(point.x / float(im.width))
					yr.append((float(im.height)-point.y) / float(im.height))

		nimages = len(images_list)
		mx = np.mean(xr)
		my = np.mean(yr)
		P = np.zeros((2, nimages))
		P[0, :] = xr
		P[1, :] = yr
		cm = np.cov(P)

		# Write position of the corresponding 30000 examples extracted during prediction
		P = np.random.multivariate_normal([mx, my], cm, params.npred)
		x_v = np.round(P[:, 0] * width)
		y_v = np.round(P[:, 1] * height)
		
		for coord in zip(x_v, y_v):
			image = cv2.circle(image, (int(coord[0]), int(coord[1])), radius=1, color=(255,0,0), thickness=-1)

		# Write annotations of the specified specie(s) and compute mean_coords
		mean_x = []
		mean_y = []
		for im in images_list:
			annotations = AnnotationCollection()
			annotations.project = params.project_id
			annotations.image = im.id
			annotations.term = terms_ids[params.term+'-'+params.side+'-lm']
			annotations.showWKT = True
			annotations.fetch()
			assert len(annotations) == 1
			for annotation in annotations:
				if terms_names[annotation.term[0]] in t:
					point = wkt.loads(annotation.location)
					image = cv2.circle(image, (int(point.x), height-int(point.y)), radius=5, color=(0,0,255), thickness=-1)
					mean_x.append(point.x)
					mean_y.append(point.y)

		# Write annotation of the mean landmark
		image = cv2.circle(image, (int(sum(mean_x)/len(mean_x)), height-int(sum(mean_y)/len(mean_y))), radius=5, color=(0,255,255), thickness=-1)
		#image = cv2.circle(image, (int(sum(mean_x)/len(mean_x)), height-int(sum(mean_y)/len(mean_y))), radius=30, color=(0,255,0), thickness=3)				

		# Write annotation of the "true" image
		annotations = AnnotationCollection()
		annotations.project = params.project_id
		annotations.image = image_id
		annotations.term = terms_ids[params.term+'-'+params.side+'-lm']
		annotations.showWKT = True
		annotations.fetch()
		assert len(annotations) == 1
		for annotation in annotations:
			point = wkt.loads(annotation.location)
			image = cv2.circle(image, (int(point.x), height-int(point.y)), radius=5, color=(0,255,0), thickness=-1)

		cv2.imwrite(f'mean_baseline_{params.side}{params.term}.png', image)
		