import logging
import random
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

from cytomine import Cytomine
from cytomine.models import AnnotationCollection, TermCollection

from shapely.geometry import Point

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
	parser.add_argument('--species', dest='species', required=True, help="The name(s) of the specie(s) (separated by ',')")
	params, _ = parser.parse_known_args(sys.argv[1:])

	with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key) as cytomine:
		
		# Get terms
		terms_collection = TermCollection().fetch_with_filter("project", params.project_id)
		terms_names = {term.id : term.name for term in terms_collection}
		t = [str(i) + '-v-lm' for i in range(1,15)] if params.side == 'v' else [str(i) + '-d-lm' for i in range(1,19)]

		# Fetch training images
		random.seed(0)
		with open('sp.pkl', 'rb') as file:
			sp = pickle.load(file)
		idx = 0 if params.side == 'd' else 1
		images_list = list()
		images_to_pred = list()
		if params.species == 'all':
			for k, v in sp.items():
				images_list += v[0+idx]

		else:
			for specie in params.species.split(','):
				images_list += sp[specie][0+idx]


		# Get test set
		nb_to_swap = len(images_list) // 5
		for i in range(nb_to_swap):
			images_to_pred.append(images_list.pop(random.randint(0, len(images_list)-i-1)))


		# Compute mean coords of training samples
		mean_coords = defaultdict(lambda: [0, 0])
		for image in images_list:
			annotations = AnnotationCollection()
			annotations.project = params.project_id
			annotations.image = image.id
			annotations.showTerm = True
			annotations.showWKT = True
			annotations.fetch()
			for annotation in annotations:
				if terms_names[annotation.term[0]] in t:
					point = wkt.loads(annotation.location)
					mean_coords[terms_names[annotation.term[0]]][0] += point.x / len(images_list)
					mean_coords[terms_names[annotation.term[0]]][1] += point.y / len(images_list)


		# Fetch grond truth annotations from image_to_pred
		annot = defaultdict(lambda: list())
		for image in images_to_pred:
			annotations = AnnotationCollection()
			annotations.project = params.project_id
			annotations.image = image.id
			annotations.showTerm = True
			annotations.showWKT = True
			annotations.fetch()
			for annotation in annotations:
				if terms_names[annotation.term[0]] in t:
					annot[terms_names[annotation.term[0]]].append(wkt.loads(annotation.location))


		# Evaluation
		threshold = 30
		mse = np.zeros(len(t))
		ht = np.zeros(len(t))
		for i, term in enumerate(t):
			mean_coord =  Point(mean_coords[term][0], mean_coords[term][1])
			for point in annot[term]:
				dist = mean_coord.distance(point)
				mse[i] += dist ** 2
				ht[i] += 1 if dist <= threshold else 0
		
		mse /= len(images_to_pred)
		rmse = np.sqrt(mse)
		ht /= len(images_to_pred)
		ht *= 100

		hit_rate = ['{:.2f}%'.format(v) for v in ht]
		table = reversed(list(zip(t, mse, rmse, hit_rate)))
		print('\n')
		print(tabulate(table, headers=['Term', 'mse', 'rmse', 'ht']))

		print('\n')
		print(f'Mean MSE => {np.mean(mse)}')
		print(f'Mean RMSE => {np.mean(rmse)}')
		print(f'Mean Hit Rate => {np.mean(ht)}%')
		