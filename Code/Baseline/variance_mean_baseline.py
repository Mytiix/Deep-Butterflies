import logging
import pickle
import sys

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
	parser.add_argument('--species', dest='species', required=True, help="The name(s) of the specie(s) (separated by ',')")
	params, _ = parser.parse_known_args(sys.argv[1:])

	with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key) as cytomine:

		# Get terms
		terms_collection = TermCollection().fetch_with_filter("project", params.project_id)
		terms_names = {term.id : term.name for term in terms_collection}
		terms_ids = {term.name : term.id for term in terms_collection}
		t = [str(i) + '-v-lm' for i in range(1,15)] + [str(i) + '-d-lm' for i in range(1,19)]

		# Get list of images from the specified specie(s)
		with open('sp.pkl', 'rb') as file:
			sp = pickle.load(file)
		idx = 0 if params.side == 'd' else 1
		images_list = list()
		if params.species == 'all':
			for k in sp.keys():
				images_list += sp[k][0+idx]

		else:
			for specie in params.species.split(','):
				images_list += sp[specie][0+idx]

		# Get cooords from every images
		coords = defaultdict(lambda: [[], []])
		for image in images_list:
			annotations = AnnotationCollection()
			annotations.project = params.project_id
			annotations.image = image.id
			annotations.showTerm = True
			annotations.showWKT = True
			annotations.fetch()
			for annotation in annotations:
				term_name = terms_names[annotation.term[0]]
				if term_name in t:
					point = wkt.loads(annotation.location)
					coords[term_name][0].append(point.x)
					coords[term_name][1].append(point.y)

		# Compute variance/std
		var_x = {k : np.var(v[0]) for k,v in coords.items()}
		var_y = {k : np.var(v[1]) for k,v in coords.items()}
		std_x = {k : np.std(v[0]) for k,v in coords.items()}
		std_y = {k : np.std(v[1]) for k,v in coords.items()}
		
		table = reversed(list(zip(coords.keys(), var_x.values(), var_y.values(), std_x.values(), std_y.values())))
		print('\n')
		print(tabulate(table, headers=['Term', 'var_x', 'var_y', 'std_x', 'std_y']))

		mean_var_x = np.mean(list(var_x.values()))
		mean_var_y = np.mean(list(var_y.values()))
		mean_std_x = np.mean(list(std_x.values()))
		mean_std_y = np.mean(list(std_y.values()))

		print('\n')
		print(f'Mean var_x : {mean_var_x}')
		print(f'Mean var_y : {mean_var_y}')
		print(f'Mean std_x : {mean_std_x}')
		print(f'Mean std_y : {mean_std_y}')

		mean_var = (mean_var_x + mean_var_y) / 2
		mean_std = (mean_std_x + mean_std_y) / 2

		print('\n')
		print(f'Mean var : {mean_var}')
		print(f'Mean std : {mean_std}')
