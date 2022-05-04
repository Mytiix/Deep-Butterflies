import logging
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
	parser.add_argument('--job_id', dest='job_id', required=True, help="The job from which the images were created")
	params, _ = parser.parse_known_args(sys.argv[1:])

	with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key) as cytomine:
		
		# Store annotations as such: {term_id : {image_id : [pred, ground_truth], ...}, ...}
		annot = defaultdict(lambda: dict())

		# Get terms
		terms_collection = TermCollection().fetch_with_filter("project", params.project_id)
		terms_names = {term.id : term.name for term in terms_collection}
		t = [str(i) + '-v-lm' for i in range(1,15)] + [str(i) + '-d-lm' for i in range(1,19)]
		#t = [str(i) + '-v-slm' for i in range(15,30)] + [str(i) + '-d-slm' for i in range(19,45)]

		# Fetch predicted annotations
		annotations = AnnotationCollection()
		annotations.project = params.project_id
		annotations.job = params.job_id
		annotations.showTerm = True
		annotations.showWKT = True
		annotations.fetch()
		for annotation in annotations:
			annot[annotation.term[0]][annotation.image] = [wkt.loads(annotation.location)]

		# Get list of images
		for v in annot.values():
			images_list = list(v.keys())
			nb_im = len(images_list)
			break

		# Fetch ground truth annotations
		for image in images_list:
			annotations = AnnotationCollection()
			annotations.project = params.project_id
			annotations.image = image
			annotations.showTerm = True
			annotations.showWKT = True
			annotations.fetch()
			for annotation in annotations:
				if terms_names[annotation.term[0]] in t:
					annot[annotation.term[0]][image].append(wkt.loads(annotation.location))


		# Evaluation
		threshold = 30
		mse = np.zeros(len(annot))
		ht = np.zeros(len(annot))
		for i, (term, images) in enumerate(annot.items()):
			for v in images.values():
				dist = v[0].distance(v[1])
				mse[i] += dist ** 2
				ht[i] += 1 if dist <= threshold else 0
		
		rmse = np.sqrt(mse)
		rmse /= nb_im
		mse /= nb_im
		ht /= nb_im
		ht *= 100

		hit_rate = ['{:.2f}%'.format(v) for v in ht]
		names = [terms_names[k] for k in annot.keys()]
		table = reversed(list(zip(names, mse, rmse, hit_rate)))
		print('\n')
		print(tabulate(table, headers=['Term', 'mse', 'rmse', 'ht']))

		print('\n')
		print(f'Mean MSE => {np.mean(mse)}')
		print(f'Mean RMSE => {np.mean(rmse)}')
		print(f'Mean Hit Rate => {np.mean(ht)}%')

		# Hit Rate graph mean_ht
		mean_ht = np.zeros(100)
		for i in range(100):
			ht = np.zeros(len(annot))
			for j, (term, images) in enumerate(annot.items()):
				for v in images.values():
					dist = v[0].distance(v[1])
					ht[j] += 1 if dist <= i else 0
			ht /= nb_im
			ht *= 100
			mean_ht[i] = np.mean(ht)

		print(f'\nMax Hit Rate value : {max(mean_ht)}')
		print(f'Max value reached at threshold : {np.argmax(mean_ht)}')
		print(mean_ht)

		plt.figure()
		plt.title('Hit Rate percentage with respect to the pixel threshold')
		plt.xlabel('Threshold value')
		plt.ylabel('Hit Rate')
		plt.plot(mean_ht)
		plt.savefig('mean_ht.pdf')
		plt.show()

		
		# Hit Rate graph per LM
		ht_lm = np.zeros((100,len(annot)))
		for i in range(100):
			for j, (term, images) in enumerate(annot.items()):
				for v in images.values():
					dist = v[0].distance(v[1])
					ht_lm[i,j] += 1 if dist <= i else 0
		ht_lm /= nb_im
		ht_lm *= 100

		#print(f'\nMax Hit Rate value : {max(ht_lm)}')
		#print(f'Max value reached at threshold : {np.argmax(ht_lm)}')
		#print(ht_lm)

		plt.figure()
		plt.title('Hit Rate percentage with respect to the pixel threshold')
		plt.xlabel('Threshold value')
		plt.ylabel('Hit Rate')
		for i in range(len(annot)):
			plt.plot(ht_lm[:,i])
		plt.savefig('ht_lm.pdf')
		plt.show()
		