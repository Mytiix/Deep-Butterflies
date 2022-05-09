import logging
import random
import pickle
import glob
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from cytomine.models import AnnotationCollection, TermCollection
from cytomine import Cytomine

from sklearn.metrics import mean_squared_error
from shapely.geometry import Point

from collections import defaultdict
from argparse import ArgumentParser
from tabulate import tabulate
from shapely import wkt

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
		
		# Extra params
		savefig = True
		extra = True

		# Get terms
		terms_collection = TermCollection().fetch_with_filter("project", params.project_id)
		terms_names = {term.id : term.name for term in terms_collection}
		t = [str(i) + '-v-lm' for i in range(1,15)] if params.side == 'v' else [str(i) + '-d-lm' for i in range(1,19)]
		#t = [str(i) + '-v-slm' for i in range(15,30)] if params.side == 'v' else [str(i) + '-d-slm' for i in range(19,45)]

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
		nb_to_swap = len(images_list) // 4
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
		rmse = np.zeros(len(t))
		ht = np.zeros(len(t))
		for i, term in enumerate(t):
			mean_coord =  np.array([mean_coords[term][0], mean_coords[term][1]])
			for point in annot[term]:
				gt_lm = np.array([point.x, point.y])
				rmse[i] += mean_squared_error(mean_coord, gt_lm, squared=False)
				dist = np.linalg.norm(mean_coord - gt_lm)
				ht[i] += 1 if dist <= threshold else 0
			
		rmse /= len(images_to_pred)
		ht /= len(images_to_pred)
		ht *= 100

		hit_rate = ['{:.2f}%'.format(v) for v in ht]
		table = reversed(list(zip(t, rmse, hit_rate)))
		print('\n')
		print(tabulate(table, headers=['Term', 'rmse', 'ht']))

		print('\n')
		print(f'Mean RMSE => {np.mean(rmse)}')
		print(f'Mean Hit Rate => {np.mean(ht)}%')
		
		# Plot/Save results
		if savefig:
			org_landmarks = []
			pred_landmarks = []
			for i, image in enumerate(images_to_pred):
				org_lm = []
				pred_lm = []
				for k, v in annot.items():
					point = annot[k][i]
					org_lm.append([point.x, point.y])
					pred_lm.append([mean_coords[k][0], mean_coords[k][1]])
				org_landmarks.append(np.array(org_lm))
				pred_landmarks.append(np.array(pred_lm))


			org_images = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/images/*.tif')
			filename = params.species+'_'+params.side

			if not os.path.exists('images/'+filename):
					os.makedirs('images/'+filename)

			for i in range(len(org_images)):
				img = cv.imread(org_images[i])
				img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
				gt_lm = org_landmarks[i]
				pred_lm = pred_landmarks[i]

				plt.figure()
				plt.title('Predicted LM (red) / gt LM (blue)')
				plt.imshow(img)
				plt.axis('off')
				plt.scatter(gt_lm[:,0], gt_lm[:,1], c='b')
				plt.scatter(pred_lm[:,0], pred_lm[:,1], c='r')
				plt.savefig('images/'+filename+'/'+str(i))
				plt.close()


		if extra:
			# Hit Rate graph mean_ht
			mean_ht = np.zeros(100)
			for i in range(100):
				ht = np.zeros(len(annot))
				for j in range(len(org_landmarks)):
					for k in range(len(annot)):
						g_lm = org_landmarks[j][k]
						p_lm = pred_landmarks[j][k]
						
						dist = np.linalg.norm(g_lm - p_lm)
						ht[k] += 1 if dist <= i else 0
				ht /= len(org_landmarks)
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
			plt.savefig('figures/mean_ht_'+filename+'.pdf')
			plt.show()


			# Hit Rate graph per LM
			ht_lm = np.zeros((100,len(annot)))
			for i in range(100):
				for j in range(len(org_landmarks)):
					for k in range(len(annot)):
						g_lm = org_landmarks[j][k]
						p_lm = pred_landmarks[j][k]
						
						dist = np.linalg.norm(g_lm - p_lm)
						ht_lm[i,k] += 1 if dist <= i else 0
			ht_lm /= len(org_landmarks)
			ht_lm *= 100

			plt.figure()
			plt.title('Hit Rate percentage with respect to the pixel threshold')
			plt.xlabel('Threshold value')
			plt.ylabel('Hit Rate')
			for i in range(len(annot)):
				plt.plot(ht_lm[:,i])
			plt.savefig('figures/ht_lm_'+filename+'.pdf')
			plt.show()