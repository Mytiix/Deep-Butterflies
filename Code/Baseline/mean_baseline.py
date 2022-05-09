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
	parser.add_argument('--side', dest='side', required=True, help="v or d (ventral or dorsal)")
	parser.add_argument('--species', dest='species', required=True, help="The name(s) of the specie(s) (separated by ',')")
	params, _ = parser.parse_known_args(sys.argv[1:])

	# Extra params
	savefig = True
	extra = True

	# Define repositories
	train_lmks = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/training/landmarks/*.txt')
	org_images = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/images/*.tif')
	org_lmks = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/landmarks/*.txt')

	
	# Compute mean coords of training samples
	mean_coords = defaultdict(lambda: [0, 0])
	for file in train_lmks:
		for i, gt_lm in enumerate(np.loadtxt(file)):
			mean_coords[i][0] += gt_lm[0] / len(train_lmks)
			mean_coords[i][1] += gt_lm[1] / len(train_lmks)

	# Fetch ground truth annotations
	org_landmarks = []
	pred_landmarks = []
	for file in org_lmks:
		org_lm = []
		pred_lm = []
		for i, gt_lm in enumerate(np.loadtxt(file)):
			org_lm.append([gt_lm[0], gt_lm[1]])
			pred_lm.append([mean_coords[i][0], mean_coords[i][1]])
		org_landmarks.append(np.array(org_lm))
		pred_landmarks.append(np.array(pred_lm))

	# Evaluation
	threshold = 30
	rmse = np.zeros(len(mean_coords))
	ht = np.zeros(len(mean_coords))
	for i in range(len(org_landmarks)):
		for j in range(len(mean_coords)):
			g_lm = org_landmarks[i][j]
			p_lm = pred_landmarks[i][j]
			
			rmse[j] += mean_squared_error(p_lm, g_lm, squared=False)
			dist = np.linalg.norm(g_lm - p_lm)
			ht[j] += 1 if dist <= threshold else 0
		
	rmse /= len(org_landmarks)
	ht /= len(org_landmarks)
	ht *= 100

	print('\n')
	print(f'Mean RMSE => {np.mean(rmse)}')
	print(f'Mean Hit Rate => {np.mean(ht)}%')
	
	# Plot/Save results
	filename = params.species+'_'+params.side
	if savefig:
		org_images = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/images/*.tif')

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
			ht = np.zeros(len(mean_coords))
			for j in range(len(org_landmarks)):
				for k in range(len(mean_coords)):
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
		plt.ylabel('Hit Rate (%)')
		plt.plot(mean_ht)
		plt.savefig('figures/baseline_mean_ht_'+filename+'.pdf')
		plt.show()


		# Hit Rate graph per LM
		ht_lm = np.zeros((100,len(mean_coords)))
		for i in range(100):
			for j in range(len(org_landmarks)):
				for k in range(len(mean_coords)):
					g_lm = org_landmarks[j][k]
					p_lm = pred_landmarks[j][k]
					
					dist = np.linalg.norm(g_lm - p_lm)
					ht_lm[i,k] += 1 if dist <= i else 0
		ht_lm /= len(org_landmarks)
		ht_lm *= 100

		plt.figure()
		plt.title('Hit Rate percentage with respect to the pixel threshold')
		plt.xlabel('Threshold value')
		plt.ylabel('Hit Rate (%)')
		for i in range(len(mean_coords)):
			plt.plot(ht_lm[:,i])
		plt.savefig('figures/baseline_ht_lm_'+filename+'.pdf')
		plt.show()