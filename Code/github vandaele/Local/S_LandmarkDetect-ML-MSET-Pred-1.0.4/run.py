# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2016. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__ = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__ = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__ = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"

from sklearn.metrics import mean_squared_error
from collections import defaultdict
from argparse import ArgumentParser
from tqdm import tqdm

import pickle
import joblib
import glob
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

sys.path.insert(0, '../')
from ldmtools import *

"""
Given the classifier clf, this function will try to find the landmark on the
image current
"""


def searchpoint_cytomine(repository, current, clf, mx, my, cm, depths, window_size, feature_type, feature_parameters,
						 image_type,
						 npred):

	simage = readimage(repository, current, image_type)
	(height, width) = simage.shape

	P = np.random.multivariate_normal([mx, my], cm, npred)
	x_v = np.round(P[:, 0] * width)
	y_v = np.round(P[:, 1] * height)

	height = height - 1
	width = width - 1

	n = len(x_v)
	pos = 0

	maxprob = -1
	maxx = []
	maxy = []

	# maximum number of points considered at once in order to not overload the
	# memory.
	step = 100000

	for index in range(len(x_v)):
		xv = x_v[index]
		yv = y_v[index]
		if (xv < 0):
			x_v[index] = 0
		if (yv < 0):
			y_v[index] = 0
		if (xv > width):
			x_v[index] = width
		if (yv > height):
			y_v[index] = height

	while (pos < n):
		xp = np.array(x_v[pos:min(n, pos + step)])
		yp = np.array(y_v[pos:min(n, pos + step)])

		DATASET = build_dataset_image(simage, window_size, xp, yp, feature_type, feature_parameters, depths)
		pred = clf.predict_proba(DATASET)
		pred = pred[:, 1]
		maxpred = np.max(pred)
		if (maxpred >= maxprob):
			positions = np.where(pred == maxpred)
			positions = positions[0]
			xsup = xp[positions]
			ysup = yp[positions]
			if (maxpred > maxprob):
				maxprob = maxpred
				maxx = xsup
				maxy = ysup
			else:
				maxx = np.concatenate((maxx, xsup))
				maxy = np.concatenate((maxy, ysup))
		pos = pos + step

	return np.median(maxx), np.median(maxy)


def main():
	## Parameters

	# Arguments
	parser = ArgumentParser()
	parser.add_argument('--side', dest='side', required=True, help="v or d (ventral or dorsal)")
	parser.add_argument('--species', dest='species', required=True, help="The name(s) of the specie(s)")
	parser.add_argument('--from_save', '-s', dest='from_save', action='store_true')
	params, _ = parser.parse_known_args(sys.argv[1:])

	# Extra params
	savefig = True
	extra = True

	# Define repositories
	repository = 'D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/'
	lm_repo = repository+'landmarks_v2/'
	im_repo = repository+'images/'

	if not os.path.exists(repository+'out/'):
		os.makedirs(repository+'out/')
	out_repo = repository+'out/'

	tr_repo = 'D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/training/'
	tr_out_repo = tr_repo+'out/'


	# Get lists of image/term ids to predict
	list_imgs, term_list = getallcoords(lm_repo)


	## Prediction

	# Store annotations as such: {term_id : {image_id : [pred, ground_truth], ...}, ...}
	annot = defaultdict(lambda: dict())
	
	if not params.from_save:
		save = []
		for id_term in tqdm(term_list):
			# Load models from training
			model_filepath = tr_out_repo+str(id_term)+'_model.joblib'
			cov_filepath = tr_out_repo+str(id_term)+'_cov.joblib'
			parameters_filepath = tr_out_repo+str(id_term)+'_parameters.joblib'

			model = joblib.load(model_filepath)
			[mx, my, cm] = joblib.load(cov_filepath)
			parameters_hash = joblib.load(parameters_filepath)

			feature_parameters = None
			if parameters_hash['feature_type'] in ['haar', 'gaussian']:
				fparameters_filepath = tr_out_repo+str(id_term)+'_fparameters.joblib'
				feature_parameters = joblib.load(fparametersl_filepath)

			# Get prediction for each images
			for id_img in list_imgs:
				(x, y) = searchpoint_cytomine(im_repo, id_img, model, mx, my, cm, 1. / (2. ** np.arange(parameters_hash['model_depth'])), parameters_hash['window_size'], parameters_hash['feature_type'], feature_parameters, 'tif', parameters_hash['model_npred'])
				annot[id_term][id_img] = [np.array([x, y])]
				save.append([id_term, id_img, x, y])



		# Save predictions
		file = open('pickle/'+params.species+'_'+params.side+'.pkl', 'wb')
		pickle.dump(save, file)

	if params.from_save:
		with open('pickle/'+params.species+'_'+params.side+'.pkl', 'rb') as file:
			save = pickle.load(file)
		for v in save:
			annot[v[0]][v[1]] = [np.array([v[2], v[3]])]
	
	# Get ground truth annotations
	files = glob.glob(lm_repo+'*.txt')
	org_landmarks = []
	pred_landmarks = []
	for file in files:
		org_lm = []
		pred_lm = []
		for gt_lm in np.loadtxt(file):
			annot[int(gt_lm[0])][int(os.path.basename(file).rstrip('.txt'))].append(np.array([gt_lm[1], gt_lm[2]]))
			org_lm.append([gt_lm[1], gt_lm[2]])
			pred_lm.append(list(annot[int(gt_lm[0])][int(os.path.basename(file).rstrip('.txt'))][0]))
		org_landmarks.append(np.array(org_lm))
		pred_landmarks.append(np.array(pred_lm))

	# Evaluation
	threshold = 30
	rmse = np.zeros(len(annot))
	ht = np.zeros(len(annot))
	for i, (term, images) in enumerate(annot.items()):
		for v in images.values():
			rmse[i] += mean_squared_error(v[0], v[1], squared=False)
			dist = np.linalg.norm(v[0] - v[1])
			ht[i] += 1 if dist <= threshold else 0

	rmse /= len(files)
	ht /= len(files)
	ht *= 100

	print(f'Mean RMSE => {np.mean(rmse)}')
	print(f'Mean Hit Rate => {np.mean(ht)}%')


	# Plot/Save results
	if savefig:
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

if __name__ == "__main__":
	main()