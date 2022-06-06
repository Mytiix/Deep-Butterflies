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

__author__ = "Marganne Louis <louis.marganne@student.uliege.be>"


from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from argparse import ArgumentParser

from utils import *

import pickle
import glob
import sys


def pred_lm(model, test_images, test_lmks, org_images, org_lmks, N, batch_size):
	test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_lmks))
	test_ds = test_ds.map(lambda image, lm: parse_data(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	test_ds = test_ds.map(lambda image, lm: normalize(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	test_ds = test_ds.batch(batch_size)

	preds = model.predict(test_ds)

	org_landmarks = []
	pred_landmarks = []
	for i in range(len(test_images)):
		org_img = cv.imread(org_images[i])
		lm_ids = np.loadtxt(org_lmks[i])[:,0:1]
		org_lm = np.loadtxt(org_lmks[i])[:,1:3]
		pred_mask = preds[i]
		pred_mask = np.reshape(pred_mask, newshape=(256,256, N))
		lm_list = []

		for j in range(N):
			x,y =  maskToKeypoints(pred_mask[:, :, j])
			lm_list.append((x,y))
		pred_lm = np.array(lm_list)
		up_size = org_img.shape[:2]
		up_lmks = up_lm(pred_lm, 256, up_size)
		org_landmarks.append(org_lm)
		pred_landmarks.append(up_lmks)

	return pred_landmarks, org_landmarks, lm_ids
	

if __name__ == '__main__':
	## Parameters
	# Arguments
	parser = ArgumentParser()
	parser.add_argument('--side', dest='side', required=True, help="v or d (ventral or dorsal)")
	parser.add_argument('--species', dest='species', required=True, help="The name of the folder containing the data")
	parser.add_argument('--lm_type', dest='lm_type', required=True, help="Type of landmarks (lm, slm, all)")
	parser.add_argument('--prob_fct', dest='prob_fct', required=True, help="The probability function used to describe the heatmap dispersion (exp, gaussian)")
	parser.add_argument('--batch_size', dest='batch_size', required=True, type=int, help="Size of batchs")
	parser.add_argument('--sigma', dest='sigma', required=True, type=int, help="Value of sigma (in probability function)")
	params, _ = parser.parse_known_args(sys.argv[1:])

	# Extra params
	if params.lm_type == 'lm':
		N = 14 if params.side == 'v' else 18
	elif params.lm_type == 'slm':
		N = 15 if params.side == 'v' else 26
	else:
		N = 29 if params.side == 'v' else 44

	# Load model
	model = load_model('./lm_scripts/saved_models/unet/unet1_'+params.species+'_'+params.side+'_sigma'+str(params.sigma)+'_'+params.prob_fct+'.hdf5')

	# Path to original test set
	test_repo = 'D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/'
	org_images = glob.glob(test_repo+'images/*.tif')

	# Build dictionary which maps image id to their specie
	with open('sp.pkl', 'rb') as file:
		sp = pickle.load(file)

	id_to_sp = {}
	for specie in sp.keys():
		idx = 0 if params.side == 'd' else 1
		for image in sp[specie][idx]:
			id_to_sp[image.id] = specie

	# Build dictionary which maps a specie to the corresponding test images
	# Dict => {specie : [test_images, test_lmks, org_images, org_lmks]}
	sp_to_testim = defaultdict(lambda: [list(), list(), list(), list()])
	for image_path in org_images:
		# Get specie
		im_id = os.path.basename(image_path)[:-4]
		im_sp = id_to_sp[int(im_id)]

		# Build dict
		sp_to_testim[im_sp][0].append(test_repo+'rescaled_v2/images/'+im_id+'.png')
		sp_to_testim[im_sp][1].append(test_repo+'rescaled_v2/landmarks_v2/'+im_id+'.txt')
		sp_to_testim[im_sp][2].append(image_path)
		sp_to_testim[im_sp][3].append(test_repo+'landmarks_v2/'+im_id+'.txt')


	## Compute prediction per specie and evalutate
	weighted_mean_rmse = []
	weighted_mean_ht = []
	for k, v in sp_to_testim.items():
		print(k+f' => nb_samples = {len(v[0])} :\n')
		if len(v[0]) == 0:
			continue

		pred_landmarks, org_landmarks, lm_ids = pred_lm(model, v[0], v[1], v[2], v[3], N, params.batch_size)

		# Compute RMSE/Hit Rate
		threshold = 30
		rmse = np.zeros(N)
		ht = np.zeros(N)
		for i in range(len(org_landmarks)):
			for j in range(N):
				g_lm = org_landmarks[i][j]
				p_lm = pred_landmarks[i][j]
				
				rmse[j] += mean_squared_error(p_lm, g_lm, squared=False)
				dist = np.linalg.norm(g_lm - p_lm)
				ht[j] += 1 if dist <= threshold else 0

		rmse /= len(org_landmarks)
		ht /= len(org_landmarks)
		ht *= 100

		# Print evaluation per landmark
		with open('terms_names.pkl', 'rb') as file:
			terms_names = pickle.load(file)

		rmse_lm = {terms_names[lm_ids[i,0]] : (rmse[i], ht[i]) for i in range(N)}
		print('LM name => rmse, ht')
		for key, value in sorted(rmse_lm.items()):
			print(key, '=>', '{:.2f}'.format(value[0]), ', {:.2f}%'.format(value[1]))

		# Print evaluation mean of every landmark
		mean_rmse = np.mean(rmse)
		mean_ht = np.mean(ht)
		print(f'\nMean RMSE => {mean_rmse}')
		print(f'Mean Hit Rate => {mean_ht}%\n')

		weighted_mean_rmse.append(mean_rmse * len(v[0]) / len(org_images))
		weighted_mean_ht.append(mean_ht * len(v[0]) / len(org_images))

	print(f'Mean evalution :')
	print(f'\nMean RMSE => {np.sum(weighted_mean_rmse)}')
	print(f'Mean Hit Rate => {np.sum(weighted_mean_ht)}%\n')