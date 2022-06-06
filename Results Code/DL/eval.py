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

__author__ = "Navdeep Kumar <nkumar@uliege.be>"
__contributors__ = ["Marganne Louis <louis.marganne@student.uliege.be>"]


from tensorflow.keras.models import load_model

from shapely.geometry import Point
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser

from utils import *

import pickle
import joblib
import glob
import sys


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
	savefig = False
	extra = False

	if params.lm_type == 'lm':
		N = 14 if params.side == 'v' else 18
	elif params.lm_type == 'slm':
		N = 15 if params.side == 'v' else 26
	else:
		N = 29 if params.side == 'v' else 44

	# Load model
	model = load_model('./lm_scripts/saved_models/unet/unet1_'+params.species+'_'+params.side+'_sigma'+str(params.sigma)+'_'+params.prob_fct+'.hdf5')

	# Path to rescaled test set
	test_images = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/rescaled/images/*.png')

	# Path to original test set
	org_images = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/images/*.tif')
	org_lmks = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/landmarks_v2/*.txt')


	## Construct testing set with tensorflow
	test_ds = tf.data.Dataset.from_tensor_slices((test_images, None))
	test_ds = test_ds.map(lambda image, lm: parse_data(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	test_ds = test_ds.map(lambda image, lm: normalize(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	test_ds = test_ds.batch(params.batch_size)


	## Predict and fetch ground truth
	preds = model.predict(test_ds)

	org_landmarks = []
	pred_landmarks= []
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


	## Evaluation
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
	for k, v in sorted(rmse_lm.items()):
		print(k, '=>', '{:.2f}'.format(v[0]), ', {:.2f}%'.format(v[1]))

	# Print evaluation mean of every landmark
	print(f'\nMean RMSE => {np.mean(rmse)}')
	print(f'Mean Hit Rate => {np.mean(ht)}%')

	# Plot/Save results
	filename = params.species+'_'+params.side+'_sigma'+str(params.sigma)+'_'+params.prob_fct
	if savefig:
		if not os.path.exists('images/'):
			os.makedirs('images/')

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
			if savefig:
				plt.savefig('images/'+filename+'/'+str(i))
			plt.close()


	if extra:
		if not os.path.exists('figures/'):
			os.makedirs('figures/')
			
		# Hit Rate graph mean_ht
		mean_ht = np.zeros(100)
		for i in range(100):
			ht = np.zeros(N)
			for j in range(len(org_landmarks)):
				for k in range(N):
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
		plt.savefig('figures/dl_mean_ht_'+filename+'.pdf')
		plt.show()


		# Hit Rate graph per LM
		ht_lm = np.zeros((100,N))
		for i in range(100):
			for j in range(len(org_landmarks)):
				for k in range(N):
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
		for i in range(N):
			plt.plot(ht_lm[:,i])
		plt.savefig('figures/dl_ht_lm_'+filename+'.pdf')
		plt.show()