from cytomine.models.image import ImageInstanceCollection
from cytomine.models import TermCollection
from cytomine import Cytomine

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from argparse import ArgumentParser

import tensorflow as tf
import numpy as np
import cv2 as cv

import pickle
import glob
import sys
import os


def parse_data(image, lm):
	"""
	Reads image and landmark files depending on
	specified extension.
	"""
	image_content = tf.io.read_file(image)

	image = tf.io.decode_png(image_content, channels=3)
	image = tf.image.resize(image, (256,256))
	
	#image = tf.image.resize_with_pad(image, 256,256)
	
	return  image, lm

def normalize(image, lm):
	image = tf.cast(image, tf.float32)/255.0
	return image, lm

def maskToKeypoints(mask):
	# mask = np.reshape(mask, newshape=(96,96))
	kp = np.unravel_index(np.argmax(mask, axis=None), shape=(256,256))
	return kp[1], kp[0]

def up_lm(lmks,curr_size, upsize):
	asp_ratio = upsize[0]/upsize[1] #h/w
	
	w = curr_size
	h = w * asp_ratio
	
	up_w = upsize[1]
	up_h = upsize[0]
	
	offset = w - h
	x_lm = lmks[:,0]
	y_lm = lmks[:,1]
	
	y_lm = y_lm - offset//2  #height is reduced
	
	up_y_lm = y_lm * up_h / h
	up_x_lm = x_lm * up_w / w
	
	return np.vstack((up_x_lm, up_y_lm)).T

def pred_lm(model, test_images, test_lmks, org_images, org_lmks, N, batch_size):
	test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_lmks))
	test_ds = test_ds.map(parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	test_ds = test_ds.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
	## Arguments
	parser = ArgumentParser()
	parser.add_argument('--side', dest='side', required=True, help="v or d (ventral or dorsal)")
	parser.add_argument('--species', dest='species', required=True, help="The name(s) of the specie(s)")
	params, _ = parser.parse_known_args(sys.argv[1:])

	## Model selection
	color = 'rgb'
	sigma = '4'
	fct = 'gaussian'

	## True landmark model
	# Load model
	model = load_model('./lm_scripts/saved_models/unet/unet1_'+params.species+'_'+params.side+'_'+color+'_sigma'+sigma+'_'+fct+'.hdf5')

	# Path to original test set
	test_repo = 'D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/'
	org_images = glob.glob(test_repo+'images/*.tif')

	
	## Hyperparameters
	# N = 14 if params.side == 'v' else 18
	N = 15 if params.side == 'v' else 26
	batch_size = 4


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

		pred_landmarks, org_landmarks, lm_ids = pred_lm(model, v[0], v[1], v[2], v[3], N, batch_size)

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