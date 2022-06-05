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


from cytomine.models.image import ImageInstanceCollection
from cytomine.models import TermCollection
from cytomine import Cytomine

from tensorflow.keras.models import load_model
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

	pred_landmarks = []
	up_sizes = []
	for i in range(len(test_images)):
		org_img = cv.imread(org_images[i])
		lm_ids = np.loadtxt(org_lmks[i])[:,0:1]
		pred_mask = preds[i]
		pred_mask = np.reshape(pred_mask, newshape=(256,256, N))
		lm_list = []

		for j in range(N):
			x,y =  maskToKeypoints(pred_mask[:, :, j])
			lm_list.append((x,y))
		pred_lm = np.array(lm_list)
		up_size = org_img.shape[:2]
		up_lmks = up_lm(pred_lm, 256, up_size)
		pred_landmarks.append(up_lmks)
		up_sizes.append(up_size)

	return pred_landmarks, lm_ids, up_sizes

if __name__ == '__main__':
	## Arguments
	parser = ArgumentParser()
	parser.add_argument('--host', dest='host', required=True, help="The Cytomine host")
	parser.add_argument('--public_key', dest='public_key', required=True, help="The Cytomine public key")
	parser.add_argument('--private_key', dest='private_key', required=True, help="The Cytomine private key")
	parser.add_argument('--project_id', dest='project_id', required=True, help="The project from which we want the images")
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

	# Path to rescaled test set
	test_images = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/rescaled/images/*.png')
	test_lmks = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/rescaled/landmarks_v2/*.txt')

	# Path to original test set
	org_images = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/images/*.tif')
	org_lmks = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/testing/landmarks_v2/*.txt')

	## Semi-landmark model
	# Load model
	model_slm = load_model('./lm_scripts/saved_models/unet/unet1_'+params.species+'_slm_v2_'+params.side+'_'+color+'_sigma'+sigma+'_'+fct+'.hdf5')

	# Path to rescaled test set
	test_lmks_slm = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'_slm_v2/'+params.side+'/testing/rescaled/landmarks_v2/*.txt')

	# Path to original test set
	org_lmks_slm = glob.glob('D:/Dataset_TFE/images_v2/'+params.species+'_slm_v2/'+params.side+'/testing/landmarks_v2/*.txt')
	
	
	## Hyperparameters
	N = 14 if params.side == 'v' else 18
	N_slm = 15 if params.side == 'v' else 26
	batch_size = 4


	## Compute predictions
	# True landmark
	pred_landmarks, lm_ids, im_sizes = pred_lm(model, test_images, test_lmks, org_images, org_lmks, N, batch_size)

	# Semi-landmark
	pred_landmarks_slm, slm_ids, _ = pred_lm(model_slm, test_images, test_lmks_slm, org_images, org_lmks_slm, N_slm, batch_size)


	## Write TPS file
	filename = 'predictions/'+params.species+'_'+params.side+'.TPS'
	file = open(filename, 'w')

	fname = 'img_scale_d.pkl' if params.side == 'd' else 'img_scale_v.pkl'
	with open(fname, 'rb') as f:
		image_scales = pickle.load(f)

	with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key) as cytomine:
		image_instances = ImageInstanceCollection().fetch_with_filter("project", params.project_id)
		terms = TermCollection().fetch_with_filter("project", params.project_id)
		images_names = {image.id : image.filename for image in image_instances}
		terms_names = {term.id : term.name for term in terms}
	
	LM = N + N_slm
	for i, landmarks in enumerate(zip(pred_landmarks, pred_landmarks_slm)):
		# Get info about current image
		image_id = os.path.basename(org_images[i])[:-4]
		image_name = images_names[int(image_id)]
		image_height = im_sizes[i][0]

		try:
			image_scale = image_scales[int(image_id)]
		except KeyError:
			continue

		# Sort landmarks according to their names
		tlm_names = [terms_names[lm_ids[i,0]] for i in range(N)]
		slm_names = [terms_names[slm_ids[i,0]] for i in range(N_slm)]

		true_lms = np.array(landmarks[0])
		semi_lms = np.array(landmarks[1])

		true_lms = true_lms[np.argsort(tlm_names)]
		semi_lms = semi_lms[np.argsort(slm_names)]


		# Write number of landmarks
		file.write(f'LM={LM}\n')

		# Write true landmarks
		for tlm in true_lms:
			file.write("%.5f %.5f\n" % (tlm[0], image_height - tlm[1]))

		# Write semi-landmarks
		idx = 6 if params.side == 'd' else 5
		for j in range(idx):
			slm = semi_lms[j]
			file.write('%.5f %.5f\n' % (slm[0], image_height - slm[1]))

		file.write('\n\n')

		for j in range(10):
			slm = semi_lms[idx+j]
			file.write('%.5f %.5f\n' % (slm[0], image_height - slm[1]))

		if params.side == 'd':
			file.write('\n')

			for j in range(10):
				slm = semi_lms[idx+10+j]
				file.write('%.5f %.5f\n' % (slm[0], image_height - slm[1]))

		# Write image name and scale
		file.write('IMAGE='+image_name+'\n')
		file.write('SCALE='+image_scale+'\n')

	file.close()