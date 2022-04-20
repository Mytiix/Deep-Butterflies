# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:45:17 2022

@author: Navdeep Kumar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import glob
import tensorflow as tf
import cv2 as cv
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras import utils
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model, load_model

# Params
species = 'all'
side = 'v'
color = 'rgb'
sigma = '4'
savefig = True
show = False
extra = True

# Load model
model = load_model('./lm_scripts/saved_models/unet/save/unet1_all_v_rgb_800e.hdf5')
#model = load_model('./lm_scripts/saved_models/unet/unet1_'+species+'_'+side+'_'+color+'_sigma'+sigma+'.hdf5')

# Path to rescaled test set
test_images = glob.glob('D:/Dataset_TFE/images_v2/'+species+'/'+side+'/testing/rescaled/images/*.png')
test_lmks = glob.glob('D:/Dataset_TFE/images_v2/'+species+'/'+side+'/testing/rescaled/landmarks/*.txt')

# Path to original size test set
org_images = glob.glob('D:/Dataset_TFE/images_v2/'+species+'/'+side+'/testing/images/*.tif')
org_lmks = glob.glob('D:/Dataset_TFE/images_v2/'+species+'/'+side+'/testing/landmarks/*.txt')


#============== hyperparameters ===============================================

image_size = (256,256,3) if color == 'rgb' else (256,256,1)
channels = [3,1]
N = 14 if side == 'v' else 18
batch_size = 4

#================ Mapping functions ===========================================

def parse_data(image, lm):
	"""
	Reads image and landmark files depending on
	specified extension.
	"""
	image_content = tf.io.read_file(image)

	chan = channels[0] if color == 'rgb' else channels[1]
	image = tf.io.decode_png(image_content, channels=chan)
	image = tf.image.resize(image, (256,256))
	
	#image = tf.image.resize_with_pad(image, 256,256)
	
	return  image, lm

def normalize(image, lm):
	image = tf.cast(image, tf.float32)/255.0
	return image, lm

	#========= some utils ========================================================
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

# Compute predictions
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_lmks))
test_ds = test_ds.map(parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(batch_size)

preds = model.predict(test_ds)

org_landmarks = []
pred_landmarks= []
for i in range(len(test_images)):
	org_img = cv.imread(org_images[i])
	org_lm = np.loadtxt(org_lmks[i])
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

# Compute RMSE
error = []
for k in range(len(org_landmarks)):
	g = org_landmarks[k]
	p = pred_landmarks[k]
	mse = mean_squared_error(g, p, squared=False)
	error.append(mse)
mean_error = np.array(error).mean()
print('Mean RMSE =>', mean_error)

# Compute Hit Rate
threshold = 30
ht = np.zeros(N)
for i in range(len(org_landmarks)):
	for j in range(N):
		g_lm = org_landmarks[i][j]
		p_lm = pred_landmarks[i][j]
		
		dist = np.linalg.norm(g_lm - p_lm)
		ht[j] += 1 if dist <= threshold else 0

ht /= len(org_landmarks)
ht *= 100
print('Mean Hit Rate =>', np.mean(ht))

# Plot/Save results
if savefig or show:
	filename = species+'_'+side+'_'+color+'_sigma'+sigma
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
		if show:
			plt.show()
		plt.close()


if extra:
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
	plt.ylabel('Hit Rate')
	plt.plot(mean_ht)
	plt.savefig('mean_ht_'+filename+'.pdf')
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
	plt.ylabel('Hit Rate')
	for i in range(N):
		plt.plot(ht_lm[:,i])
	plt.savefig('ht_lm_'+filename+'.pdf')
	plt.show()