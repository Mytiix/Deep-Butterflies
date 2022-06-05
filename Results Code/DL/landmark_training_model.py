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

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from cytomine import CytomineJob

from landmark_HM_models import UNET

import albumentations as A
import tensorflow as tf
import numpy as np

import random
import glob
import math
import sys
import os

from tensorflow.python.client import device_lib
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(device_lib.list_local_devices())


# Custom data augmentation
transforms = A.Compose([
					   A.RandomBrightnessContrast(),
					   A.Affine(scale=(0.8,1), 
								translate_px= (0, 10), 
								rotate= (-10,10), 
								shear=(0,10), 
								interpolation=0, # nearest
								mode=2, # mode replicate
								fit_output=False), 
					   A.HorizontalFlip(p=0.3),
					   ], 
						keypoint_params=A.KeypointParams(format='xy',remove_invisible=False))


# Read image and convert to tensorflow object
def parse_data(image, lm):
	image_content = tf.io.read_file(image)
	image = tf.io.decode_png(image_content, channels=3)
	image = tf.image.resize(image, (256,256))
	return  image, lm

# Normalize image
def normalize(image, lm):
	image = tf.cast(image, tf.float32)/255.0
	return image, lm

# Apply custom data augmentation
def aug_fn(image, lm):
	image = tf.keras.preprocessing.image.img_to_array(image, dtype=np.uint8)
	lm = lm.numpy()
	aug_data = transforms(image=image, keypoints=lm)
	image_aug = aug_data['image'] 
	lm_aug = aug_data['keypoints']
	lm_aug = np.array(lm_aug, dtype=np.int32)
	return image_aug, lm_aug

# Apply custom data augmentation through py_function
def aug_apply(image,lm):
	image, lm = tf.py_function(aug_fn, (image, lm), (tf.float32, tf.float32))
	image.set_shape(image_size)
	lm.set_shape((N,2))
	return image, lm

# Exponential probability function which describes the spread of the heatmap
def _exp(xL, yL, sigma, H, W):
	xx, yy = np.mgrid[0:H:1, 0:W:1]
	kernel = np.exp(-np.log(2) * 0.5 * (np.abs(yy - xL) + np.abs(xx - yL)) / sigma)
	kernel = np.float32(kernel)
	return kernel

# Gaussian probibility function which describes the spread of the heatmap
def _gaussian(xL, yL, sigma, H, W):
	xx, yy = np.mgrid[0:H:1, 0:W:1]
	kernel = np.exp(-0.5 * (np.square(yy - xL) + np.square(xx - yL)) / np.square(sigma))
	kernel = np.float32(kernel)
	return kernel


# Convert an image to an heatmap
def _convertToHM(img, keypoints):
	H = img.shape[0]
	W = img.shape[1]
	nKeypoints = len(keypoints)

	img_hm = np.zeros(shape=(H, W, nKeypoints), dtype=np.float32)

	for i in range(0, nKeypoints):
		x = keypoints[i][0]
		y = keypoints[i][1]

		channel_hm = _exp(x, y, sigma, H, W) if fct == 'exp' else _gaussian(x, y, sigma, H, W)
		img_hm[:, :, i] = channel_hm
	
	img_hm = np.reshape(img_hm, newshape=(img_hm.shape[0]*img_hm.shape[1]*nKeypoints, 1))
	return img, img_hm

# Convert an image to a heatmap through py_function
def to_hm(image, lm):
	image, lm = tf.py_function(_convertToHM, (image,lm), (tf.float32, tf.float32))
	image.set_shape(image_size)
	lm.set_shape((256*256*N, 1))
	return image, lm  


if __name__ == "__main__":

	# Params
	species = 'all_lm_slm'
	side = 'v'
	color = 'rgb'
	fct = 'gaussian'
	from_save = False
	
	repository = 'D:/Dataset_TFE/images_v2/'+species+'/'+side+'/training/rescaled'
	filename = species+'_'+side+'_'+color
	n_epochs = int(sys.argv[1])
	batch_size = int(sys.argv[2])
	sigma = int(sys.argv[3])
	
	image_size = (256,256,3) if color == 'rgb' else (256,256,1) # input image size to the model
	channels = [3,1]  #for decoding images in tf 3:RGB, 1:gray
	# N = 14 if side == 'v' else 18 #number of landmarks
	# N = 15 if side == 'v' else 26
	N = 29 if side == 'v' else 44
	seed = 44 # to generate the same indexes for each re-run
	
	image_paths = glob.glob(os.path.join(repository,'images/*.png'))
	lm_paths = glob.glob(os.path.join(repository,'landmarks_v2/*.txt'))
	
	idxs = []
	kp_list = []
	for i in range(len(lm_paths)):  # loading landmarks as list of arrays
		idxs.append(i)
				
		kp = np.loadtxt(lm_paths[i])
		kp = np.array(kp).astype(np.int32)
		#kp = kp.flatten('C')
		kp_list.append(kp)

#============== tf dataset generation =========================================
		
	train_idxs, val_idxs = train_test_split(idxs, test_size=0.1, random_state=seed) 
	
	train_images = [image_paths[i] for i in train_idxs]
	train_lmks = [kp_list[i] for i in train_idxs]
	val_images = [image_paths[i] for i in val_idxs]
	val_lmks = [kp_list[i] for i in val_idxs] 
	
	
	steps_per_epoch = math.ceil(len(train_images) / batch_size)
	val_steps = math.ceil(len(val_images) / batch_size)
	
#=========== train dataset ====================================================
	tr_ds = tf.data.Dataset.from_tensor_slices((train_images, train_lmks))
	tr_ds = tr_ds.map(parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	tr_ds = tr_ds.map(aug_apply, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	tr_ds = tr_ds.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	tr_ds = tr_ds.map(to_hm, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	tr_ds = tr_ds.prefetch(tf.data.experimental.AUTOTUNE).shuffle(
			random.randint(0, len(image_paths))).batch(batch_size).repeat()
#=========== validation dataset (no augmentation) =============================    
	val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_lmks))
	val_ds = val_ds.map(parse_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)   
	val_ds = val_ds.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	val_ds = val_ds.map(to_hm, num_parallel_calls=tf.data.experimental.AUTOTUNE) 
	val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE).shuffle(
			random.randint(0, len(image_paths))).batch(batch_size).repeat()
#================= model building and training ================================    
	model_name = 'unet' # or FCN8
	filepath="./lm_scripts/saved_models/unet/"+model_name+str(1)+'_'+filename+'_sigma'+str(sigma)+'_'+fct+".hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks = [checkpoint]
	

	if from_save:
		model = tf.keras.models.load_model(filepath)
	else:
		optim = RMSprop(learning_rate=0.001)
		rmse = tf.keras.metrics.RootMeanSquaredError()
		mse = tf.keras.metrics.MeanSquaredError()
		model = UNET(input_shape=image_size, H=256, W=256, nKeypoints=N)
		model.compile(loss="mse", optimizer=optim, metrics=[rmse])
	history = model.fit(tr_ds, epochs=n_epochs, callbacks=callbacks,
							validation_data=val_ds, steps_per_epoch=steps_per_epoch,
							validation_steps= val_steps,verbose=1)
	

	
	
	
	
	
		
	
		
	
		
	
		
	
		
	
		
	
	

	



