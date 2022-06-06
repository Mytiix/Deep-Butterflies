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


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

from landmark_HM_models import UNET
from utils import *

import random
import glob
import math
import sys


if __name__ == "__main__":
	## Parameters
	# Arguments
	parser = ArgumentParser()
	parser.add_argument('--side', dest='side', required=True, help="v or d (ventral or dorsal)")
	parser.add_argument('--species', dest='species', required=True, help="The name of the folder containing the data")
	parser.add_argument('--lm_type', dest='lm_type', required=True, help="Type of landmarks (lm, slm, all)")
	parser.add_argument('--prob_fct', dest='prob_fct', required=True, help="The probability function used to describe the heatmap dispersion (exp, gaussian)")
	parser.add_argument('--n_epochs', dest='n_epochs', required=True, type=int, help="Number of epochs")
	parser.add_argument('--batch_size', dest='batch_size', required=True, type=int, help="Size of batchs")
	parser.add_argument('--sigma', dest='sigma', required=True, type=int, help="Value of sigma (in probability function)")
	params, _ = parser.parse_known_args(sys.argv[1:])

	# Extra params
	from_save = False
	seed = 44 

	if params.lm_type == 'lm':
		N = 14 if params.side == 'v' else 18
	elif params.lm_type == 'slm':
		N = 15 if params.side == 'v' else 26
	else:
		N = 29 if params.side == 'v' else 44

	# Define repository
	repository = 'D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/training/rescaled'
	image_paths = glob.glob(os.path.join(repository,'images/*.png'))
	lm_paths = glob.glob(os.path.join(repository,'landmarks_v2/*.txt'))
	
	## Construct training and validation set with tensorflow
	# Loading landmarks as list of arrays
	idxs = []
	kp_list = []
	for i in range(len(lm_paths)):
		idxs.append(i)
		kp = np.loadtxt(lm_paths[i])
		kp = np.array(kp).astype(np.int32)
		kp_list.append(kp)

	# Generation of tensorflow dataset
	train_idxs, val_idxs = train_test_split(idxs, test_size=0.1, random_state=seed) 
	
	train_images = [image_paths[i] for i in train_idxs]
	train_lmks = [kp_list[i] for i in train_idxs]
	val_images = [image_paths[i] for i in val_idxs]
	val_lmks = [kp_list[i] for i in val_idxs] 
	
	
	steps_per_epoch = math.ceil(len(train_images) / params.batch_size)
	val_steps = math.ceil(len(val_images) / params.batch_size)
	
	# Train dataset
	tr_ds = tf.data.Dataset.from_tensor_slices((train_images, train_lmks))
	tr_ds = tr_ds.map(lambda image, lm: parse_data(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	tr_ds = tr_ds.map(lambda image, lm: aug_apply(image, lm, N), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	tr_ds = tr_ds.map(lambda image, lm: normalize(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	tr_ds = tr_ds.map(lambda image, lm: to_hm(image, lm, N, params.sigma, params.prob_fct), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	tr_ds = tr_ds.prefetch(tf.data.experimental.AUTOTUNE).shuffle(
			random.randint(0, len(image_paths))).batch(params.batch_size).repeat()

	# Validation dataset
	val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_lmks))
	val_ds = val_ds.map(lambda image, lm: parse_data(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)   
	val_ds = val_ds.map(lambda image, lm: normalize(image, lm), num_parallel_calls=tf.data.experimental.AUTOTUNE)
	val_ds = val_ds.map(lambda image, lm: to_hm(image, lm, N, params.sigma, params.prob_fct), num_parallel_calls=tf.data.experimental.AUTOTUNE) 
	val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE).shuffle(
			random.randint(0, len(image_paths))).batch(params.batch_size).repeat()


	## Train the model
	# Checkpoint setup
	filename = 'unet_'+params.species+'_'+params.side+'_sigma'+str(params.sigma)+'_'+params.prob_fct+".hdf5"
	filepath="./lm_scripts/saved_models/unet/"+filename
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks = [checkpoint]
	
	# Pre-trained
	if from_save:
		model = tf.keras.models.load_model(filepath)

	# From scratch
	else:
		optim = RMSprop(learning_rate=0.001)
		rmse = tf.keras.metrics.RootMeanSquaredError()
		model = UNET(input_shape=(256,256,3), H=256, W=256, nKeypoints=N)
		model.compile(loss="mse", optimizer=optim, metrics=[rmse])

	model.fit(tr_ds, epochs=params.n_epochs, callbacks=callbacks,
				validation_data=val_ds, steps_per_epoch=steps_per_epoch,
				validation_steps= val_steps, verbose=1)