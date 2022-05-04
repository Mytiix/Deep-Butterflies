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

from download import *

from sklearn.ensemble import ExtraTreesClassifier
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np

import random
import joblib
import pickle
import glob
import sys

sys.path.insert(0, '../')
from ldmtools import *


def main():
	## Parameters

	# Arguments
	parser = ArgumentParser()
	parser.add_argument('--side', dest='side', required=True, help="v or d (ventral or dorsal)")
	parser.add_argument('--species', dest='species', required=True, help="The name(s) of the specie(s)")
	parser.add_argument('--model_njobs', dest='model_njobs', required=True, type=int)
	parser.add_argument('--model_RMAX', dest='model_RMAX', required=True, type=int)
	parser.add_argument('--model_R', dest='model_R', required=True, type=int)
	parser.add_argument('--model_P', dest='model_P', required=True, type=int)
	parser.add_argument('--model_npred', dest='model_npred', required=True, type=int)
	parser.add_argument('--model_ntrees', dest='model_ntrees', required=True, type=int)
	parser.add_argument('--model_ntimes', dest='model_ntimes', required=True, type=int)
	parser.add_argument('--model_angle', dest='model_angle', required=True, type=int)
	parser.add_argument('--model_depth', dest='model_depth', required=True, type=int)
	parser.add_argument('--model_step', dest='model_step', required=True, type=int)
	parser.add_argument('--model_wsize', dest='model_wsize', required=True, type=int)
	parser.add_argument('--model_feature_type', dest='model_feature_type', required=True)
	parser.add_argument('--model_feature_haar', dest='model_feature_haar_n', required=True, type=int)
	parser.add_argument('--model_feature_gaussian_n', dest='model_feature_gaussian_n', required=True, type=int)
	parser.add_argument('--model_feature_gaussian_std', dest='model_feature_gaussian_std', required=True, type=int)
	params, _ = parser.parse_known_args(sys.argv[1:])

	# Define repositories
	repository = 'D:/Dataset_TFE/images_v2/'+params.species+'/'+params.side+'/training/'
	lm_repo = repository+'landmarks_v2/'
	im_repo = repository+'images/'

	if not os.path.exists(repository+'out/'):
		os.makedirs(repository+'out/')
	out_repo = repository+'out/'

	# Get extra parameters
	depths = 1. / (2. ** np.arange(params.model_depth))
	tr_im, term_list = getallcoords(lm_repo)

	
	## Train models

	# Initialization
	DATA = None
	REP = None
	be = 0

	# Train a model for each landmark
	sfinal = ""
	for id_term in tqdm(term_list):
		sfinal+="%d "%id_term

		(xc, yc, xr, yr) = getcoordsim(lm_repo, id_term, tr_im)
		nimages = np.max(xc.shape)
		mx = np.mean(xr)
		my = np.mean(yr)
		P = np.zeros((2, nimages))
		P[0, :] = xr
		P[1, :] = yr
		cm = np.cov(P)
		passe = False

		# additional parameters
		feature_parameters = None
		if params.model_feature_type.lower() == 'gaussian':
			std_matrix = np.eye(2) * (params.model_feature_gaussian_std ** 2)
			feature_parameters = np.round(np.random.multivariate_normal([0, 0], std_matrix, params.model_feature_gaussian_n)).astype(int)
		
		elif params.model_feature_type.lower() == 'haar':
			W = params.model_wsize
			n = params.model_feature_haar_n // (5 * params.model_depth)
			h2 = generate_2_horizontal(W, n)
			v2 = generate_2_vertical(W, n)
			h3 = generate_3_horizontal(W, n)
			v3 = generate_3_vertical(W, n)
			sq = generate_square(W, n)
			feature_parameters = (h2, v2, h3, v3, sq)

		for times in range(params.model_ntimes):
			if times == 0:
				rangrange = 0
			else:
				rangrange = params.model_angle

			T = build_datasets_rot_mp(im_repo, tr_im, xc, yc, params.model_R, params.model_RMAX, params.model_P, params.model_step, rangrange, params.model_wsize, params.model_feature_type, feature_parameters, depths, nimages, 'tif', params.model_njobs)
			for i in range(len(T)):
				(data, rep, img) = T[i]
				(height, width) = data.shape
				if not passe:
					passe = True
					DATA = np.zeros((height * (len(T) + 100) * params.model_ntimes, width))
					REP = np.zeros(height * (len(T) + 100) * params.model_ntimes)
					b = 0
					be = height

				DATA[b:be, :] = data
				REP[b:be] = rep
				b = be
				be = be + height

		REP = REP[0:b]
		DATA = DATA[0:b, :]

		# Fit the model
		clf = ExtraTreesClassifier(n_jobs=params.model_njobs, n_estimators=params.model_ntrees)
		clf = clf.fit(DATA, REP)

		# Save model and extra info for prediction
		parameters_hash = {}
		parameters_hash['model_R'] = params.model_R
		parameters_hash['model_RMAX'] = params.model_RMAX
		parameters_hash['model_P'] = params.model_P
		parameters_hash['model_npred'] = params.model_npred
		parameters_hash['model_ntrees'] = params.model_ntrees
		parameters_hash['model_ntimes'] = params.model_ntimes
		parameters_hash['model_angle'] = params.model_angle
		parameters_hash['model_depth'] = params.model_depth
		parameters_hash['model_step'] = params.model_step
		parameters_hash['window_size'] = params.model_wsize
		parameters_hash['feature_type'] = params.model_feature_type
		parameters_hash['feature_haar_n'] = params.model_feature_haar_n
		parameters_hash['feature_gaussian_n'] = params.model_feature_gaussian_n
		parameters_hash['feature_gaussian_std'] = params.model_feature_gaussian_std

		model_filename = joblib.dump(clf, os.path.join(out_repo, '%d_model.joblib' % (id_term)), compress=3)[0]
		cov_filename = joblib.dump([mx, my, cm], os.path.join(out_repo, '%d_cov.joblib' % (id_term)), compress=3)[0]
		parameter_filename = joblib.dump(parameters_hash, os.path.join(out_repo, '%d_parameters.joblib' % id_term), compress=3)[0]
		
		if params.model_feature_type == 'haar' or params.model_feature_type == 'gaussian':
			add_filename = joblib.dump(feature_parameters, out_repo.rstrip('/')+'/'+'%d_fparameters.joblib' % (id_term))[0]

if __name__ == "__main__":
	main()