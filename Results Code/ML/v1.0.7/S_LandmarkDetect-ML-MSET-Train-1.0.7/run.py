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
from ldmtools import *

from cytomine import CytomineJob, Cytomine
from cytomine.models import Job, ImageInstanceCollection, AnnotationCollection, Property, AttachedFile, TermCollection
from sklearn.ensemble import ExtraTreesClassifier

import numpy as np

import random
import joblib
import pickle
import sys


def main():
	with CytomineJob.from_cli(sys.argv) as conn:
		conn.job.update(status=Job.RUNNING, progress=0, status_comment="Initialization of the training phase")

		# 1. Create working directories on the machine:
		# - WORKING_PATH/in: input images
		# - WORKING_PATH/out: output images
		# - WORKING_PATH/ground_truth: ground truth images
		# - WORKING_PATH/tmp: temporary path

		base_path = "{}".format(os.getenv("HOME"))
		gt_suffix = "_lbl"
		working_path = os.path.join(base_path, str(conn.job.id))
		in_path = os.path.join(working_path, "in/")
		in_txt = os.path.join(in_path, 'txt/')
		out_path = os.path.join(working_path, "out/")
		gt_path = os.path.join(working_path, "ground_truth/")
		tmp_path = os.path.join(working_path, "tmp/")

		if not os.path.exists(working_path):
			os.makedirs(working_path)
			os.makedirs(in_path)
			os.makedirs(out_path)
			os.makedirs(gt_path)
			os.makedirs(tmp_path)
			os.makedirs(in_txt)
		# 2. Download the images (first input, then ground truth image)
		conn.job.update(progress=10, statusComment="Downloading images (to {})...".format(in_path))
		print(conn.parameters)
		#images = ImageInstanceCollection().fetch_with_filter("project", conn.parameters.cytomine_id_project)

		# Fetch training images
		random.seed(0)
		# id_terms NEED TO BE d or v
		with open('sp.pkl', 'rb') as file:
			sp = pickle.load(file)
		idx = 0 if conn.parameters.cytomine_id_terms == 'd' else 1
		images_list = list()
		images_to_pred = list()
		if conn.parameters.cytomine_training_images == 'all':
			for k in sp.keys():
				images_list += sp[k][0+idx]
				#images_to_pred += v[2+idx]

		else:
			for specie in conn.parameters.cytomine_training_images.split(','):
				images_list += sp[specie][0+idx]
				#images_to_pred += sp[specie][2+idx]

		# Get test set
		xpos = {}
		ypos = {}
		terms = {}

		terms_collection = TermCollection().fetch_with_filter("project", conn.parameters.cytomine_id_project)
		terms_names = {term.id : term.name for term in terms_collection}
		check_terms = [str(i) + '-v-lm' for i in range(1,15)] + [str(i) + '-d-lm' for i in range(1,19)]

		nb_to_swap = len(images_list) // 5
		for i in range(nb_to_swap):
			images_to_pred.append(images_list.pop(random.randint(0, len(images_list)-i-1)))


		for image in images_list:
			image.dump(dest_pattern=in_path.rstrip('/')+'/%d.%s'%(image.id, 'jpg'))

			annotations = AnnotationCollection()
			annotations.project = conn.parameters.cytomine_id_project
			annotations.showWKT = True
			annotations.showMeta = True
			annotations.showGIS = True
			annotations.showTerm = True
			annotations.image = image.id
			annotations.fetch()

			for ann in annotations:
				if terms_names[ann.term[0]] in check_terms:
					l = ann.location
					poi = shapely.wkt.loads(l)
					(cx, cy) = poi.xy
					xpos[(ann.term[0], image.id)] = int(cx[0])
					ypos[(ann.term[0], image.id)] = image.height - int(cy[0])
					terms[ann.term[0]] = 1

		for image in images_list:
			F = open(in_txt + '%d.txt' % image.id, 'w')
			for t in terms.keys():
				if (t, image.id) in xpos:
					F.write('%d %d %d %f %f\n' % (
					t, xpos[(t, image.id)], ypos[(t, image.id)], xpos[(t, image.id)] / float(image.width),
					ypos[(t, image.id)] / float(image.height)))
			F.close()


		depths = 1. / (2. ** np.arange(conn.parameters.model_depth))

		ims, t_to_i = getallcoords(in_txt)

		#if conn.parameters.cytomine_id_terms == 'all':
		#	term_list = t_to_i.keys()
		#elif conn.parameters.cytomine_id_terms == 'v' or conn.parameters.cytomine_id_terms == 'd':
		with open('t.pkl', 'rb') as file:
			t = pickle.load(file)
		term_list = [int(elem) for elem in t[conn.parameters.cytomine_id_terms]]

		
		tr_im = ims
		
		DATA = None
		REP = None
		be = 0

		sfinal = ""
		for id_term in conn.monitor(term_list, start=10, end=90, period=0.05, prefix="Model building for terms..."):
			sfinal+="%d "%id_term

			(xc, yc, xr, yr) = getcoordsim(in_txt, id_term, tr_im)
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
			if conn.parameters.model_feature_type.lower() == 'gaussian':
				std_matrix = np.eye(2) * (conn.parameters.model_feature_gaussian_std ** 2)
				feature_parameters = np.round(np.random.multivariate_normal([0, 0], std_matrix, conn.parameters.model_feature_gaussian_n)).astype(int)
			elif conn.parameters.model_feature_type.lower() == 'haar':
				W = conn.parameters.model_wsize
				n = conn.parameters.model_feature_haar_n // (5 * conn.parameters.model_depth)
				h2 = generate_2_horizontal(W, n)
				v2 = generate_2_vertical(W, n)
				h3 = generate_3_horizontal(W, n)
				v3 = generate_3_vertical(W, n)
				sq = generate_square(W, n)
				feature_parameters = (h2, v2, h3, v3, sq)

			for times in range(conn.parameters.model_ntimes):
				if times == 0:
					rangrange = 0
				else:
					rangrange = conn.parameters.model_angle

				T = build_datasets_rot_mp(in_path, tr_im, xc, yc, conn.parameters.model_R, conn.parameters.model_RMAX, conn.parameters.model_P, conn.parameters.model_step, rangrange, conn.parameters.model_wsize, conn.parameters.model_feature_type, feature_parameters, depths, nimages, 'jpg', conn.parameters.model_njobs)
				for i in range(len(T)):
					(data, rep, img) = T[i]
					(height, width) = data.shape
					if not passe:
						passe = True
						DATA = np.zeros((height * (len(T) + 100) * conn.parameters.model_ntimes, width))
						REP = np.zeros(height * (len(T) + 100) * conn.parameters.model_ntimes)
						b = 0
						be = height
					DATA[b:be, :] = data
					REP[b:be] = rep
					b = be
					be = be + height

			REP = REP[0:b]
			DATA = DATA[0:b, :]

			clf = ExtraTreesClassifier(n_jobs=conn.parameters.model_njobs, n_estimators=conn.parameters.model_ntrees)
			clf = clf.fit(DATA, REP)

			parameters_hash = {}

			parameters_hash['cytomine_id_terms'] = conn.parameters.cytomine_id_terms
			parameters_hash['model_R'] = conn.parameters.model_R
			parameters_hash['model_RMAX'] = conn.parameters.model_RMAX
			parameters_hash['model_P'] = conn.parameters.model_P
			parameters_hash['model_npred'] = conn.parameters.model_npred
			parameters_hash['model_ntrees'] = conn.parameters.model_ntrees
			parameters_hash['model_ntimes'] = conn.parameters.model_ntimes
			parameters_hash['model_angle'] = conn.parameters.model_angle
			parameters_hash['model_depth'] = conn.parameters.model_depth
			parameters_hash['model_step'] = conn.parameters.model_step
			parameters_hash['window_size'] = conn.parameters.model_wsize
			parameters_hash['feature_type'] = conn.parameters.model_feature_type
			parameters_hash['feature_haar_n'] = conn.parameters.model_feature_haar_n
			parameters_hash['feature_gaussian_n'] = conn.parameters.model_feature_gaussian_n
			parameters_hash['feature_gaussian_std'] = conn.parameters.model_feature_gaussian_std

			model_filename = joblib.dump(clf, os.path.join(out_path, '%d_model.joblib' % (id_term)), compress=3)[0]
			#print(f'\n\n-------- id_term => mx={mx} my={my} cm={cm} ---------\n\n')
			cov_filename = joblib.dump([mx, my, cm], os.path.join(out_path, '%d_cov.joblib' % (id_term)), compress=3)[0]
			parameter_filename = joblib.dump(parameters_hash, os.path.join(out_path, '%d_parameters.joblib' % id_term), compress=3)[0]
			AttachedFile(
				conn.job,
				domainIdent=conn.job.id,
				filename=model_filename,
				domainClassName="be.cytomine.processing.Job"
			).upload()
			AttachedFile(
				conn.job,
				domainIdent=conn.job.id,
				filename=cov_filename,
				domainClassName="be.cytomine.processing.Job"
			).upload()
			AttachedFile(
				conn.job,
				domainIndent=conn.job.id,
				filename=parameter_filename,
				domainClassName="be.cytomine.processing.Job"
			).upload()
			if conn.parameters.model_feature_type == 'haar' or conn.parameters.model_feature_type == 'gaussian':
				add_filename = joblib.dump(feature_parameters, out_path.rstrip('/')+'/'+'%d_fparameters.joblib' % (id_term))[0]
				AttachedFile(
					conn.job,
					domainIdent=conn.job.id,
					filename=add_filename,
					domainClassName="be.cytomine.processing.Job"
				).upload()

		Property(conn.job, key="id_terms", value=sfinal.rstrip(" ")).save()
		conn.job.update(progress=100, status=Job.TERMINATED, statusComment="Job terminated.")

		print('\n\n\nIMAGES TO PREDICT:')
		print(','.join([str(image.id) for image in images_to_pred]))

if __name__ == "__main__":
	main()