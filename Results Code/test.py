import logging

from shapely.geometry import Point
from collections import defaultdict
from tabulate import tabulate

from cytomine import Cytomine
from cytomine.models import Property, Annotation, AnnotationTerm, AnnotationCollection, TermCollection
from cytomine.models.image import ImageInstanceCollection

import pickle
import glob
import tqdm

logging.basicConfig()
logger = logging.getLogger("cytomine.client")
logger.setLevel(logging.INFO)

if __name__ == '__main__':
	host = "https://research.cytomine.be" 
	public_key = "50af36d2-3ccc-4928-a21d-aab32e437fbf"
	private_key = "49725276-a001-4d25-9800-76409c8aad9a"
	id_project = 535588540
	image_name = "PB13.0397-v-1.tif"
	# 535842433,535836746,535829645,535823929,535773914 544004777

	
	with Cytomine(host=host, public_key=public_key, private_key=private_key) as cytomine:
		
		image_instances = ImageInstanceCollection().fetch_with_filter("project", id_project)
		terms = TermCollection().fetch_with_filter("project", id_project)
		
		'''	
		## Save mapping between species and image instance
		sp = {}
		for image in image_instances:
			prop = Property(image).fetch(key='specie')
			sp[image] = prop.value

		# Dict => {specie : [d, v, no_annot_d, no_annot_v]}
		new_sp = defaultdict(lambda: [list(), list(), list(), list()])
		for k, v in sp.items():
			if k.numberOfAnnotations == 0:
				if k.instanceFilename[10] == 'd':
					new_sp[v][2].append(k)
				elif k.instanceFilename[10] == 'v':
					new_sp[v][3].append(k)
			elif k.instanceFilename[10] == 'd':
				new_sp[v][0].append(k)
			elif k.instanceFilename[10] == 'v':
				new_sp[v][1].append(k)
			else:
				print('ERROR')
			
		file = open('sp.pkl', 'wb')
		pickle.dump(dict(new_sp), file)
		print(new_sp)
		'''

		
		# Print sp data
		with open('sp.pkl', 'rb') as file:
			sp = pickle.load(file)
		table = [[k, len(v[0]), len(v[1]), len(v[2]), len(v[3])] for k, v in sp.items()]
		print(tabulate(table, headers=['Specie', 'd', 'v', 'd0', 'v0']))
		
			
		
		'''
		## Save mapping for terms
		#v = [str(i) + '-v-lm' for i in range(1,15)]
		#d = [str(i) + '-d-lm' for i in range(1,19)]
		v = [str(i) + '-v-slm' for i in range(15,30)]
		d = [str(i) + '-d-slm' for i in range(19,45)]
		t = defaultdict(lambda: list())
		for term in terms:
			if term.name in v:
				t['v'].append(term.id)
				print('v =>', term.name)
			if term.name in d:
				t['d'].append(term.id)
				print('d =>', term.name)
			#print(term.name)
		file = open('t.pkl', 'wb')
		pickle.dump(dict(t), file)
		print(t)
		'''

		'''
		images_ids = {}
		for image in image_instances:
			if image.filename in images_ids.keys():
				images_ids[image.filename].append(image.id)
			else:
				images_ids[image.filename] = [image.id]
		'''
		
		# images_ids = {image.filename : image.id for image in image_instances}
		# terms_ids = {term.name : term.id for term in terms}
		# terms_names = {term.id : term.name for term in terms}

		# file = open('terms_names.pkl', 'wb')
		# pickle.dump(terms_names, file)
		# file.close()

		# print(terms_names[539978547])
		# print(terms_names[539978278])
		# print(terms_names[539978532])
		# print(terms_names[539978259])
		# for k,v in terms_ids.items():
		# 	print(k,v)
		'''
		print(images_ids['PB13.0001-v.tif'],images_ids['PB13.0002-v.tif'],images_ids['PB13.0003-v.tif'],images_ids['PB13.0004-v.tif'],images_ids['PB13.0005-v.tif'], sep=',')
		print(terms_ids['1-v-lm'])
		'''



#python .\upload_image.py --cytomine_host "https://research.cytomine.be" --cytomine_public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --cytomine_private_key 49725276-a001-4d25-9800-76409c8aad9a --cytomine_upload_host "https://research-upload.cytomine.be" --cytomine_id_project 535588540 --filepath "D:\Users\marga\Downloads\Dataset_TFE\Images\397-425\PB13.0397-d-1.tif"
#python .\upload_dataset.py --cytomine_host "https://research.cytomine.be" --cytomine_public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --cytomine_private_key 49725276-a001-4d25-9800-76409c8aad9a --cytomine_upload_host "https://research-upload.cytomine.be" --cytomine_id_project 535588540 --dataset_path "D:\Users\marga\Downloads\Dataset_TFE\Images\"


#PB13.0397-d-1.tif


# python run.py --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --software_id 127520103 --project_id 535588540 --cytomine_id_terms 539978191 --cytomine_training_images 535637680 --model_njobs 1 --model_RMAX 100 --model_R 6 --model_P 3 --model_npred 50000 --model_ntrees 50 --model_ntimes 1 --model_angle 10 --model_depth 5 --model_step 1 --model_wsize 8 --model_feature_type raw --model_feature_haar 1600 --model_feature_gaussian_n 1600 --model_feature_gaussian_std 20

# python run.py --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --software_id 127520103 --project_id 6575282 --cytomine_id_terms all --cytomine_training_images all --model_njobs 1 --model_RMAX 100 --model_R 6 --model_P 3 --model_npred 50000 --model_ntrees 50 --model_ntimes 1 --model_angle 10 --model_depth 5 --model_step 1 --model_wsize 8 --model_feature_type raw --model_feature_haar 1600 --model_feature_gaussian_n 1600 --model_feature_gaussian_std 20

# docker run droso --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --software_id 127520103 --project_id 6575282 --cytomine_id_terms 6579647 --cytomine_training_images 6603655 --model_njobs 1 --model_RMAX 100 --model_R 6 --model_P 3 --model_npred 50000 --model_ntrees 50 --model_ntimes 1 --model_angle 10 --model_depth 5 --model_step 1 --model_wsize 8 --model_feature_type raw --model_feature_haar 1600 --model_feature_gaussian_n 1600 --model_feature_gaussian_std 20

####### COMMANDS
## DROSO
# docker build -t droso .
# docker run droso --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --software_id 127520103 --project_id 6575282 --cytomine_id_terms 6579647 --cytomine_training_images 6603655 --model_njobs 1 --model_RMAX 100 --model_R 6 --model_P 3 --model_npred 50000 --model_ntrees 50 --model_ntimes 1 --model_angle 10 --model_depth 5 --model_step 1 --model_wsize 8 --model_feature_type raw --model_feature_haar 1600 --model_feature_gaussian_n 1600 --model_feature_gaussian_std 20
# --log_level DEBUG > log_DEBUG.txt

# docker build -t droso_pred .
# docker run droso_pred --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --software_id 127518601 --project_id 6575282 --images_to_predict 6603655,6603649 --model_to_use 543209747

## BUTTERFLIES
# docker build -t butterflies .
# docker run butterflies --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --software_id 127520103 --project_id 535588540 --cytomine_id_terms 540036969,540036979,540036985,540036993,540037005,540037031,540037039,540037045,540037053,540037059,540037069,540037077,540037085,540037091 --cytomine_training_images 535662336,535751048,535756645,535758151,535759624,535763742,535765340,535768198,535771056,535776772,535608001,535779630,535782488,535785357,535662042,535788204,535793832,535799647,535805352,535808210,535811068,535813926,535816785,535819648,535825358,535832280,535836790,535843891 --model_njobs 1 --model_RMAX 100 --model_R 6 --model_P 3 --model_npred 50000 --model_ntrees 50 --model_ntimes 1 --model_angle 10 --model_depth 5 --model_step 1 --model_wsize 8 --model_feature_type raw --model_feature_haar 1600 --model_feature_gaussian_n 1600 --model_feature_gaussian_std 20

# docker run droso_pred --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --software_id 127518601 --project_id 535588540 --images_to_predict 535773914,535793920,535798163,535803790 --model_to_use 543838212

# docker build -t morpho .
# docker run morpho_v2 --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --software_id 127520103 --project_id 535588540 --cytomine_id_terms v --cytomine_training_images polyphemus --model_njobs 4 --model_RMAX 100 --model_R 6 --model_P 3 --model_npred 50000 --model_ntrees 50 --model_ntimes 1 --model_angle 10 --model_depth 5 --model_step 1 --model_wsize 8 --model_feature_type raw --model_feature_haar 1600 --model_feature_gaussian_n 1600 --model_feature_gaussian_std 20

# Param from paper
# docker run morpho --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --software_id 127520103 --project_id 535588540 --cytomine_id_terms v --cytomine_training_images polyphemus --model_njobs 4 --model_RMAX 300 --model_R 9 --model_P 2 --model_npred 30000 --model_ntrees 50 --model_ntimes 3 --model_angle 30 --model_depth 5 --model_step 2 --model_wsize 8 --model_feature_type raw --model_feature_haar 1600 --model_feature_gaussian_n 1600 --model_feature_gaussian_std 20

# python eval.py --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --project_id 535588540 --job_id 544244203

# python plot_mean_baseline.py --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --project_id 535588540 --side v --term 14 --species amphitryon --npred 30000

# python mean_baseline.py --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --project_id 535588540 --side v --species all

# python run.py --side v --species polyphemus --model_njobs 4 --model_RMAX 100 --model_R 6 --model_P 3 --model_npred 50000 --model_ntrees 50 --model_ntimes 1 --model_angle 10 --model_depth 5 --model_step 1 --model_wsize 8 --model_feature_type raw --model_feature_haar 1600 --model_feature_gaussian_n 1600 --model_feature_gaussian_std 20

# python .\generate_tps.py --host https://research.cytomine.be --public_key 50af36d2-3ccc-4928-a21d-aab32e437fbf --private_key 49725276-a001-4d25-9800-76409c8aad9a --project_id 535588540 --side v --species all

####### MODELS

## ID models droso:
# raw all all => 543211756
# sub all all => 543209747


## ID models morpho:
# First tests

# raw hercules v 5 => 543835569
# raw hercules v all => 543838212 / 544004777
# raw hercules v all => 544006604
# raw hercules v all (test on sliding landmarks) => 544111815
# raw hercules 5 => 543834926 (with every images)


# Results

# raw hercules v all => 544376109
# sub hercules v all => 544470615 
# sub hercules v all 10W => 544687133
# sub hercules v all 7D => 544699392
# gaussian hercules v all => 545658963 - 545660881
# raw hercules d all => 544375067 
# sub hercules d all => 544472539 - 544472539
# sub hercules d all 10W => 544683041
# sub hercules d all 7D => 544695198
# gaussian hercules d all => 545663813 - 545665898

# raw polyphemus v all => 544033676 / 544107267
# raw polyphemus v all p2 => 545020102 - 545020958
# raw polyphemus v all (test on sliding landmarks) => 544109618
# raw polyphemus v all (test with paper parm) => 544880861 
# sub polyphemus v all => 544466864
# gaussian polyphemus v all => 545592644 - 545593545
# haar polyphemus v all => 545016907 - 545031288
# raw polyphemus d all => 544338505
# raw polyphemus d all (test on sliding landmarks) => 544338305
# raw polyphemus d all (test with paper parm) => 544884376 - 544891094
# sub polyphemus d all => 544467028
# gaussian polyphemus d all => 545595581 - 545596772

# raw granadensis v all => 544371845
# sub granadensis v all => 544475981 - 544475981
# sub granadensis v all 10W => 544678411 - 544679450
# sub granadensis v all 7D => 544690730
# gaussian granadensis v all => 545668633 - 545669438
# raw granadensis d all => 544373436
# sub granadensis d all => 544481589 - 544482946
# sub granadensis d all 10W => 544680554 - 544681732
# sub granadensis d all 7D => 544692823
# gaussian granadensis d all => 545670334 - 545671279


# raw telemachus v all => 544426328
# sub telemachus v all => 544480731 - 544482762
# gaussian telemachus v all => 545672767 - 545674127
# raw telemachus d all => 544427595
# sub telemachus d all => 544484969 - 544489197
# gaussian telemachus d all => 545677781 - 545680487

# raw amphitryon v all => 544430071
# raw polyphemus v all p2 => 545022961 - 545027295
# gaussian amphitryon v all => 545671697 - 545673382
# sub amphitryon v all => 544484822 - 544487485
# raw amphitryon d all => 544432454
# sub amphitryon d all => 544489014 - 544492518
# gaussian amphitryon d all => 545675212 - 545676420


########### NOTE
# hercules from 1 to 32 / [10 19 20 22]

# - Auto train for specific species (local) DONE
#   - Adjust im to predict (maybe not always im with 0 annot for a specie) DONE
#   - Try other species => Tried: [hercules,polyphemus,]
#   - Add case to train everything at once ? DONE
# - Do eval code
# - Don't download useless images DONE
# - Correct getallcoord() ? DONE
#   - Try and lanch without it DONE
# - Try with clean new projects ?
# - Error 404 on link in email (Computer Vision slm)