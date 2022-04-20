import random
import shutil
import glob
import os

if __name__ == '__main__':
	repository = 'D:\\Dataset_TFE\\images_v2'

	for folder in glob.glob(repository+'/*/*'):

		# Create directories
		if not os.path.exists(folder+'/training'):
			os.makedirs(folder+'/training')
			os.makedirs(folder+'/training/images')
			os.makedirs(folder+'/training/landmarks')
		if not os.path.exists(folder+'/testing'):
			os.makedirs(folder+'/testing')
			os.makedirs(folder+'/testing/images')
			os.makedirs(folder+'/testing/landmarks')

		# Compute splits
		files = glob.glob(folder+'/images/*.tif')
		nb_test = len(files) // 4
		random_splits = [random.randint(0, len(files)-i-1) for i in range(nb_test)]

		# Move images
		for rnd in random_splits:
			file = files.pop(rnd)
			shutil.copy2(file, folder+'/testing/images/'+os.path.basename(file))

		for file in files:
			shutil.copy2(file, folder+'/training/images/'+os.path.basename(file))

		# Move landmarks
		files = glob.glob(folder+'/landmarks/*.txt')
		for rnd in random_splits:
			file = files.pop(rnd)
			shutil.copy2(file, folder+'/testing/landmarks/'+os.path.basename(file))

		for file in files:
			shutil.copy2(file, folder+'/training/landmarks/'+os.path.basename(file))