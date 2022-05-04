import random
import shutil
import glob
import os

if __name__ == '__main__':
	repository = 'D:/Dataset_TFE/images_v2/hercules_telemachus_amphitryon/d'

	os.makedirs(repository+'/training/landmarks_v2')
	os.makedirs(repository+'/testing/landmarks_v2')

	lm = [os.path.basename(f) for f in glob.glob(repository+'/training/landmarks/*.txt')]

	for folder in glob.glob(repository+'/landmarks_v2/*.txt'):
		filename = os.path.basename(folder)

		if filename in lm:
			shutil.copy2(folder, repository+'/training/landmarks_v2/'+filename)

		else:
			shutil.copy2(folder, repository+'/testing/landmarks_v2/'+filename)