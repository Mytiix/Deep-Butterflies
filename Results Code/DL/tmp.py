import random
import shutil
import glob
import os

if __name__ == '__main__':
	repository = 'D:/Dataset_TFE/images_v2/all_slm_v2'

	for folder in glob.glob(repository+'/*'):	
		lm = [os.path.basename(f)[:-4]+'.txt' for f in glob.glob(folder+'/training/images/*.tif')]
	
		for file in glob.glob(folder+'/landmarks_v2/*.txt'):
			filename = os.path.basename(file)

			if filename in lm:
				shutil.copy2(file, folder+'/training/landmarks_v2/'+filename)

			else:
				shutil.copy2(file, folder+'/testing/landmarks_v2/'+filename)