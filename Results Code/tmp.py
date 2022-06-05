from cytomine.models.image import ImageInstanceCollection
from cytomine import Cytomine

import glob
import os


if __name__ == '__main__':
	host = "https://research.cytomine.be" 
	public_key = "50af36d2-3ccc-4928-a21d-aab32e437fbf"
	private_key = "49725276-a001-4d25-9800-76409c8aad9a"
	id_project = 535588540

	repository = "D:/Dataset_TFE/images_v2/all/d/testing/images/"

	with Cytomine(host=host, public_key=public_key, private_key=private_key) as cytomine:
		
		image_instances = ImageInstanceCollection().fetch_with_filter("project", id_project)
		images_names = {image.id : image.filename for image in image_instances}

		for filepath in glob.glob(repository+'*.tif'):
			try:
				fname = images_names[int(os.path.basename(filepath)[:-4])]
			except (KeyError, ValueError):
				continue

			try:
				os.rename(filepath, repository+fname)
			except FileExistsError:
				os.rename(filepath, repository+fname[:-4]+'_2.tif')