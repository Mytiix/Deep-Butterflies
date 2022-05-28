import logging
import glob
import tqdm
import sys
import os

from argparse import ArgumentParser

from cytomine import Cytomine
from cytomine.models import StorageCollection, Project, UploadedFile

logging.basicConfig()
logger = logging.getLogger("cytomine.client")
logger.setLevel(logging.INFO)

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--cytomine_host', dest='host',
						default='demo.cytomine.be', help="The Cytomine host")
	parser.add_argument('--cytomine_public_key', dest='public_key',
						help="The Cytomine public key")
	parser.add_argument('--cytomine_private_key', dest='private_key',
						help="The Cytomine private key")
	parser.add_argument('--cytomine_upload_host', dest='upload_host',
						default='demo-upload.cytomine.be', help="The Cytomine upload host")
	parser.add_argument('--cytomine_id_project', dest='id_project', required=False,
						help="The project from which we want the images (optional)")
	parser.add_argument('--dataset_path', dest='dataset_path',
						help='Path to the dataset to import')
	params, _ = parser.parse_known_args(sys.argv[1:])

	with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key) as cytomine:

		# Check that the given project exists
		if params.id_project:
			project = Project().fetch(params.id_project)
			if not project:
				raise ValueError("Project not found")

		# Get the ID of the Cytomine storage.
		storages = StorageCollection().fetch()
		my_storage = next(filter(lambda storage: storage.user == cytomine.current_user.id, storages))
		if not my_storage:
			raise ValueError("Storage not found")


		for sub_dir in tqdm.tqdm(glob.glob(params.dataset_path + "/*/"), desc='Uploading data'):
			for filepath in glob.glob(sub_dir + '*.tif'):
				uploaded_file = cytomine.upload_image(upload_host=params.upload_host,
													  filename=filepath,
													  id_storage=my_storage.id,
													  id_project=params.id_project)
				print(uploaded_file)