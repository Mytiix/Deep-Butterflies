import logging

from shapely.geometry import Point

from cytomine import Cytomine
from cytomine.models import Property, Annotation, AnnotationTerm, AnnotationCollection, TermCollection
from cytomine.models.image import ImageInstanceCollection

from annotations_tps_ant_check import get_dict_annot_ant
from annotations_tps_post_check import get_dict_annot_post

logging.basicConfig()
logger = logging.getLogger("cytomine.client")
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    host = "https://research.cytomine.be" 
    public_key = "50af36d2-3ccc-4928-a21d-aab32e437fbf"
    private_key = "49725276-a001-4d25-9800-76409c8aad9a"
    id_project = 535588540
    image_name = "PB13.0823-d.tif"


    with Cytomine(host=host, public_key=public_key, private_key=private_key) as cytomine:
        
        image_instances = ImageInstanceCollection().fetch_with_filter("project", id_project)
        #terms = TermCollection().fetch_with_filter("project", id_project)

        images_ids = {image.filename : image.id for image in image_instances}

        annotations = AnnotationCollection()
        annotations.project = id_project
        annotations.image = images_ids[image_name]
        annotations.fetch()
        for annotation in annotations:
            annotation.delete()
        

