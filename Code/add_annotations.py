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
    #image_name = "PB13.0397-v-1.tif"

    annot_path = "D:\\Dataset_TFE\\Annot\\"

    side = 'd'

    if side == 'd':
        filename = "Concat_Aile_Ant_Gauche[261].TPS"
        coordinates = get_dict_annot_ant(annot_path, filename)
    else:
        filename = "Concat_Aile_Post_Gauche[260].TPS"
        coordinates = get_dict_annot_post(annot_path, filename)

    with Cytomine(host=host, public_key=public_key, private_key=private_key) as cytomine:
        
        image_instances = ImageInstanceCollection().fetch_with_filter("project", id_project)
        terms = TermCollection().fetch_with_filter("project", id_project)

        images_ids = {}
        for image in image_instances:
            if image.filename in images_ids.keys():
                images_ids[image.filename].append(image.id)
            else:
                images_ids[image.filename] = [image.id]

        #images_ids = {image.filename : image.id for image in image_instances}
        terms_ids = {term.name : term.id for term in terms}

        for image_name in images_ids.keys():
            if image_name not in coordinates.keys():
                continue

            for image_id in images_ids[image_name]:
                check_annot = AnnotationCollection()
                check_annot.image = image_id
                check_annot.fetch()

                if len(check_annot) == 0:
                    term_count = 1
                    annotations = AnnotationCollection()
                    for i in range(len(coordinates[image_name][0])):
                        for lm in coordinates[image_name][0][i]:
                            coord = tuple([float(lm[0]), float(lm[1])])
                            point = Point(coord)

                            if side == 'd':
                                term_name = str(term_count) + '-d-lm' if term_count <= 18 else str(term_count) + '-d-slm'
                            else:
                                term_name = str(term_count) + '-v-lm' if term_count <= 14 else str(term_count) + '-v-slm'
                            
                            annotation = Annotation(location=point.wkt, id_image=image_id, id_terms=terms_ids[term_name], id_project=id_project)
                            annotations.append(annotation)
                                            
                            term_count += 1

                    annotations.save()
                
