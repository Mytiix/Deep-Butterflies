import logging

from cytomine import Cytomine
from cytomine.models import Property, ImageInstance
from cytomine.models.image import ImageInstanceCollection

logging.basicConfig()
logger = logging.getLogger("cytomine.client")
logger.setLevel(logging.INFO)


def get_dict_sp():
    annot_path = "D:\\Dataset_TFE\\Annot\\"

    filename1 = "ClassifierAA_2021[258].txt"
    filename2 = "ClassifierAP_2021[259].txt"

    lines = []
    with open(annot_path + filename1) as f1:
        lines = f1.readlines()[1:]

    with open(annot_path + filename2) as f2:
        lines += f2.readlines()[1:]


    sp = {}
    check = {}
    for line in lines:
        # [ID, Individu, sexe, genre, sp]
        split = line.split()

        if split[1] not in check.keys():
            check[split[1]] = split[4]
        else:
            if check[split[1]] != split[4]:
                print(split[1])
            break

        if split[1] not in sp.keys():
            sp[split[1]] = split[4].lower()


    for i in range(1, 946):
        name = 'PB13-' + str(i).zfill(4)
        if name not in sp.keys():
            sp[name] = 'unknown'


    return sp



if __name__ == '__main__':
    host = "https://research.cytomine.be" 
    public_key = "50af36d2-3ccc-4928-a21d-aab32e437fbf"
    private_key = "49725276-a001-4d25-9800-76409c8aad9a"
    id_project = 535588540
    #image_name = "PB13.0397-v-1.tif"

    annot_path = "D:\\Dataset_TFE\\Annot\\"

    key = 'specie'

    sp = get_dict_sp()

    with Cytomine(host=host, public_key=public_key, private_key=private_key) as cytomine:
        
        image_instances = ImageInstanceCollection().fetch_with_filter("project", id_project)

        images_ids = {}
        for image in image_instances:
            if image.filename in images_ids.keys():
                images_ids[image.filename].append(image.id)
            else:
                images_ids[image.filename] = [image.id]


        for image_name in images_ids.keys():
            name = image_name[:9].replace('.', '-')
            for image_id in images_ids[image_name]:
                image = ImageInstance().fetch(image_id)
                prop = Property(image, key=key, value=sp[name]).save()
            