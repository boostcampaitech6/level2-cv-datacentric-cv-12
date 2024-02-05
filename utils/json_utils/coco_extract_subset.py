import os 
import zipfile
from pylabel import importer

#Specify path to the coco.json file
path_to_annotations = "/data/ephemeral/home/data/medical/ufo/train_coco.json"
#Specify the path to the images (if they are in a different folder than the annotations)
path_to_images = ""

#Import the dataset into the pylable schema 
dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="BCCD_coco")

dataset.path_to_annotations = 'data/coco/'
dataset.name = 'coco'

dataset.export.ExportToCoco()