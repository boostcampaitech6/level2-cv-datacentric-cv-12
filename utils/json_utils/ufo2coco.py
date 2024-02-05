# -*- coding:utf-8 -*-
from typing import Dict
import json
import datetime


now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d %H:%M:%S')

input_path = './data/medical/output_json/train_instances_default_ufo.json'
output_path = './data/medical/output_json/train_instances_default_coco.json'

info = {
    'year': 2024,
    'version': '1.0',
    'description': 'OCR Competition Data',
    'contributor': 'Naver Boostcamp',
    'url': 'https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000273/data/data.tar.gz',
    'date_created': now
}
licenses = {
    'id': '1',
    'name': 'For Naver Boostcamp Competition',
    'url': None
}
categories = [{
    'id': 1,
    'name': 'word'
}]

def ufo_to_coco(file: Dict, output_path: str) -> None:
    img_id = 1 
    img_id_anno = 1 #COCO는 1부터
    annotation_id = 1 #COCO는 1부터
    images = []
    annotations = []

    for fname, data in file.items():
        if img_id <= 60:
            image = {
                "id": img_id_anno,
                "width": data['img_w'],
                "height": data['img_h'],
                "file_name": fname,
                "license": 1,
                # "flickr_url": None,
                # "coco_url": None,
                "date_captured": now
            }
            images.append(image)
            for anno_id, annotation in data['words'].items():
                if annotation['illegibility'] == True:
                    continue
                min_x = min(item[0] for item in annotation['points'])
                min_y = min(item[1] for item in annotation['points'])
                max_x = max(item[0] for item in annotation['points'])
                max_y = max(item[1] for item in annotation['points'])
                width = max_x - min_x
                height = max_y - min_y
                coco_annotation = {
                    "id": annotation_id,
                    "image_id": img_id_anno,
                    "category_id": 1,
                    "segmentation": [[value for sublist in annotation['points'] for value in sublist]],
                    # "area": width * height,
                    # "bbox": [min_x, min_y, width, height],
                    "iscrowd": 0,
                    "attributes":{
                        # "transcription": annotation["transcription"],
                        #  "orientation":annotation["orientation"],
                        #  "language": annotation["language"],
                        "tags": annotation["tags"],
                        #  "confidence": annotation["confidence"],
                        #  "illegibility": annotation["illegibility"]
                        }
                }
                annotations.append(coco_annotation)
                annotation_id += 1
            img_id_anno += 1
        img_id += 1
        
    coco = {
        'info' : info,
        'images' : images,
        'annotations' : annotations,
        'licenses' : licenses,
        'categories' : categories
    }
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=4)

with open(input_path, 'r') as f:
    file = json.load(f)
ufo_to_coco(file['images'], output_path)