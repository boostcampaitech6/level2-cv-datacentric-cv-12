# -*- coding:utf-8 -*-
from typing import Dict
import os
import json
import datetime


now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d %H:%M:%S')

input_path = './data/medical/ai_hub/'
output_path = './data/medical/ai_hub_coco/bank_coco.json'

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

def sort_polygon(points):

    output = []
    sorted_points = sorted(points, key=lambda x: -x[1])
    output.append(sorted_points.pop())
    output.append(sorted_points.pop())
    output.sort(key=lambda x: x[0])

    sorted_points = sorted(sorted_points, key=lambda x: (x[0]))
    output.append(sorted_points.pop())
    output.append(sorted_points.pop())
    return output

def hub_to_coco(file: Dict, annotation: Dict, img_id, annotation_id) -> None:

    image = {
        "id": img_id,
        "width": file['width'],
        "height": file['height'],
        "file_name": file["name"],
        "license": 1,
        # "flickr_url": None,
        # "coco_url": None,
        "date_captured": now
    }
    images.append(image)

    for ann in annotation:
        if ann["type"] == 1: # 인쇄체만

            coco_annotation = {
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": [[value for sublist in sort_polygon(ann["points"]) for value in sublist]],
                "iscrowd": 0,
                "attributes":{
                    "type": ann["type"],
                    }
            }
            annotations.append(coco_annotation)
            annotation_id += 1
    img_id += 1

    return images, annotation, img_id, annotation_id


files = os.listdir(input_path)

 #COCO는 1부터
img_id = 1 
annotation_id = 1

images = []
annotations = []

for file in files:
    with open(f"{input_path}{file}", 'r') as f:
        json_file = json.load(f)
    images, annotation, img_id, annotation_id = hub_to_coco(json_file['images'][0], json_file["annotations"][0]["polygons"], img_id, annotation_id)

coco = {
    'info' : info,
    'images' : images,
    'annotations' : annotations,
    'licenses' : licenses,
    'categories' : categories
}

with open(output_path, 'w') as f:
    json.dump(coco, f, indent=4)