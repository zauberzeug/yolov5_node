from learning_loop_node.trainer.training import Training
import pytest
from imantics import Dataset, Category, Image, Annotation, BBox
from icecream import ic
import json


def test_creating_coco_from_training_object(use_test_dir):
    training = Training.parse_file('example_training.json')

    dataset = Dataset('example')
    categories = {}
    assert len(training.data.box_categories) == 2
    for c in training.data.box_categories:
        categories[c['id']] = Category(c['name'], id=c['id'], color=c['color'])

    assert len(training.data.image_data) == 4
    for i, image in enumerate(training.data.image_data):
        location = f'{training.images_folder}/{image["id"]}.jpg'
        coco_image = Image.from_path(location)
        coco_image.id = i
        dataset.add(coco_image)

        for a in image['box_annotations']:
            bbox = [a['x'], a['y'], a['width'], a['height']]
            coco_bbox = BBox(bbox, style=BBox.WIDTH_HEIGHT)
            coco_annotation = Annotation(coco_image, categories[a['category_id']], bbox=coco_bbox)
            dataset.add(coco_annotation)

    coco = dataset.coco()

    #print(json.dumps(coco, indent=2), flush=True)
    assert len(coco['categories']) == 2
    assert len(coco['images']) == 4
    assert len(coco['annotations']) == 4*3
