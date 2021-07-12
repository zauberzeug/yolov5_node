from learning_loop_node.trainer.training import Training
from imantics import Dataset, Category, Image, Annotation, BBox
from pathlib import Path
import json
import os


class BugfixedDataset(Dataset):

    def __init__(self, name):
        super().__init__(name)

        self.annotations = {}
        self.categories = {}
        self.images = {}


def create_dataset(training: Training, set: str):
    global id
    dataset = BugfixedDataset(f'COCO for {set}')
    categories = {}
    for c in training.data.box_categories:
        categories[c['id']] = Category(c['name'], id=c['id'], color=c['color'])

    for i, image in enumerate(training.data.image_data):
        if image['set'] != set:
            continue

        location = os.path.abspath(f'{training.images_folder}/{image["id"]}.jpg')
        coco_image = Image.from_path(location)
        coco_image.id = i
        dataset.add(coco_image)

        for a in image['box_annotations']:
            bbox = [a['x'], a['y'], a['width'], a['height']]
            coco_bbox = BBox(bbox, style=BBox.WIDTH_HEIGHT)
            coco_annotation = Annotation(coco_image, categories[a['category_id']], bbox=coco_bbox)
            dataset.add(coco_annotation)

    return dataset


def export_as_coco(training: Training):
    path = training.training_folder
    Path(path).mkdir(parents=True, exist_ok=True)

    with open(f'{path}/train.json', 'w') as f:
        json.dump(create_dataset(training, set='train').coco(), f)

    with open(f'{path}/test.json', 'w') as f:
        json.dump(create_dataset(training, set='test').coco(), f)
