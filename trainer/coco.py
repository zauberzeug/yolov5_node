from learning_loop_node.trainer.training import Training
from imantics import Dataset, Category, Image, Annotation, BBox
import json


def create_dataset(training: Training):
    dataset = Dataset('COCO')
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

    return dataset


def export_as_coco(training: Training, path: str):
    dataset = create_dataset(training)

    with open(path, 'w') as f:
        json.dump(dataset.coco(), f)
