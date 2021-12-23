from learning_loop_node.trainer.training import Training
import yaml
from pathlib import Path
import shutil
import os


def create_set(training: Training, set_name: str):
    categories = [c['id'] for c in training.data.categories]
    training_path = training.training_folder
    images_path = f'{training_path}/{set_name}'
    os.makedirs(images_path, exist_ok=True)
    size = 20
    for image in training.data.image_data:
        if image['set'] == set_name:
            image_name = image['id'] + '.jpg'
            image_path = f"{images_path}/{image_name}"
            width = float(image['width'])
            height = float(image['height'])
            os.symlink(f'{os.path.abspath(training.images_folder)}/{image_name}', image_path)

            # box format: https://docs.ultralytics.com/tutorials/train-custom-datasets/
            # class x_center y_center width height
            # normalized coordinates
            yolo_boxes = []
            for box in image['box_annotations']:
                coords = [
                    (box['x'] + box['width'] / 2) / width,
                    (box['y'] + box['height'] / 2) / height,
                    box['width'] / width,
                    box['height'] / height,
                ]
                id = str(categories.index(box['category_id']))
                yolo_boxes.append(id + ' ' + ' '.join([str("%.6f" % c) for c in coords]) + '\n')
            for point in image['point_annotations']:
                coords = [
                    point['x']/width,
                    point['y']/height,
                    size/width,
                    size/height,
                ]
                id = str(categories.index(point['category_id']))
                yolo_boxes.append(id + ' ' + ' '.join([str("%.6f" % c) for c in coords]) + '\n')

            with open(f'{images_path}/{image["id"]}.txt', 'w') as l:
                l.writelines(yolo_boxes)


def create_yaml(training: Training):
    path = training.training_folder
    data = {
        'train': path + '/train',
        'test': path + '/test',
        'val': path + '/test',
        'nc': len(training.data.categories),
        'names': [c['name'] for c in training.data.categories],
    }

    with open(f'{path}/dataset.yaml', 'w') as f:
        yaml.dump(data, f)


def create_file_structure(training: Training):
    path = training.training_folder
    Path(path).mkdir(parents=True, exist_ok=True)

    create_set(training, 'test')
    create_set(training, 'train')
    create_yaml(training)
