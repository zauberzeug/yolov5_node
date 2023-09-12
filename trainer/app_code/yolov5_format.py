import logging
import os
import shutil
from pathlib import Path
from typing import Dict

import yaml
from learning_loop_node.data_classes import Hyperparameter, Training
from ruamel.yaml import YAML


def category_lookup_from_training(training: Training) -> Dict:
    assert training.data is not None, 'Training should have data'
    return {c.name: c.id for c in training.data.categories}


def _create_set(training: Training, set_name: str) -> int:
    categories = list(category_lookup_from_training(training).values())
    training_path = training.training_folder
    images_path = f'{training_path}/{set_name}'

    shutil.rmtree(images_path, ignore_errors=True)
    os.makedirs(images_path, exist_ok=True)
    img_count = 0

    assert training.data is not None, 'Training should have data'
    logging.info(f'imagedata: {training.data.image_data}')
    for image in training.data.image_data:
        logging.info(f'processing image {image["set"]} - {image["set"] == set_name}')
        if image['set'] == set_name:
            img_count += 1
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
                c_id = str(categories.index(box['category_id']))
                yolo_boxes.append(c_id + ' ' + ' '.join([f"{c:.6f}" for c in coords]) + '\n')

            for point in image['point_annotations']:
                size = [c for c in training.data.categories if c.id == point['category_id']][0].point_size or 20
                coords = [
                    point['x']/width,
                    point['y']/height,
                    size/width,
                    size/height,
                ]
                c_id = str(categories.index(point['category_id']))
                yolo_boxes.append(c_id + ' ' + ' '.join([f"{c:.6f}" for c in coords]) + '\n')

            with open(f'{images_path}/{image["id"]}.txt', 'w') as l:
                l.writelines(yolo_boxes)

    return img_count


def _create_set_cla(training: Training, set_name: str):
    training_path = training.training_folder
    images_path = f'{training_path}/{set_name}'

    shutil.rmtree(images_path, ignore_errors=True)
    os.makedirs(images_path, exist_ok=True)
    for category in category_lookup_from_training(training).keys():
        os.makedirs(f'{images_path}/{category}', exist_ok=True)

    # classification format:
        # dataset
        # ├── train
        # │   ├── class1
        # │   │   ├── image1.jpg
        # |── test
        # │   ├── class1
        # │   │   ├── image2.jpg

    count = 0
    assert training.data is not None, 'Training should have data'
    for image in training.data.image_data:
        if image['set'] == set_name:
            image_name = image['id'] + '.jpg'
            classification = image['classification_annotation']
            if classification:
                count += 1
                category = classification['category_id']
                category_name = [c for c in training.data.categories if c.id == category][0].name
                image_path = f"{images_path}/{category_name}/{image_name}"
                # logging.info(f'linking {image_name} to {image_path}')
                os.symlink(f'{os.path.abspath(training.images_folder)}/{image_name}', image_path)
    logging.info(f'Created {count} image links')


def create_yaml(training: Training):
    categories = category_lookup_from_training(training)
    path = training.training_folder
    data = {
        'train': path + '/train',
        'test': path + '/test',
        'val': path + '/test',
        'nc': len(categories),
        'names': list(categories.keys())
    }
    logging.info(f'ordered names: {data["names"]}')
    with open(f'{path}/dataset.yaml', 'w') as f:
        yaml.dump(data, f)


def create_file_structure_cla(training: Training):
    path = training.training_folder
    assert path is not None, 'Training should have a path'
    Path(path).mkdir(parents=True, exist_ok=True)

    _create_set_cla(training, 'test')
    _create_set_cla(training, 'train')


def create_file_structure(training: Training):
    path = training.training_folder
    Path(path).mkdir(parents=True, exist_ok=True)

    num_test_imgs = _create_set(training, 'test')
    num_train_imgs = _create_set(training, 'train')
    create_yaml(training)

    logging.info(f'Prepared file structure with {num_train_imgs} training images and {num_test_imgs} test images')


def update_hyp(yaml_path: str, hyperparameter: Hyperparameter):
    yaml_ = YAML()

    with open(yaml_path) as f:
        content = yaml_.load(f)

    content['fliplr'] = 0.5 if hyperparameter.flip_rl else 0
    content['flipud'] = 0.5 if hyperparameter.flip_ud else 0

    with open(yaml_path, 'w') as f:
        yaml_.dump(content, f)
