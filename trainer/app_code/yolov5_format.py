import logging
import os
import shutil
from pathlib import Path
from typing import Any

from learning_loop_node.data_classes import Training
from learning_loop_node.enums import CategoryType
from learning_loop_node.trainer.exceptions import CriticalError
from ruamel.yaml import YAML

yaml = YAML()


def get_ids_and_sizes_of_point_classes(training: Training) -> tuple[list[str], list[str]]:
    """Returns a list of trainingids and sizes (in px) of point classes in the training data."""
    assert training is not None, 'Training should have data'
    point_ids, point_sizes = [], []
    for i, category in enumerate(training.categories):
        if category.type == CategoryType.Point:
            point_ids.append(str(i))
            point_sizes.append(str(category.point_size or 20))
    return point_ids, point_sizes


def category_lookup_from_training(training: Training) -> dict[str, str]:
    return {c.name: c.id for c in training.categories}


def _create_set(training: Training, set_name: str) -> int:
    """Create training folder structure for a set (train or test).
    - Images in the set are linked from the images folder (symlinks)
    - Annotations are created in the set folder
    Annotations are boxes in the format:
    "class(id) x_center y_center width height" (normalized by image width and height)
    Note that the id here is not the uuid but the training id (0, 1, 2, ...).
    [see here](https://docs.ultralytics.com/tutorials/train-custom-datasets/)."""

    category_uuids = list(category_lookup_from_training(training).values())

    training_path = training.training_folder
    images_path = f'{training_path}/{set_name}'

    shutil.rmtree(images_path, ignore_errors=True)
    os.makedirs(images_path, exist_ok=True)
    img_count = 0

    for image in training.image_data or []:
        if image['set'] == set_name:
            img_count += 1
            image_name = image['id'] + '.jpg'
            image_path = f"{images_path}/{image_name}"
            width = float(image['width'])
            height = float(image['height'])
            os.symlink(f'{os.path.abspath(training.images_folder)}/{image_name}', image_path)

            # Create annotation file
            yolo_boxes = []
            for box in image['box_annotations']:
                coords = [
                    (box['x'] + box['width'] / 2) / width,
                    (box['y'] + box['height'] / 2) / height,
                    box['width'] / width,
                    box['height'] / height,
                ]
                c_id = str(category_uuids.index(box['category_id']))
                yolo_boxes.append(c_id + ' ' + ' '.join([f"{c:.6f}" for c in coords]) + '\n')

            for point in image['point_annotations']:
                size = [c for c in training.categories if c.id == point['category_id']][0].point_size or 20
                coords = [
                    point['x']/width,
                    point['y']/height,
                    size/width,
                    size/height,
                ]
                c_id = str(category_uuids.index(point['category_id']))
                yolo_boxes.append(c_id + ' ' + ' '.join([f"{c:.6f}" for c in coords]) + '\n')

            with open(f'{images_path}/{image["id"]}.txt', 'w') as l:
                l.writelines(yolo_boxes)

    return img_count


def _create_set_cla(training: Training, set_name: str) -> None:
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
    assert training.image_data is not None, 'Training should have image data'
    for image in training.image_data:
        if image['set'] == set_name:
            image_name = image['id'] + '.jpg'
            classification = image['classification_annotation']
            if classification:
                count += 1
                category = classification['category_id']
                category_name = [c for c in training.categories if c.id == category][0].name
                image_path = f"{images_path}/{category_name}/{image_name}"
                # logging.info(f'linking {image_name} to {image_path}')
                os.symlink(f'{os.path.abspath(training.images_folder)}/{image_name}', image_path)

    logging.info(f'Created {count} image links')


def create_dataset_yaml(training: Training) -> None:
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


def create_file_structure_cla(training: Training) -> None:
    path = training.training_folder
    assert path is not None, 'Training should have a path'
    Path(path).mkdir(parents=True, exist_ok=True)

    _create_set_cla(training, 'test')
    _create_set_cla(training, 'train')


def create_file_structure(training: Training) -> None:
    """Uses:
    - training.training_folder to create the file structure.
    - training.image_data to create the image links and annotations.
    - training.categories to create the annotations."""
    path = training.training_folder
    Path(path).mkdir(parents=True, exist_ok=True)

    num_test_imgs = _create_set(training, 'test')
    num_train_imgs = _create_set(training, 'train')
    create_dataset_yaml(training)

    logging.info(f'Prepared file structure with {num_train_imgs} training images and {num_test_imgs} test images')


def set_hyperparameters_in_file(yaml_path: str, hyperparameter: dict[str, Any]) -> None:

    with open(yaml_path) as f:
        content = yaml.load(f)

    if 'flip_rl' in hyperparameter:
        hyperparameter['fliplr'] = 0.5 if hyperparameter['flip_rl'] else 0.0
    if 'flip_lr' in hyperparameter:
        hyperparameter['fliplr'] = 0.5 if hyperparameter['flip_lr'] else 0.0
    if 'flip_ud' in hyperparameter:
        hyperparameter['flipud'] = 0.5 if hyperparameter['flip_ud'] else 0.0

    for param in content:
        if param in hyperparameter:
            yaml_value = content[param]
            hp_value = hyperparameter[param]

            try:
                # Try to convert to float as all original yolov5 hyperparameters are floats
                hp_value = float(hp_value)
            except (ValueError, TypeError) as e:
                raise CriticalError(f'Parameter {param} cannot be converted from {type(hp_value)} to float') from e

            logging.info(f'Setting hyperparameter {param} to {hp_value} (Default was {yaml_value})')
            content[param] = hp_value

    with open(yaml_path, 'w') as f:
        yaml.dump(content, f)
