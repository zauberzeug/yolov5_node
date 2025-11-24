import logging
import os
import shutil
from pathlib import Path
from typing import Any

from learning_loop_node.data_classes import Training
from learning_loop_node.trainer.exceptions import CriticalError
from ruamel.yaml import YAML
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarint import ScalarInt
from ruamel.yaml.scalarstring import LiteralScalarString

yaml = YAML()


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
                size = next(c for c in training.categories if c.id == point['category_id']).point_size or 20
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
    logging.info('ordered names: %s', data['names'])
    with open(f'{path}/dataset.yaml', 'w') as f:
        yaml.dump(data, f)


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

    logging.info('Prepared file structure with %d training images and %d test images', num_train_imgs, num_test_imgs)


def set_hyperparameters_in_file(yaml_path: str, hyperparameter: dict[str, Any]) -> None:
    """Override the hyperparameters in the yaml file used by the yolov5 trainer with the ones from the hyperparameter dict (coming from the loop configuration).
    The yaml file is modified in place."""

    with open(yaml_path) as f:
        content = yaml.load(f)

    if 'flip_rl' in hyperparameter:
        hyperparameter['fliplr'] = 0.5 if hyperparameter['flip_rl'] else 0.0
    if 'flip_lr' in hyperparameter:
        hyperparameter['fliplr'] = 0.5 if hyperparameter['flip_lr'] else 0.0
    if 'flip_ud' in hyperparameter:
        hyperparameter['flipud'] = 0.5 if hyperparameter['flip_ud'] else 0.0

    for param in content:
        if (hp_value := hyperparameter.get(param)) is not None:
            yaml_value = content[param]
            content[param] = convert_type(hp_value, yaml_value)

    logging.info('Hyps after update: %s', content)

    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(content, f)
    except Exception as e:
        logging.error('Error writing to %s: %s', yaml_path, e)
        raise CriticalError(f'Error writing to {yaml_path}') from None


def convert_type(value, reference: Any):

    if isinstance(reference, (LiteralScalarString, str)):
        return str(value)
    if isinstance(reference, (ScalarFloat, float)):
        return float(value)
    if isinstance(reference, (ScalarInt, int)):
        return int(value)
    if isinstance(reference, (ScalarBoolean, bool)):
        return bool(value)

    raise CriticalError(f'Unknown type: {type(reference)}')
