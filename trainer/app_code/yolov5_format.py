import logging
import os
import shutil
from pathlib import Path
from typing import Any

from learning_loop_node.data_classes import Training
from learning_loop_node.trainer.exceptions import CriticalError
from learning_loop_node.trainer.hyperparameters import FLIP_ALIASES, merge_into_yaml
from ruamel.yaml import YAML

from .hyperparameters import NAMES as HYPERPARAMETER_NAMES

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

    if num_train_imgs == 0:
        raise CriticalError('No training images found. Cannot start training without images in the train set.')
    if num_test_imgs == 0:
        raise CriticalError('No validation images found. Training requires at least 1 image in the test set.')

    create_dataset_yaml(training)

    logging.info('Prepared file structure with %d training images and %d test images', num_train_imgs, num_test_imgs)


def set_hyperparameters_in_file(yaml_path: str, hyperparameter: dict[str, Any]) -> None:
    """Write the loop's hyperparameters into the yolov5 trainer's yaml template, in place.

    Only the template's own keys are written -- they are the ones the vendored training reads.
    Anything the loop configured that neither the template nor this node's declaration covers is
    named in the log rather than dropped in silence.
    """
    hyperparameter = dict(hyperparameter)
    for name, yaml_name in FLIP_ALIASES.items():
        if name in hyperparameter:  # the loop sends a flag; yolov5 wants a probability
            hyperparameter[yaml_name] = 0.5 if hyperparameter[name] else 0.0

    merge_into_yaml(yaml_path, hyperparameter,
                    ignore=(*HYPERPARAMETER_NAMES, *FLIP_ALIASES))
