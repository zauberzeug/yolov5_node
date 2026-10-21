import logging
import os
import shutil
from pathlib import Path
from typing import Any

from learning_loop_node.data_classes import Training
from learning_loop_node.trainer.exceptions import CriticalError
from ruamel.yaml import YAML
from ruamel.yaml.representer import RoundTripRepresenter


class _Representer(RoundTripRepresenter):
    """Writes floats the way PyYAML does, since ``train_det.py`` reads the hyp.yaml with it.

    PyYAML follows YAML 1.1, which reads ``5e-05`` as a string and only ``5.0e-05`` as a number.
    """

    def represent_float(self, data: float) -> Any:
        text = repr(data).lower()
        if '.' not in text and 'e' in text:
            text = text.replace('e', '.0e', 1)
        return self.represent_scalar('tag:yaml.org,2002:float', text)


_Representer.add_representer(float, _Representer.represent_float)

yaml = YAML()
yaml.Representer = _Representer


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


FLIP_PROBABILITY = 0.5
"""How often a flip is applied when it is on: half of the images, so flipped and original views are equally likely."""


def set_hyperparameters_in_file(yaml_path: str, hyperparameter: dict[str, Any]) -> None:
    """Override the hyperparameters in the yaml file used by the yolov5 trainer with the training's resolved ones.
    The yaml file is modified in place.

    Every key of the template is declared with the template's type, so the resolved values need no conversion.
    """

    with open(yaml_path) as f:
        content = yaml.load(f)

    for param in content:
        if param in hyperparameter:
            content[param] = hyperparameter[param]
    for flip in ('fliplr', 'flipud'):  # the training keeps the declared on or off, the yaml gets a probability
        content[flip] = FLIP_PROBABILITY if content[flip] else 0.0

    logging.info('Hyps after update: %s', content)

    try:
        with open(yaml_path, 'w') as f:
            yaml.dump(content, f)
    except Exception as e:
        logging.error('Error writing to %s: %s', yaml_path, e)
        raise CriticalError(f'Error writing to {yaml_path}') from None
