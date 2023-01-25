from learning_loop_node.trainer.training import Training
import yaml
from pathlib import Path
import logging
import os
from learning_loop_node.trainer.hyperparameter import Hyperparameter
import shutil


def category_lookup_from_training(training: Training) -> dict:
    return {c.name: c.id for c in training.data.categories}


def create_set(training: Training, set_name: str):
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
    for image in training.data.image_data:
        if image['set'] == set_name:
            image_name = image['id'] + '.jpg'
            classification = image['classification_annotation']
            if classification:
                category = classification['category_id']
                category_name = [c for c in training.data.categories if c.id == category][0].name
                logging.error(f'category_name: {category_name}')
                image_path = f"{images_path}/{category_name}/{image_name}"
                logging.info('image_path: ' + image_path)
                logging.info(f'linking {image_name} to {image_path}')
                os.symlink(f'{os.path.abspath(training.images_folder)}/{image_name}', image_path)


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


def create_file_structure(training: Training):
    path = training.training_folder
    Path(path).mkdir(parents=True, exist_ok=True)

    create_set(training, 'test')
    create_set(training, 'train')
    create_yaml(training)


def update_hyp(yaml_path: str, hyperparameter: Hyperparameter):
    from ruamel.yaml import YAML
    yaml = YAML()

    with open(yaml_path) as f:
        content = yaml.load(f)

    content['fliplr'] = 0.5 if hyperparameter.flip_rl else 0
    content['flipud'] = 0.00856 if hyperparameter.flip_ud else 0
    with open(yaml_path, 'w') as f:
        yaml.dump(content, f)
