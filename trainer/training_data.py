from learning_loop_node.trainer.training import Training
import yaml
from pathlib import Path
import shutil
import os


def create_set(training: Training, set_name: str):
    path = training.training_folder
    images_path = f'{path}/images'
    os.makedirs(images_path, exist_ok=True)
    with open(f'{path}/{set_name}.txt', 'w') as f:
        for image in training.data.image_data:
            if image['set'] == set_name:
                image_name = image['id'] + '.jpg'
                image_path = f"{images_path}/{image_name}"
                f.write(f"{image_path}\n")
                os.symlink(f'{os.path.abspath(training.images_folder)}/{image_name}', image_path)


def create_yaml(training: Training):
    path = training.training_folder
    data = {
        'train': path + '/train.txt',
        'test': path + '/test.txt',
        'val': path + '/test.txt',
        'nc': len(training.data.box_categories),
        'names': [c['name'] for c in training.data.box_categories],
    }

    with open(f'{path}/dataset.yaml', 'w') as f:
        yaml.dump(data, f)


def export(training: Training):
    path = training.training_folder
    Path(path).mkdir(parents=True, exist_ok=True)

    create_set(training, 'test')
    create_set(training, 'train')
    create_yaml(training)
