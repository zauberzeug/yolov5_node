import glob
import json
import logging
import os
import time
from pathlib import Path
from time import sleep
from typing import Dict
from uuid import uuid4

import pytest
from learning_loop_node import Context
from learning_loop_node.data_classes.category import Category, CategoryType
from learning_loop_node.loop import loop
from learning_loop_node.model_information import ModelInformation
from learning_loop_node.tests import test_helper
from learning_loop_node.trainer import Executor, Training, TrainingsDownloader
from learning_loop_node.trainer.training_data import TrainingData

import yolov5_format
from yolov5_trainer import Yolov5Trainer


@pytest.mark.asyncio()
async def test_create_file_structure(use_training_dir):
    categories = [
        Category(name='classification_category_1', id='uuid_of_class_1'),
        Category(name='classification_category_2', id='uuid_of_class_2')]
    image_data = [{
        'set': 'train',
        'id': 'image_1',
        'width': 100,
        'height': 100,
        'box_annotations': [],
        'point_annotations': [],
        'classification_annotation': {
            'category_id': 'uuid_of_class_1',
        }
    },
        {
        'set': 'test',
        'id': 'image_2',
        'width': 100,
        'height': 100,
        'box_annotations': [],
        'point_annotations': [],
        'classification_annotation': {
            'category_id': 'uuid_of_class_2',
        }
    }]
    trainer = Yolov5Trainer()
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'), project_folder='./',
                                images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=image_data, categories=categories)

    yolov5_format.create_file_structure(trainer.training)

    assert Path('./train/classification_category_1/image_1.jpg').is_symlink()
    assert Path('./test/classification_category_2/image_2.jpg').is_symlink()


@pytest.mark.asyncio()
async def test_training_creates_model(use_training_dir):
    training = Training(
        id=str(uuid4()),
        project_folder=os.getcwd(),
        training_folder=os.getcwd() + '/training',
        images_folder=os.getcwd() + '/images',
        base_model='model.pt',
        context=Context(project='demo', organization='zauberzeug'),
    )
    training.data = await create_training_data(training)

    yolov5_format.create_file_structure(training)
    logging.info(training.training_folder)  # /tmp/test_training/
    logging.info([p for p in Path(f'{training.training_folder}/train/green classification/').iterdir()])
    executor = Executor(os.getcwd())
    # from https://github.com/WongKinYiu/yolor#training
    cmd = f'python /yolov5/classify/train.py --project training --name result --batch 4 --img 416 --data {training.training_folder} --model yolov5s-cls.pt --epochs 1'
    executor.start(cmd)
    while executor.is_process_running():
        sleep(1)
        logging.debug(executor.get_log())

    logging.info(executor.get_log())
    assert 'Training complete' in executor.get_log()
    assert 'best.pt' in executor.get_log()
    best = training.training_folder + '/result/weights/best.pt'
    assert os.path.isfile(best)


@pytest.mark.asyncio()
async def test_parse_progress_from_log(use_training_dir):
    trainer = Yolov5Trainer()
    trainer.epochs = 2
    trainer.training = Training(
        id=str(uuid4()),
        project_folder=os.getcwd(),
        training_folder=os.getcwd() + '/training',
        images_folder=os.getcwd() + '/images',
        base_model='model.pt',
        context=Context(project='demo', organization='zauberzeug'),
    )
    trainer.training.data = await create_training_data(trainer.training)
    yolov5_format.create_file_structure(trainer.training)

    trainer.executor = Executor(os.getcwd())
    cmd = f'python /yolov5/classify/train.py --project training --name result --batch 4 --img 416 --data {trainer.training.training_folder} --model yolov5s-cls.pt --epochs {trainer.epochs}'
    trainer.executor.start(cmd)
    while trainer.executor.is_process_running():
        sleep(1)

    logging.info(trainer.executor.get_log())
    assert f'{trainer.epochs}/{trainer.epochs}' in trainer.executor.get_log()
    assert trainer.progress == 1.0


def test_new_model_discovery(use_training_dir):
    trainer = Yolov5Trainer()
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'), project_folder='./',
                                images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=[], categories=[
                                         Category(name='class_a', id='uuid_of_class_a', type='classification')])
    assert trainer.get_new_model() is None, 'should not find any models'

    model_path = 'result/weights/published/latest.pt'
    mock_epoch({'class_a': {'fp': 0, 'tp': 1, 'fn': 0}})
    model = trainer.get_new_model()
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 1
    trainer.on_model_published(model)
    assert os.path.isfile(model_path)
    modification_date = os.path.getmtime(model_path)

    mock_epoch({'class_a': {'fp': 1, 'tp': 2, 'fn': 1}})
    model = trainer.get_new_model()
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 2
    trainer.on_model_published(model)

    assert trainer.get_new_model() is None, 'again we should not find any new models'

    time.sleep(0.1)

    mock_epoch({'class_a': {'fp': 0, 'tp': 3, 'fn': 1}})
    model = trainer.get_new_model()
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 3
    trainer.on_model_published(model)
    assert os.path.isfile(model_path)
    new_modification_date = os.path.getmtime(model_path)
    assert new_modification_date > modification_date

    # get_latest_model_file
    files = trainer.get_latest_model_files()
    assert files == {'yolov5_pytorch': ['/tmp/model.pt', './/result/opt.yaml']}


def test_old_model_files_are_deleted_on_publish(use_training_dir):
    trainer = Yolov5Trainer()
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'),
                                project_folder='./', images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=[], categories=[
                                         Category(name='class_a', id='uuid_of_class_a', type='classification')])
    assert trainer.get_new_model() is None, 'should not find any models'

    mock_epoch({'class_a': {'fp': 0, 'tp': 1, 'fn': 0}})
    model = trainer.get_new_model()
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 1
    mock_epoch({'class_a': {'fp': 1, 'tp': 2, 'fn': 1}})

    _, _, files = next(os.walk("result/weights"))
    assert len(files) == 2

    model = trainer.get_new_model()
    trainer.on_model_published(model)
    _, _, files = next(os.walk("result/weights/published"))
    assert len(files) == 1
    assert os.path.isfile('result/weights/published/latest.pt')

    _, _, files = next(os.walk("result/weights"))
    assert len(files) == 0


@pytest.mark.asyncio()
async def test_clear_training_data():
    trainer = Yolov5Trainer()
    os.makedirs('/data/o/p/trainings/some_uuid', exist_ok=True)
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'), project_folder='./',
                                images_folder='./', training_folder='/data/o/p/trainings/some_uuid')
    os.makedirs(f'{trainer.training.training_folder}/result/weights/', exist_ok=True)
    os.makedirs(f'{trainer.training.training_folder}/result/weights/published/', exist_ok=True)

    # Create needed .pt files and some dummy data
    with open(f'{trainer.training.training_folder}/result/weights/published/best.pt', 'wb') as f:
        f.write(b'0')
    with open(f'{trainer.training.training_folder}/result/weights/best.pt', 'wb') as f:
        f.write(b'0')
    with open(f'{trainer.training.training_folder}/result/weights/last.pt', 'wb') as f:
        f.write(b'0')
    with open(f'{trainer.training.training_folder}/last_training.log', 'wb') as f:
        f.write(b'0')

    data = glob.glob(trainer.training.training_folder + '/**', recursive=True)
    assert len(data) == 8
    files = [f for f in data if os.path.isfile(f)]
    assert len(files) == 4

    await trainer.clear_training_data(trainer.training.training_folder)
    data = glob.glob(trainer.training.training_folder + '/**', recursive=True)
    assert len(data) == 5
    files = [f for f in data if os.path.isfile(f)]
    print(files)
    assert len(files) == 2  # Note: Do not delete last_training.log und last.pt


@pytest.fixture()
def create_project():
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'classification_categories': 2, 'inbox': 0,
                             'annotate': 0, 'review': 0, 'complete': 0, 'image_style': 'plain', 'thumbs': False, 'trainings': 1}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


def test_parse_detections_file():
    os.makedirs('/tmp/results', exist_ok=True)
    with open(f'/tmp/results/detection.txt', 'w') as f:
        f.writelines(['1.00 Thripse\n', '0.00 Kastanienminiermotte\n',
                     '0.00 Johannisbeerblasenlaus\n', '0.00 Blattlaeuse\n', '0.00 Wolllaeuse'])
    model_information = ModelInformation(id='someid', organization='o', project='p', version='1.1', categories=[
                                         Category(id='some_id', name='Thripse', type=CategoryType.Classification)], resolution=320)
    file = os.scandir('/tmp/results').__next__()
    detections = Yolov5Trainer._parse_file(model_information=model_information, filename=file)
    assert len(detections) > 0
    os.remove('/tmp/results/detection.txt')


async def create_training_data(training: Training) -> TrainingData:
    training_data = TrainingData()

    image_data, _ = await TrainingsDownloader(training.context).download_training_data(training.images_folder)
    response = test_helper.LiveServerSession().get(f"/api/zauberzeug/projects/demo/data")
    assert response.status_code == 200
    data = response.json()
    training_data.categories = Category.from_list(
        [category for category in data['categories'] if category['type'] == 'classification'])
    training_data.image_data = image_data
    return training_data


def mock_epoch(confusion_matrix: Dict):
    os.makedirs('result/weights/', exist_ok=True)
    with open(f'result/weights/best.json', 'w') as f:
        json.dump(confusion_matrix, f)
    with open(f'result/weights/best.pt', 'wb') as f:
        f.write(b'0')
