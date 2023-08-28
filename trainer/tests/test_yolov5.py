import glob
import json
import logging
import os
import time
from time import sleep
from typing import Dict
from uuid import uuid4

import pytest
from dotenv import load_dotenv
from learning_loop_node import Context
from learning_loop_node.data_classes.category import Category
from learning_loop_node.gdrive_downloader import g_download
from learning_loop_node.loop import loop
from learning_loop_node.tests import test_helper
from learning_loop_node.trainer import (Executor, Trainer, Training,
                                        TrainingsDownloader)
from learning_loop_node.trainer.training_data import TrainingData
from pydantic.types import Json

import model_files
import yolov5_format
from yolov5_trainer import Yolov5Trainer

load_dotenv()
print(f'loading .env from {os.getcwd()}')


@pytest.mark.asyncio()
async def test_create_file_structure_box_size(use_training_dir):
    categories = [
        Category(name='point_category_1', id='uuid_of_class_1'),
        Category(name='point_category_2', id='uuid_of_class_2', point_size=30)]
    image_data = [{
        'set': 'train',
        'id': 'image_1',
        'width': 100,
        'height': 100,
        'box_annotations': [],
        'point_annotations': [{
            'category_id': 'uuid_of_class_1',
            'x': 50,
            'y': 60,
        }, {
            'category_id': 'uuid_of_class_2',
            'x': 60,
            'y': 70,
        }]
    }]
    trainer = Yolov5Trainer()
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'), project_folder='./',
                                images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=image_data, categories=categories)

    yolov5_format.create_file_structure(trainer.training)

    with open('./train/image_1.txt', 'r') as f:
        lines = f.readlines()

    assert '0 0.500000 0.600000 0.200000 0.200000' in lines[0]
    assert '1 0.600000 0.700000 0.300000 0.300000' in lines[1]


@pytest.mark.asyncio()
async def test_training_creates_model(use_training_dir):  # FLAKY! model.pt can sometimes not be accessed
    os.remove('/tmp/model.pt')
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

    executor = Executor(os.getcwd())
    # from https://github.com/WongKinYiu/yolor#training
    cmd = f'WANDB_MODE=disabled python /yolov5/train.py --project training --name result --batch 4 --img 416 --data training/dataset.yaml --weights model.pt --epochs 1'
    executor.start(cmd)
    while executor.is_process_running():
        sleep(1)
        logging.debug(executor.get_log())

    logging.info(executor.get_log())
    assert '1 epochs completed' in executor.get_log()
    assert 'best.pt' in executor.get_log()
    best = training.training_folder + '/result/weights/best.pt'
    assert os.path.isfile(best)


@pytest.mark.asyncio()
async def test_parse_progress_from_log(use_training_dir):  # FLAKY! model.pt can sometimes not be accessed
    trainer = Yolov5Trainer()
    trainer.epochs = 2
    os.remove('/tmp/model.pt')
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
    cmd = f'WANDB_MODE=disabled python /yolov5/train.py --project training --name result --batch 4 --img 416 --data training/dataset.yaml --weights model.pt --epochs {trainer.epochs}'
    trainer.executor.start(cmd)
    while trainer.executor.is_process_running():
        sleep(1)

    logging.info(trainer.executor.get_log())
    assert f'{trainer.epochs} epochs completed' in trainer.executor.get_log()
    assert trainer.progress == 1.0


def test_new_model_discovery(use_training_dir):
    trainer = Yolov5Trainer()
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'), project_folder='./',
                                images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=[], categories=[
                                         Category(name='class_a', id='uuid_of_class_a', type='box')])
    assert trainer.get_new_model() is None, 'should not find any models'

    model_path = 'result/weights/published/latest.pt'
    mock_epoch(1, {'class_a': {'fp': 0, 'tp': 1, 'fn': 0}})
    model = trainer.get_new_model()
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 1
    trainer.on_model_published(model)
    assert os.path.isfile(model_path)
    modification_date = os.path.getmtime(model_path)

    mock_epoch(2, {'class_a': {'fp': 1, 'tp': 2, 'fn': 1}})
    model = trainer.get_new_model()
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 2
    trainer.on_model_published(model)

    assert trainer.get_new_model() is None, 'again we should not find any new models'

    time.sleep(0.1)

    mock_epoch(3, {'class_a': {'fp': 0, 'tp': 3, 'fn': 1}})
    model = trainer.get_new_model()
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 3
    trainer.on_model_published(model)
    assert os.path.isfile(model_path)
    new_modification_date = os.path.getmtime(model_path)
    assert new_modification_date > modification_date

    # get_latest_model_file
    files = trainer.get_latest_model_files()
    assert files == {'yolov5_pytorch': ['/tmp/model.pt', './/hyp.yaml'], 'yolov5_wts': ['/tmp/model.wts']}


def test_newest_model_is_used(use_training_dir):
    trainer = Yolov5Trainer()
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'), project_folder='./',
                                images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=[], categories=[
                                         Category(name='class_a', id='uuid_of_class_a', type='box')])

    # create some models.
    mock_epoch(10, {})
    mock_epoch(200, {})

    new_model = trainer.get_new_model()
    assert 'epoch10.pt' not in new_model.meta_information['weightfile']
    assert 'epoch200.pt' in new_model.meta_information['weightfile']


def test_old_model_files_are_deleted_on_publish(use_training_dir):
    trainer = Yolov5Trainer()
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'),
                                project_folder='./', images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=[], categories=[
                                         Category(name='class_a', id='uuid_of_class_a', type='box')])
    assert trainer.get_new_model() is None, 'should not find any models'

    mock_epoch(1, {'class_a': {'fp': 0, 'tp': 1, 'fn': 0}})
    model = trainer.get_new_model()
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 1
    mock_epoch(2, {'class_a': {'fp': 1, 'tp': 2, 'fn': 1}})

    _, _, files = next(os.walk("result/weights"))
    assert len(files) == 4

    model = trainer.get_new_model()
    trainer.on_model_published(model)
    _, _, files = next(os.walk("result/weights/published"))
    assert len(files) == 1
    assert os.path.isfile('result/weights/published/latest.pt')

    _, _, files = next(os.walk("result/weights"))
    assert len(files) == 0


def test_newer_model_files_are_kept_during_deleting(use_training_dir):
    trainer = Yolov5Trainer()
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'), project_folder='./',
                                images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=[], categories=[
                                         Category(name='class_a', id='uuid_of_class_a', type='box')])

    # create some models.
    mock_epoch(10, {})
    mock_epoch(200, {})
    new_model = trainer.get_new_model()
    assert 'epoch200.pt' in new_model.meta_information['weightfile']
    mock_epoch(201, {})  # An epoch is finished after during communication with the LearningLoop

    trainer.on_model_published(new_model)

    all_model_files = model_files.get_all(trainer.training.training_folder)
    assert len(all_model_files) == 1
    assert 'epoch201.pt' in all_model_files[0], 'Epoch201 is not yed synced. It should not be deleted.'


@pytest.mark.asyncio()
async def test_clear_training_data():
    trainer = Yolov5Trainer()
    os.makedirs('/data/o/p/trainings/some_uuid', exist_ok=True)
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'), project_folder='./',
                                images_folder='./', training_folder='/data/o/p/trainings/some_uuid')
    os.makedirs(f'{trainer.training.training_folder}/result/weights/', exist_ok=True)
    os.makedirs(f'{trainer.training.training_folder}/result/weights/published/', exist_ok=True)

    # Create needed .pt files and some dummy data
    with open(f'{trainer.training.training_folder}/result/weights/published/some_model_id.pt', 'wb') as f:
        f.write(b'0')
    with open(f'{trainer.training.training_folder}/result/weights/test.txt', 'wb') as f:
        f.write(b'0')
    with open(f'{trainer.training.training_folder}/result/weights/best.pt', 'wb') as f:
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
    assert len(files) == 2  # Note: Do not delete last_training.log und best.pt


@pytest.fixture()
def create_project():
    test_helper.LiveServerSession().delete(f"/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {
        'project_name': 'pytest', 'box_categories': 2, 'point_categories': 1, 'inbox': 0, 'annotate': 0, 'review': 0,
        'complete': 0, 'image_style': 'plain', 'thumbs': False, 'trainings': 1}
    assert test_helper.LiveServerSession().post(f"/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/zauberzeug/projects/pytest?keep_images=true")


# @pytest.mark.asyncio()
# async def test_detecting(create_project): # TODO: Fix this test (results in file not found on server)
#     # from learning_loop_node.loop import Loop
#     # loop = Loop()
#     logging.debug('downloading model from gdrive')

#     file_id = '1sZWa053fWT9PodrujDX90psmjhFVLyBV'
#     destination = '/tmp/model.zip'
#     g_download(file_id, destination)

#     test_helper.unzip(destination, '/tmp/model')

#     logging.debug('uploading model')
#     data = ['/tmp/model/model.pt', '/tmp/model/model.json']
#     response = await loop.put(f'api/zauberzeug/projects/pytest/1/models/latest/yolov5_pytorch/file', files=data)
#     if response.status != 200:
#         msg = f'unexpected status code {response.status} while putting model'
#         logging.error(msg)
#         raise (Exception(msg))
#     model = await response.json()

#     data = test_helper.prepare_formdata(['tests/example_images/8647fc30-c46c-4d13-a3fd-ead3b9a67652.jpg'])
#     response = await loop.post(f'api/zauberzeug/projects/pytest/images', files=data)
#     if response.status != 200:
#         msg = f'unexpected status code {response.status} while posting a new image'
#         logging.error(msg)
#         raise (Exception(msg))
#     image = await response.json()

#     trainer = Yolov5Trainer()
#     context = Context(organization='zauberzeug', project='pytest')
#     trainer.training = Trainer.generate_training(context)
#     trainer.training.model_id_for_detecting = model['id']
#     detections = await trainer._do_detections()
#     assert len(detections) > 0


async def create_training_data(training: Training) -> TrainingData:
    training_data = TrainingData()

    image_data, _ = await TrainingsDownloader(training.context).download_training_data(training.images_folder)
    response = test_helper.LiveServerSession().get(f"/zauberzeug/projects/demo/data")
    assert response.status_code != 401, 'Authentification error - did you set LOOP_USERNAME and LOOP_PASSWORD in your environment?'
    assert response.status_code == 200
    data = response.json()
    training_data.categories = Category.from_list(data['categories'])
    training_data.image_data = image_data
    return training_data


def mock_epoch(number: int, confusion_matrix: Dict):
    os.makedirs('result/weights/', exist_ok=True)
    with open(f'result/weights/epoch{number}.json', 'w') as f:
        json.dump(confusion_matrix, f)
    with open(f'result/weights/epoch{number}.pt', 'wb') as f:
        f.write(b'0')
