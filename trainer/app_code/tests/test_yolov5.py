import asyncio
import glob
import json
import logging
import os
import shutil
import time
from pathlib import Path
from time import sleep
from typing import Dict
from uuid import uuid4

import pytest
from learning_loop_node.data_classes import (Category, Context, Hyperparameter,
                                             Training, TrainingData)
from learning_loop_node.data_exchanger import DataExchanger
from learning_loop_node.helper_functions.gdrive_downloader import g_download
from learning_loop_node.loop_communication import LoopCommunicator
from learning_loop_node.tests import test_helper
from learning_loop_node.trainer.downloader import TrainingsDownloader
from learning_loop_node.trainer.executor import Executor
from learning_loop_node.trainer.trainer_logic import TrainerLogic
from ruamel.yaml import YAML

from .. import model_files, yolov5_format
from ..yolov5_format import update_hyp
from ..yolov5_trainer import Yolov5TrainerLogic

os.environ['LOOP_ORGANIZATION'] = 'zauberzeug'
os.environ['LOOP_PROJECT'] = 'demo'

logging.basicConfig(level=logging.DEBUG)


yaml = YAML()


def test_update_hyperparameter():
    def assert_yaml_content(yaml_path, **kwargs):
        with open(yaml_path) as f:
            content = yaml.load(f)
        for key, value in kwargs.items():
            # assert key in content
            assert content[key] == value

    shutil.copy('app_code/tests/test_data/hyp.yaml', '/tmp')

    hyperparameter = Hyperparameter(resolution=600, flip_rl=True, flip_ud=True)

    assert_yaml_content('/tmp/hyp.yaml', fliplr=0, flipud=0)
    update_hyp('/tmp/hyp.yaml', hyperparameter)
    assert_yaml_content('/tmp/hyp.yaml', fliplr=0.5, flipud=0.5)


async def test_communicator(glc: LoopCommunicator):
    response = await glc.get('/zauberzeug/projects/demo/data')
    assert response.status_code == 200


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

    os.environ['YOLOV5_MODE'] = 'DETECTION'
    trainer = Yolov5TrainerLogic()
    trainer._training = Training(id='someid', context=Context(organization='o', project='p'),  # pylint: disable=protected-access
                                 project_folder='./', images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=image_data, categories=categories)

    yolov5_format.create_file_structure(trainer.training)

    with open('./train/image_1.txt', 'r') as f:
        lines = f.readlines()

    assert '0 0.500000 0.600000 0.200000 0.200000' in lines[0]
    assert '1 0.600000 0.700000 0.300000 0.300000' in lines[1]


# @pytest.mark.asyncio()
# FLAKY! model.pt can sometimes not be accessed (unpickling stack underflow)
async def test_training_creates_model(use_training_dir, data_exchanger: DataExchanger, glc: LoopCommunicator):

    os.remove('/tmp/model.pt')
    await asyncio.sleep(0.1)

    training = Training(
        id=str(uuid4()),
        project_folder=os.getcwd(),
        training_folder=os.getcwd() + '/training',
        images_folder=os.getcwd() + '/images',
        base_model_id='model.pt',
        context=Context(project='demo', organization='zauberzeug'),
    )
    training.data = await create_training_data(training, data_exchanger, glc)

    yolov5_format.create_file_structure(training)
    print('CREATED FILE STRUCTURE', os.getcwd())

    executor = Executor(os.getcwd())
    # from https://github.com/WongKinYiu/yolor#training

    ROOT = Path(__file__).resolve().parents[2]
    cmd = f'WANDB_MODE=disabled python {ROOT/"train_det.py"} --project training --name result --batch 4 --img 416 --data training/dataset.yaml --weights model.pt --epochs 1'
    executor.start(cmd)
    while executor.is_process_running():
        sleep(1)
        # logging.debug(executor.get_log())

    logging.info(executor.get_log())
    sleep(1)
    assert '1 epochs completed' in executor.get_log()
    assert 'best.pt' in executor.get_log()
    best = training.training_folder + '/result/weights/best.pt'
    assert os.path.isfile(best)


async def test_parse_progress_from_log(use_training_dir, data_exchanger: DataExchanger, glc: LoopCommunicator):
    os.environ['YOLOV5_MODE'] = 'DETECTION'
    trainer = Yolov5TrainerLogic()
    trainer.epochs = 2
    os.remove('/tmp/model.pt')
    trainer._training = Training(  # pylint: disable=protected-access
        id=str(uuid4()),
        project_folder=os.getcwd(),
        training_folder=os.getcwd() + '/training',
        images_folder=os.getcwd() + '/images',
        base_model_id='model.pt',
        context=Context(project='demo', organization='zauberzeug'),
    )
    trainer.training.data = await create_training_data(trainer.training, data_exchanger, glc)
    yolov5_format.create_file_structure(trainer.training)

    trainer._executor = Executor(os.getcwd())
    ROOT = Path(__file__).resolve().parents[2]
    cmd = f'WANDB_MODE=disabled python {ROOT/"train_det.py"} --project training --name result --batch 4 --img 416 --data training/dataset.yaml --weights model.pt --epochs {trainer.epochs}'
    trainer.executor.start(cmd)
    while trainer.executor.is_process_running():
        sleep(1)

    logging.info(trainer.executor.get_log())
    assert f'{trainer.epochs} epochs completed' in trainer.executor.get_log()
    assert trainer.progress == 1.0


def test_new_model_discovery(use_training_dir):
    os.environ['YOLOV5_MODE'] = 'DETECTION'
    trainer = Yolov5TrainerLogic()
    trainer._training = Training(id='someid', context=Context(organization='o', project='p'),  # pylint: disable=protected-access
                                 project_folder='./',  images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=[], categories=[
                                         Category(name='class_a', id='uuid_of_class_a', type='box')])
    assert trainer.get_new_model() is None, 'should not find any models'

    model_path = 'result/weights/published/latest.pt'
    mock_epoch(1, {'class_a': {'fp': 0, 'tp': 1, 'fn': 0}})
    model = trainer.get_new_model()

    assert model is not None and model.confusion_matrix is not None
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 1
    trainer.on_model_published(model)
    assert os.path.isfile(model_path)
    modification_date = os.path.getmtime(model_path)

    mock_epoch(2, {'class_a': {'fp': 1, 'tp': 2, 'fn': 1}})
    model = trainer.get_new_model()

    assert model is not None and model.confusion_matrix is not None
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 2

    trainer.on_model_published(model)

    assert trainer.get_new_model() is None, 'again we should not find any new models'

    time.sleep(0.1)

    mock_epoch(3, {'class_a': {'fp': 0, 'tp': 3, 'fn': 1}})
    model = trainer.get_new_model()

    assert model is not None and model.confusion_matrix is not None
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 3
    trainer.on_model_published(model)
    assert os.path.isfile(model_path)
    new_modification_date = os.path.getmtime(model_path)
    assert new_modification_date > modification_date

    # get_latest_model_file
    files = trainer.get_latest_model_files()
    print(files)
    assert files == {
        'yolov5_pytorch': ['/tmp/model.pt', '/tmp/test_training/hyp.yaml'],
        'yolov5_wts': ['/tmp/model.wts']}


def test_newest_model_is_used(use_training_dir):
    os.environ['YOLOV5_MODE'] = 'DETECTION'
    trainer = Yolov5TrainerLogic()
    trainer._training = Training(id='someid', context=Context(organization='o', project='p'),  # pylint: disable=protected-access
                                 project_folder='./', images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=[], categories=[
                                         Category(name='class_a', id='uuid_of_class_a', type='box')])

    # create some models.
    mock_epoch(10, {})
    mock_epoch(200, {})

    new_model = trainer.get_new_model()
    assert new_model is not None and new_model.meta_information is not None
    assert 'epoch10.pt' not in new_model.meta_information['weightfile']
    assert 'epoch200.pt' in new_model.meta_information['weightfile']


def test_old_model_files_are_deleted_on_publish(use_training_dir):
    os.environ['YOLOV5_MODE'] = 'DETECTION'
    trainer = Yolov5TrainerLogic()
    trainer._training = Training(id='someid', context=Context(organization='o', project='p'),  # pylint: disable=protected-access
                                 project_folder='./', images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=[], categories=[
                                         Category(name='class_a', id='uuid_of_class_a', type='box')])
    assert trainer.get_new_model() is None, 'should not find any models'

    mock_epoch(1, {'class_a': {'fp': 0, 'tp': 1, 'fn': 0}})
    new_model = trainer.get_new_model()

    assert new_model is not None and new_model.confusion_matrix is not None
    assert new_model.confusion_matrix['uuid_of_class_a']['tp'] == 1
    mock_epoch(2, {'class_a': {'fp': 1, 'tp': 2, 'fn': 1}})

    _, _, files = next(os.walk("result/weights"))
    assert len(files) == 4

    new_model = trainer.get_new_model()
    assert new_model is not None
    trainer.on_model_published(new_model)
    _, _, files = next(os.walk("result/weights/published"))
    assert len(files) == 1
    assert os.path.isfile('result/weights/published/latest.pt')

    _, _, files = next(os.walk("result/weights"))
    assert len(files) == 0


def test_newer_model_files_are_kept_during_deleting(use_training_dir):
    os.environ['YOLOV5_MODE'] = 'DETECTION'
    trainer = Yolov5TrainerLogic()
    trainer._training = Training(id='someid', context=Context(organization='o', project='p'),  # pylint: disable=protected-access
                                 project_folder='./',  images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=[], categories=[
                                         Category(name='class_a', id='uuid_of_class_a', type='box')])

    # create some models.
    mock_epoch(10, {})
    mock_epoch(200, {})
    new_model = trainer.get_new_model()
    assert new_model is not None and new_model.meta_information is not None
    assert 'epoch200.pt' in new_model.meta_information['weightfile']
    mock_epoch(201, {})  # An epoch is finished after during communication with the LearningLoop

    trainer.on_model_published(new_model)

    all_model_files = model_files.get_all_weightfiles(Path(trainer.training.training_folder))
    assert len(all_model_files) == 1
    assert 'epoch201.pt' in str(all_model_files[0]), 'Epoch201 is not yed synced. It should not be deleted.'


@pytest.mark.asyncio()
async def test_clear_training_data(use_training_dir):
    os.environ['YOLOV5_MODE'] = 'DETECTION'
    trainer = Yolov5TrainerLogic()
    trainer._training = Training(id='someid', context=Context(organization='o', project='p'),  # pylint: disable=protected-access
                                 project_folder='./', images_folder='./', training_folder='./')
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
    for file in data:
        print(file)
    assert len(data) == 9
    files = [f for f in data if os.path.isfile(f)]
    assert len(files) == 5

    await trainer.clear_training_data(trainer.training.training_folder)
    data = glob.glob(trainer.training.training_folder + '/**', recursive=True)
    assert len(data) == 5
    files = [f for f in data if os.path.isfile(f)]
    assert len(files) == 2  # Note: Do not delete last_training.log und best.pt


@pytest.mark.skip(reason="This test needs to be updated to newever version of learning-loop. Functionality is tested in learning_loop_node")
@pytest.mark.asyncio()
# TODO: Fix this test (results in file not found on server)
async def test_detecting(create_project, glc: LoopCommunicator):
    # from learning_loop_node.loop import Loop
    # loop = Loop()
    logging.info('downloading model from gdrive')

    file_id = '1sZWa053fWT9PodrujDX90psmjhFVLyBV'
    destination = '/tmp/model.zip'
    g_download(file_id, destination)

    test_helper.unzip(destination, '/tmp/model')

    logging.info('uploading model')
    data = ['/tmp/model/model.pt', '/tmp/model/model.json']
    response = await glc.put('zauberzeug/projects/demo/1/models/latest/yolov5_pytorch/file', files=data)
    if response.status_code != 200:
        msg = f'unexpected status code {response.status_code} while putting model'
        logging.error(msg)
        raise (Exception(msg))
    model = await response.json()

    # data = test_helper.prepare_formdata(['tests/example_images/8647fc30-c46c-4d13-a3fd-ead3b9a67652.jpg'])
    # response = await glc.post(f'zauberzeug/projects/pytest/images', files=data)
    # if response.status != 200:
    #     msg = f'unexpected status code {response.status} while posting a new image'
    #     logging.error(msg)
    #     raise (Exception(msg))
    # image = await response.json()

    os.environ['YOLOV5_MODE'] = 'DETECTION'
    trainer = Yolov5TrainerLogic()
    context = Context(organization='zauberzeug', project='demo')
    trainer._training = TrainerLogic.generate_training(context)  # pylint: disable=protected-access
    trainer.training.model_id_for_detecting = model['id']
    detections = await trainer._do_detections()  # pylint: disable=protected-access
    assert detections is not None
    assert len(detections) > 0

# -------------------- HELPERS ----------------------------------------------------------------


async def create_training_data(training: Training, data_exchanger: DataExchanger, glc: LoopCommunicator) -> TrainingData:
    training_data = TrainingData()

    image_data, _ = await TrainingsDownloader(data_exchanger).download_training_data(training.images_folder)
    logging.info(f'got {len(image_data)} images')

    response = await glc.get("/zauberzeug/projects/demo/data")
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
