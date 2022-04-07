import os
from typing import Dict
from learning_loop_node import Context
from learning_loop_node.trainer import Executor, Training, TrainingsDownloader
from learning_loop_node.trainer.training_data import TrainingData
from pydantic.types import Json
import yolov5_format
from yolov5_trainer import Yolov5Trainer
import logging
from time import sleep
import pytest
from uuid import uuid4
import json
import glob
from learning_loop_node.tests import test_helper
from learning_loop_node.gdrive_downloader import g_download
from learning_loop_node.loop import loop
from learning_loop_node.data_classes.category import Category


@pytest.mark.asyncio()
async def test_training_creates_model(use_training_dir):
    os.remove('/tmp/model.pt')
    training = Training(
        id=str(uuid4()),
        project_folder=os.getcwd(),
        training_folder=os.getcwd() + '/training',
        images_folder=os.getcwd() + '/images',
        base_model='model.pt',
        context=Context(project='demo', organization='zauberzeug'),
    )
    training.data = await TrainingsDownloader(training.context).download_training_data(training.images_folder)
    response = test_helper.LiveServerSession().get(f"/api/zauberzeug/projects/demo/data")
    assert response.status_code == 200
    data = response.json()
    training.data.categories = Category.from_list(data['categories'])

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


def test_new_model_discovery(use_training_dir):
    trainer = Yolov5Trainer()
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'), project_folder='./',
                                images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=[], categories=[Category(name='class_a', id='uuid_of_class_a', type='box')])
    assert trainer.get_new_model() is None, 'should not find any models'
    mock_epoch(1, {'class_a': {'fp': 0, 'tp': 1, 'fn': 0}})
    model = trainer.get_new_model()
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 1
    trainer.on_model_published(model, 'uuid1')
    assert os.path.isfile('result/weights/published/uuid1.pt'), 'weightfile should be renamed to learning loop id'

    mock_epoch(2, {'class_a': {'fp': 1, 'tp': 2, 'fn': 1}})
    model = trainer.get_new_model()
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 2
    trainer.on_model_published(model, 'uuid2')

    assert trainer.get_new_model() is None, 'again we should not find any new models'

    mock_epoch(3, {'class_a': {'fp': 0, 'tp': 3, 'fn': 1}})
    model = trainer.get_new_model()
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 3
    trainer.on_model_published(model, 'uuid3')
    assert os.path.isfile('result/weights/published/uuid3.pt'), 'weightfile should be renamed to learning loop id'


def test_old_model_files_are_deleted_on_publish(use_training_dir):
    trainer = Yolov5Trainer()
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'), project_folder='./',
                                images_folder='./', training_folder='./')
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
    trainer.on_model_published(model, 'uuid2')
    _, _, files = next(os.walk("result/weights/published"))
    assert len(files) == 1
    assert os.path.isfile('result/weights/published/uuid2.pt'), 'weightfile should be renamed to learning loop id'

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
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'box_categories': 2,  'point_categories': 1, 'inbox': 0, 'annotate': 0, 'review': 0, 'complete': 0, 'image_style': 'plain',
                             'thumbs': False}
    assert test_helper.LiveServerSession().post(f"/api/zauberzeug/projects/generator",
                                                json=project_configuration).status_code == 200
    yield
    test_helper.LiveServerSession().delete(f"/api/zauberzeug/projects/pytest?keep_images=true")


@pytest.mark.asyncio()
async def test_detecting(create_project):
    from learning_loop_node.loop import Loop
    loop = Loop()
    logging.debug('downloading model from gdrive')
    if not os.path.exists('/tmp/model/model.pt'):
        file_id = '1sZWa053fWT9PodrujDX90psmjhFVLyBV'
        destination = '/tmp/model.zip'
        g_download(file_id, destination)
        test_helper.unzip(destination, '/tmp/model')

    logging.debug('uploading model')
    data = test_helper.prepare_formdata(['/tmp/model/model.pt'])
    async with loop.post(f'api/zauberzeug/projects/pytest/models/yolov5_pytorch', data) as response:
        if response.status != 200:
            msg = f'unexpected status code {response.status} while putting model'
            logging.error(msg)
            raise(Exception(msg))
        model = await response.json()

    data = test_helper.prepare_formdata(['tests/example_images/8647fc30-c46c-4d13-a3fd-ead3b9a67652.jpg'])
    async with loop.post(f'api/zauberzeug/projects/pytest/images', data) as response:
        if response.status != 200:
            msg = f'unexpected status code {response.status} while posting a new image'
            logging.error(msg)
            raise(Exception(msg))
        image = await response.json()

    trainer = Yolov5Trainer()
    detections = await trainer.do_detections(Context(organization='zauberzeug', project='pytest'), model['id'], 'yolov5_pytorch')
    assert len(detections) > 0


def mock_epoch(number: int, confusion_matrix: Dict):
    os.makedirs('result/weights/', exist_ok=True)
    with open(f'result/weights/epoch{number}.json', 'w') as f:
        json.dump(confusion_matrix, f)
    with open(f'result/weights/epoch{number}.pt', 'wb') as f:
        f.write(b'0')
