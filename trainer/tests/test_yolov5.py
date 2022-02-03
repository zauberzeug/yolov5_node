import os
import shutil
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
    training.data = await TrainingsDownloader(training.context).download_training_data(training.images_folder)
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
    trainer.training.data = TrainingData(image_data=[], categories={'class_a': 'uuid_of_class_a'})
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
    assert False

def test_old_model_files_are_deleted_on_publish(use_training_dir):
    trainer = Yolov5Trainer()
    trainer.training = Training(id='someid', context=Context(organization='o', project='p'), project_folder='./',
                                images_folder='./', training_folder='./')
    trainer.training.data = TrainingData(image_data=[], categories={'class_a': 'uuid_of_class_a'})
    assert trainer.get_new_model() is None, 'should not find any models'
    
    mock_epoch(1, {'class_a': {'fp': 0, 'tp': 1, 'fn': 0}})
    model = trainer.get_new_model()
    assert model.confusion_matrix['uuid_of_class_a']['tp'] == 1
    mock_epoch(2, {'class_a': {'fp': 1, 'tp': 2, 'fn': 1}})

    _, _, files = next(os.walk("result/weights"))
    logging.info(files)
    assert len(files) == 4

    model = trainer.get_new_model()
    trainer.on_model_published(model, 'uuid2')
    _, _, files = next(os.walk("result/weights/published"))
    assert len(files) == 1
    assert os.path.isfile('result/weights/published/uuid2.pt'), 'weightfile should be renamed to learning loop id'

    _, _, files = next(os.walk("result/weights"))
    assert len(files) == 0


def mock_epoch(number: int, confusion_matrix: Dict):
    os.makedirs('result/weights/', exist_ok=True)
    with open(f'result/weights/epoch{number}.json', 'w') as f:
        json.dump(confusion_matrix, f)
    with open(f'result/weights/epoch{number}.pt', 'wb') as f:
        f.write(b'0')
