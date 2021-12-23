import os
from learning_loop_node import Context
from learning_loop_node.trainer import Executor, Training, TrainingsDownloader
import onnx
import yolov5_format
import logging
from time import sleep
import glob
import pytest
from uuid import uuid4


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
    cmd = f'WANDB_MODE=disabled python /yolov5/train.py --project training --name result --batch 4 --img 416 --data training/dataset.yaml --weights yolov5n.pt --epochs 1'
    executor.start(cmd)
    while executor.is_process_running():
        sleep(1)
        logging.debug(executor.get_log())

    logging.info(executor.get_log())
    assert '1 epochs completed' in executor.get_log()
    assert 'best.pt' in executor.get_log()
    best = training.training_folder + '/result/weights/best.pt'
    assert os.path.isfile(best)
