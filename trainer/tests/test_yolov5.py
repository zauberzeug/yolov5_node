import os
from learning_loop_node.trainer.executor import Executor
from learning_loop_node.trainer.training import Training
import onnx
import yolov5_format
import logging
from time import sleep
import glob


def test_training_creates_model(use_test_dir):
    training = Training.parse_file('example_training.json')
    yolov5_format.create_file_structure(training)

    executor = Executor(training.training_folder)
    # from https://github.com/WongKinYiu/yolor#training
    cmd = f'python /yolov5/train.py --batch-size 4 --img 400 400 --data {training.training_folder}/dataset.yaml --cfg model.cfg --weights model.pt --device 0 --name yolor --hyp /yolor/data/hyp.scratch.1280.yaml --epochs 1'
    executor.start(cmd)
    while executor.is_process_running():
        sleep(1)
        logging.debug(executor.get_log())

    assert '1 epochs completed' in executor.get_log()
    assert 'best.pt' in executor.get_log()
    best = training.training_folder + '/runs/train/yolor/weights/best.pt'
    assert os.path.isfile(best)
    os.rename(best, training.training_folder + '/best.pt')


def test_onnx_export():
    yolor_weights = '/tmp/model.pt'
    onnx_model = onnx.export(yolor_weights)
    assert os.path.isfile(onnx_model)