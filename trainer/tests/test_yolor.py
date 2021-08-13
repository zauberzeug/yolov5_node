import os
from learning_loop_node.trainer.executor import Executor
from learning_loop_node.trainer.training import Training
import yolor_format
import logging
from time import sleep


def test_training_creates_model(use_test_dir):
    training = Training.parse_file('example_training.json')
    yolor_format.create_file_structure(training)

    executor = Executor(training.training_folder)
    # from https://github.com/WongKinYiu/yolor#training
    cmd = f'python /yolor/train.py --batch-size 4 --img 800 800 --data {training.training_folder}/dataset.yaml --cfg config.cfg --weights weights.pt --device 0 --name yolor --hyp /yolor/data/hyp.scratch.1280.yaml --epochs 1'
    executor.start(cmd)
    while executor.is_process_running():
        sleep(1)
        logging.debug(executor.get_log())

    assert '1 epochs completed' in executor.get_log()
    assert 'best.pt' in executor.get_log()
    assert os.path.isfile(training.training_folder + '/runs/train/yolor/weights/best.pt')
