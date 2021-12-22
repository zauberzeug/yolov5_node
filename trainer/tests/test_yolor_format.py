from learning_loop_node.trainer.training import Training
from yolor_format import create_yaml


def test_creating_yaml(use_test_dir):
    training = Training.parse_file('example_training.json')
    create_yaml(training)
    with open(f'{training.training_folder}/dataset.yaml', 'r') as f:
        yaml = f.read()

    assert yaml == '''names:
- purple
- green
nc: 2
test: /tmp/test_training/test.txt
train: /tmp/test_training/train.txt
val: /tmp/test_training/test.txt
'''
