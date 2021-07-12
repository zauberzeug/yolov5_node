from coco import create_dataset, create_yaml
from learning_loop_node.trainer.training import Training
from icecream import ic


def test_creating_coco_from_training_object(use_test_dir):
    training = Training.parse_file('example_training.json')

    train = create_dataset(training, set='train').coco()
    test = create_dataset(training, set='test').coco()

    assert len(train['categories']) == 2
    assert len(train['images']) == 3
    assert len(train['annotations']) == 3*3

    assert len(test['categories']) == 2
    assert len(test['images']) == 1
    assert len(test['annotations']) == 3


def test_creating_yaml(use_test_dir):
    training = Training.parse_file('example_training.json')
    yaml = create_yaml(training)

    assert yaml == '''names:
- purple
- green
nc: 2
test: /tmp/test_training/test.json
train: /tmp/test_training/train.json
val: /tmp/test_training/test.json
'''
