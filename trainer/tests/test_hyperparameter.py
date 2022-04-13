from ruamel.yaml import YAML

from learning_loop_node.trainer.hyperparameter import Hyperparameter
from sympy import hyper
from yolov5_format import update_hyp
yaml = YAML()


def test_update_hyperparameter():
    def assert_yaml_content(yaml_path, **kwargs):
        with open(yaml_path) as f:
            content = yaml.load(f)
        for key, value in kwargs.items():
            # assert key in content
            assert content[key] == value

    import shutil
    shutil.copy('tests/test_data/hyp.yaml', '/tmp')

    hyperparameter = Hyperparameter(resolution=600, flip_rl=True, flip_ud=True)

    assert_yaml_content('/tmp/hyp.yaml', fliplr=0, flipud=0)
    update_hyp('/tmp/hyp.yaml', hyperparameter)
    assert_yaml_content('/tmp/hyp.yaml', fliplr=0.5, flipud=0.00856)
