import shutil

import yaml  # type: ignore
from learning_loop_node.trainer.hyperparameters import check_declaration, resolve

from ..hyperparameters import _TEMPLATE, HYPERPARAMETERS, TEMPLATE_PATH
from ..yolov5_format import FLIP_PROBABILITY, set_hyperparameters_in_file


def test_every_template_key_is_declared_with_the_template_default():
    declared = {h.name: h for h in HYPERPARAMETERS}
    for name, value in _TEMPLATE.items():
        assert name in declared, f'{name} from hyp_det.yaml is not declared'
        assert declared[name].default == value, name


def test_the_declaration_passes_the_check_the_node_runs_when_it_starts():
    """No name twice, no contradiction, every default fits; a failure here would stop the trainer node."""
    check_declaration(HYPERPARAMETERS)


def test_the_declared_defaults_resolve_to_themselves():
    resolved = resolve(HYPERPARAMETERS, {})
    assert resolved == {h.name: h.default for h in HYPERPARAMETERS}
    assert resolved['epochs'] == _TEMPLATE['epochs']
    assert resolved['reset_points'] is False


def test_a_flip_that_is_on_flips_half_of_the_images(tmp_path):
    hyp = tmp_path / 'hyp.yaml'
    shutil.copy(TEMPLATE_PATH, hyp)
    hyperparameters = {'fliplr': True, 'flipud': False}
    set_hyperparameters_in_file(str(hyp), hyperparameters)
    written = yaml.safe_load(hyp.read_text())
    assert written['fliplr'] == FLIP_PROBABILITY
    assert written['flipud'] == 0.0
    assert hyperparameters == {'fliplr': True, 'flipud': False}, 'the training keeps the declared form'


def test_train_det_reads_a_tiny_float_back_as_a_number(tmp_path):
    """train_det.py reads hyp.yaml with PyYAML, which takes 5e-05 for a string."""
    hyp = tmp_path / 'hyp.yaml'
    shutil.copy(TEMPLATE_PATH, hyp)
    perspective = 0.00005
    set_hyperparameters_in_file(str(hyp), resolve(HYPERPARAMETERS, {'perspective': perspective}))
    assert yaml.safe_load(hyp.read_text())['perspective'] == perspective

def test_every_hyperparameter_has_a_note():
    """The loop shows the note as the description under each hyperparameter."""
    for hyperparameter in HYPERPARAMETERS:
        assert hyperparameter.notes, f'{hyperparameter.name} has no note'
        assert hyperparameter.notes[0].isupper(), f'the note of {hyperparameter.name} starts lowercase'
