import pytest
import yaml  # type: ignore
from learning_loop_node.trainer.exceptions import CriticalError
from learning_loop_node.trainer.hyperparameters import check_declaration, resolve

from ..hyperparameters import _TEMPLATE, _UPSTREAM_BY_SIZE, HYPERPARAMETERS, SIZE_DEPENDENT, size_dependent_values
from ..yolov5_format import FLIP_PROBABILITY, create_hyperparameter_file
from ..yolov5_trainer import _parse_point_sizes


def test_every_template_key_is_declared_with_the_template_default():
    declared = {h.name: h for h in HYPERPARAMETERS}
    for name, value in _TEMPLATE.items():
        assert name in declared, f'{name} from hyp_det.yaml is not declared'
        assert declared[name].default == value, name


def test_every_upstream_key_is_declared_with_upstream_default_unless_it_depends_on_the_size():
    declared = {h.name: h for h in HYPERPARAMETERS}
    for name, value in _UPSTREAM_BY_SIZE['s'].items():
        assert name in declared, f'{name} from the upstream hyperparameters is not declared'
        if name in _TEMPLATE:
            continue  # hyp_det.yaml overrides it, see above
        assert declared[name].default == (None if name in SIZE_DEPENDENT else value), name


def test_the_size_dependent_hyperparameters_are_the_ones_upstream_tunes_per_size():
    assert SIZE_DEPENDENT == ['lrf', 'cls', 'obj', 'scale', 'mixup', 'copy_paste']
    assert next(h for h in HYPERPARAMETERS if h.name == 'scale').notes.endswith('0.5 for n and s, 0.9 for m, l and x')


@pytest.mark.parametrize('variant, scale, copy_paste', [('s', 0.5, 0.0), ('m6', 0.9, 0.0), ('x', 0.9, 0.1)])
def test_the_size_dependent_values_follow_the_model_size(variant: str, scale: float, copy_paste: float):
    values = size_dependent_values(variant)
    assert values.keys() == set(SIZE_DEPENDENT)
    assert values['scale'] == scale
    assert values['copy_paste'] == copy_paste


def test_the_declared_defaults_keep_upstream_values_for_the_model_size(tmp_path):
    """What the trainer writes: the resolved hyperparameters, filled with the size-dependent values."""
    hyperparameters = resolve(HYPERPARAMETERS, {'mixup': 0.0})
    for name, value in size_dependent_values('l').items():
        hyperparameters.setdefault(name, value)
    hyp = tmp_path / 'hyp.yaml'
    create_hyperparameter_file(str(hyp), 'l', hyperparameters)
    written = yaml.safe_load(hyp.read_text())
    high = _UPSTREAM_BY_SIZE['l']
    assert high['scale'] != _UPSTREAM_BY_SIZE['s']['scale'], 'the test needs a value that differs by size'
    assert written['scale'] == high['scale']
    assert written['copy_paste'] == high['copy_paste']
    assert written['mixup'] == 0.0, "the training's own value"
    assert hyperparameters['scale'] == high['scale'], 'reported with the training'


def test_the_declaration_passes_the_check_the_node_runs_when_it_starts():
    """No name twice, no contradiction, every default fits; a failure here would stop the trainer node."""
    check_declaration(HYPERPARAMETERS)


def test_the_declared_defaults_resolve_to_themselves():
    resolved = resolve(HYPERPARAMETERS, {})
    assert resolved == {h.name: h.default for h in HYPERPARAMETERS if h.default is not None}
    assert resolved['epochs'] == _TEMPLATE['epochs']
    assert resolved['reset_points'] is False


def test_a_flip_that_is_on_flips_half_of_the_images(tmp_path):
    hyp = tmp_path / 'hyp.yaml'
    hyperparameters = {'fliplr': True, 'flipud': False}
    create_hyperparameter_file(str(hyp), 's', hyperparameters)
    written = yaml.safe_load(hyp.read_text())
    assert written['fliplr'] == FLIP_PROBABILITY
    assert written['flipud'] == 0.0
    assert hyperparameters == {'fliplr': True, 'flipud': False}, 'the training keeps the declared form'


def test_train_det_reads_a_tiny_float_back_as_a_number(tmp_path):
    """train_det.py reads hyp.yaml with PyYAML, which takes 5e-05 for a string."""
    hyp = tmp_path / 'hyp.yaml'
    perspective = 0.00005
    create_hyperparameter_file(str(hyp), 's', resolve(HYPERPARAMETERS, {'perspective': perspective}))
    assert yaml.safe_load(hyp.read_text())['perspective'] == perspective


def test_every_hyperparameter_has_a_note():
    """The loop shows the note as the description under each hyperparameter."""
    for hyperparameter in HYPERPARAMETERS:
        assert hyperparameter.notes, f'{hyperparameter.name} has no note'
        assert hyperparameter.notes[0].isupper(), f'the note of {hyperparameter.name} starts lowercase'


@pytest.mark.parametrize('name', ['flip_label_pairs', 'point_sizes_by_id'])
def test_a_malformed_pair_is_refused_before_the_training_starts(name):
    with pytest.raises(CriticalError, match=name):
        resolve(HYPERPARAMETERS, {name: 'a:b:c'})


def test_the_point_sizes_tolerate_spaces_and_a_trailing_comma():
    resolved = resolve(HYPERPARAMETERS, {'point_sizes_by_id': ' a : 0.05 ,'})
    assert _parse_point_sizes(resolved['point_sizes_by_id']) == {'a': 0.05}


@pytest.mark.parametrize('size', ['big', '0', '-0.1', 'nan'])
def test_a_point_size_that_is_not_a_positive_number_ends_the_training(size):
    with pytest.raises(CriticalError, match='point_sizes_by_id'):
        _parse_point_sizes(f'a:{size}')
