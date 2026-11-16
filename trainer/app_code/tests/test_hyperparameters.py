import pytest
import yaml  # type: ignore
from learning_loop_node.trainer.exceptions import CriticalError
from learning_loop_node.trainer.hyperparameters import check_declaration, for_variant, resolve

from ..hyperparameters import _TEMPLATE, _UPSTREAM_BY_VARIANT, HYPERPARAMETERS, MODEL_VARIANTS
from ..yolov5_format import FLIP_PROBABILITY, create_hyperparameter_file
from ..yolov5_trainer import Yolov5TrainerLogic, _parse_point_sizes


def test_every_template_key_is_declared_with_the_template_default():
    declared = {h.name: h for h in HYPERPARAMETERS}
    for name, value in _TEMPLATE.items():
        assert name in declared, f'{name} from hyp_det.yaml is not declared'
        assert declared[name].default == value, name


@pytest.mark.parametrize('variant', MODEL_VARIANTS)
def test_every_upstream_key_is_declared_with_the_upstream_default_of_each_variant(variant: str):
    declared = {h.name: h for h in for_variant(HYPERPARAMETERS, variant)}
    for name, value in _UPSTREAM_BY_VARIANT[variant].items():
        assert name in declared, f'{name} from the upstream hyperparameters is not declared'
        if name in _TEMPLATE:
            continue  # hyp_det.yaml overrides it, see above
        assert declared[name].default == value, f'{name} for {variant}'


def test_the_variant_defaults_are_the_ones_upstream_tunes_per_size():
    by_variant = {h.name: h.default_by_variant for h in HYPERPARAMETERS if h.default_by_variant}
    assert list(by_variant) == ['lrf', 'cls', 'obj', 'scale', 'mixup', 'copy_paste']
    assert set(by_variant['scale']) == {'m', 'l', 'x', 'm6', 'l6', 'x6'}, 'n and s take the default'


def test_the_model_variants_are_the_pretrained_models():
    assert MODEL_VARIANTS == [model.name for model in Yolov5TrainerLogic.provided_pretrained_models.fget(None)]


def test_a_training_of_a_large_variant_writes_upstream_values_for_its_size(tmp_path):
    """What the trainer writes: the hyperparameters the node library resolved for the training's variant."""
    hyperparameters = resolve(for_variant(HYPERPARAMETERS, 'l'), {'mixup': 0.0})
    hyp = tmp_path / 'hyp.yaml'
    create_hyperparameter_file(str(hyp), 'l', hyperparameters)
    written = yaml.safe_load(hyp.read_text())
    high = _UPSTREAM_BY_VARIANT['l']
    assert high['scale'] != _UPSTREAM_BY_VARIANT['s']['scale'], 'the test needs a value that differs by size'
    assert written['scale'] == high['scale']
    assert written['copy_paste'] == high['copy_paste']
    assert written['mixup'] == 0.0, "the training's own value"
    assert hyperparameters['scale'] == high['scale'], 'reported with the training'


def test_the_declaration_passes_the_check_the_node_runs_when_it_starts():
    """No name twice, no contradiction, every default fits; a failure here would stop the trainer node."""
    check_declaration(HYPERPARAMETERS, MODEL_VARIANTS)


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
