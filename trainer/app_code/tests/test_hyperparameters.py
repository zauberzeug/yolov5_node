from learning_loop_node.trainer.hyperparameters import resolve

from ..hyperparameters import _TEMPLATE, HYPERPARAMETERS


def test_every_template_key_is_declared_with_the_template_default():
    declared = {h.name: h for h in HYPERPARAMETERS}
    for name, value in _TEMPLATE.items():
        assert name in declared, f'{name} from hyp_det.yaml is not declared'
        assert declared[name].default == value, name


def test_no_name_is_declared_twice():
    assert len({h.name for h in HYPERPARAMETERS}) == len(HYPERPARAMETERS)


def test_the_declared_defaults_resolve_to_themselves():
    resolved = resolve(HYPERPARAMETERS, {})
    assert resolved == {h.name: h.default for h in HYPERPARAMETERS}
    assert resolved['epochs'] == _TEMPLATE['epochs']
    assert resolved['reset_points'] is False
