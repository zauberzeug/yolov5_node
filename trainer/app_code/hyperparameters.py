"""The hyperparameters this trainer accepts, as reported to the loop.

YOLOv5's own hyperparameters take upstream's defaults for the model size (see
``yolov5_format.create_hyperparameter_file``); those that differ between the sizes are declared without a default,
the trainer fills them in when a training starts. The node's own ones take theirs from ``hyp_det.yaml``. The ones the
trainer reads itself are the standard ones, the rest are expert hyperparameters. The loop sends only what a project
sets, and the declared defaults fill in the rest before a training starts.
"""
from pathlib import Path
from typing import Any

import yaml  # type: ignore
from learning_loop_node.data_classes import Hyperparameter
from learning_loop_node.trainer.batch_size import REQUESTED_BATCH_SIZE
from learning_loop_node.trainer.training_results import UPLOAD_EXAMPLES, UPLOAD_LOGS
from learning_loop_node.trainer.validation_set import COMPUTE_VALIDATION_SET_HYPERPARAMETER

from .batch_size_calculation import MIN_STRIDE
from .yolov5_format import NODE_HYPERPARAMETERS, UPSTREAM_LEVEL_BY_SIZE, upstream_hyperparameter_file


def _load(path: Path) -> dict[str, Any]:
    with open(path, encoding='utf-8') as file:
        return dict(yaml.safe_load(file))


_TEMPLATE = _load(NODE_HYPERPARAMETERS)
_UPSTREAM_BY_SIZE = {size: _load(upstream_hyperparameter_file(size)) for size in UPSTREAM_LEVEL_BY_SIZE}

SIZE_DEPENDENT = [name for name in _UPSTREAM_BY_SIZE['s']
                  if len({str(values[name]) for values in _UPSTREAM_BY_SIZE.values()}) > 1]
"""YOLOv5's hyperparameters whose upstream default differs between the model sizes."""

_TYPES = {bool: 'bool', int: 'int', float: 'float', str: 'str'}

DEFAULT_POINT_SIZE = 0.03
"""The point size, as a fraction of the image, of a point category ``point_sizes_by_id`` does not name."""


def size_dependent_values(model_variant: str) -> dict[str, Any]:
    """The upstream values of the ``SIZE_DEPENDENT`` hyperparameters for the size of ``model_variant``.

    :raises CriticalError: If the model variant has no upstream hyperparameters.
    """
    upstream = _load(upstream_hyperparameter_file(model_variant))
    return {name: upstream[name] for name in SIZE_DEPENDENT}


def _native(name: str, *, notes: str, minimum: float | None = None, maximum: float | None = None,
            expert: bool = True) -> Hyperparameter:
    """One of YOLOv5's own hyperparameters with upstream's default, or none where that depends on the model size."""
    by_size = {size: values[name] for size, values in _UPSTREAM_BY_SIZE.items()}
    value = by_size['s']
    if name in SIZE_DEPENDENT:
        notes = f'{notes}; default {_by_size_text(by_size)}'
    return Hyperparameter(name=name, type=_TYPES[type(value)], default=None if name in SIZE_DEPENDENT else value,
                          minimum=minimum, maximum=maximum, notes=notes, expert=expert)


def _by_size_text(by_size: dict[str, Any]) -> str:
    """E.g. ``0.5 for n and s, 0.9 for m, l and x``."""
    sizes_by_value: dict[str, list[str]] = {}
    for size, value in by_size.items():
        sizes_by_value.setdefault(str(value), []).append(size)
    return ', '.join(f'{value} for {_and(sizes)}' for value, sizes in sizes_by_value.items())


def _and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f'{", ".join(items[:-1])} and {items[-1]}'


def _from_template(name: str, *, notes: str, minimum: float | None = None, maximum: float | None = None,
                   expert: bool = True) -> Hyperparameter:
    """A hyperparameter of ``hyp_det.yaml``, its type and default taken from the template."""
    default = _TEMPLATE[name]
    return Hyperparameter(name=name, type=_TYPES[type(default)], default=default, minimum=minimum, maximum=maximum,
                          notes=notes, expert=expert)


def _probability(name: str, notes: str) -> Hyperparameter:
    return _native(name, notes=notes, minimum=0, maximum=1)


def _pairs(name: str, notes: str, *, expert: bool = True) -> Hyperparameter:
    """A ``"key:value,..."`` hyperparameter of ``hyp_det.yaml``, checked by the node library before a training starts."""
    return Hyperparameter(name=name, type='pairs', default=_TEMPLATE[name], notes=notes, expert=expert)


HYPERPARAMETERS = [
    Hyperparameter(name='resolution', type='int', default=832, minimum=2 * MIN_STRIDE,
                   notes="Square input size in pixels, a multiple of the model's largest stride"),
    _from_template('epochs', minimum=1, expert=False,
                   notes='Maximum number of epochs; early stopping usually ends the training earlier'),
    _from_template('patience', minimum=0, expert=False,
                   notes='Epochs without improvement before the training stops early, 0 disables early stopping'),
    Hyperparameter(name=REQUESTED_BATCH_SIZE, type='int', default=0, minimum=0,
                   notes='Upper bound for the measured batch size, 0 means 128'),
    # on or off rather than a probability, see yolov5_format.FLIP_PROBABILITY; the template keeps 0.0 for off
    Hyperparameter(name='fliplr', type='bool', default=False, notes='Flip half of the images horizontally'),
    Hyperparameter(name='flipud', type='bool', default=False, notes='Flip half of the images vertically'),
    _pairs('flip_label_pairs', 'Categories swapped by a horizontal flip, as "uuid:uuid,..."', expert=False),
    _from_template('detect_nms_conf_thres', minimum=0, maximum=1,
                   notes='Minimum confidence of a detection, in validation and in detection'),
    _from_template('detect_nms_iou_thres', minimum=0, maximum=1,
                   notes='IoU above which the weaker of two detections is dropped, in validation and in detection'),
    _native('lr0', notes='Initial learning rate', minimum=0, expert=False),
    _native('lrf', notes='Final learning rate, as a fraction of lr0', minimum=0),
    _native('momentum', notes='SGD momentum, Adam beta1', minimum=0, maximum=1),
    _native('weight_decay', notes='Weight decay of the optimizer', minimum=0),
    _native('warmup_epochs', minimum=0,
            notes='Epochs over which the learning rate warms up, fractions are fine'),
    _native('warmup_momentum', notes='Momentum at the start of the warmup', minimum=0, maximum=1),
    _native('warmup_bias_lr', notes='Learning rate of the biases at the start of the warmup', minimum=0),
    _native('box', notes='Box loss gain', minimum=0),
    _native('cls', notes='Class loss gain', minimum=0),
    _native('cls_pw', notes='Class BCE loss positive weight', minimum=0),
    _native('obj', notes='Objectness loss gain', minimum=0),
    _native('obj_pw', notes='Objectness BCE loss positive weight', minimum=0),
    _native('iou_t', notes='Unused by this YOLOv5 version; matching uses anchor_t', minimum=0, maximum=1),
    _native('anchor_t', notes='Largest size ratio between a label and an anchor that still matches', minimum=0),
    _native('fl_gamma', notes='Focal loss gamma, 0 disables it', minimum=0),
    _probability('hsv_h', 'Hue augmentation, as a fraction'),
    _probability('hsv_s', 'Saturation augmentation, as a fraction'),
    _probability('hsv_v', 'Value augmentation, as a fraction'),
    _native('degrees', notes='Image rotation, +/- degrees'),
    _probability('translate', 'Image translation, +/- fraction'),
    # below 1, or the drawn scale factor 1 +/- scale can reach zero; upstream evolves it within 0.9 too
    _native('scale', notes='Image scale, +/- gain', minimum=0, maximum=0.9),
    _native('shear', notes='Image shear, +/- degrees'),
    _native('perspective', notes='Image perspective, +/- fraction, 0 to 0.001', minimum=0, maximum=0.001),
    _probability('mosaic', 'Probability of combining four images into one'),
    _probability('mixup', 'Probability of blending two mosaic images, only applied when mosaic is'),
    _probability('copy_paste', 'Copy-paste of segments; has no effect on box and point labels'),
    _from_template('reset_points', notes='Resize the point labels to point_sizes_by_id'),
    _pairs('point_sizes_by_id',
           f'Point size per category as a fraction of the image, "uuid:size,...", used by reset_points; '
           f'{DEFAULT_POINT_SIZE} if unset'),
    COMPUTE_VALIDATION_SET_HYPERPARAMETER,
    Hyperparameter(name=UPLOAD_EXAMPLES, type='bool', default=False,
                   notes='Upload the example images with the training results', expert=True),
    Hyperparameter(name=UPLOAD_LOGS, type='bool', default=False,
                   notes='Upload the training log with the training results', expert=True),
]
"""What this trainer accepts, reported to the loop."""
