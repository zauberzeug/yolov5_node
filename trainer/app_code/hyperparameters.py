"""The hyperparameters this trainer accepts, as reported to the loop.

The defaults of everything ``hyp_det.yaml`` holds are read from that template, which ``train_det.py`` is run with,
so they live in one place. The template's own knobs of YOLOv5 are declared as expert hyperparameters; the ones the
trainer reads itself are the standard ones. The loop sends only what a project sets, and the declared defaults fill in
the rest before a training starts.
"""
from pathlib import Path
from typing import Any

import yaml  # type: ignore
from learning_loop_node.data_classes import Hyperparameter
from learning_loop_node.trainer.batch_size import REQUESTED_BATCH_SIZE
from learning_loop_node.trainer.validation_set import COMPUTE_VALIDATION_SET_HYPERPARAMETER

from .batch_size_calculation import MIN_STRIDE

TEMPLATE_PATH = Path(__file__).resolve().parents[1] / 'hyp_det.yaml'
"""The hyperparameter file a training starts from, see ``Yolov5TrainerLogic._start_training_from_scratch``."""

with open(TEMPLATE_PATH, encoding='utf-8') as _file:
    _TEMPLATE: dict[str, Any] = dict(yaml.safe_load(_file))

_TYPES = {bool: 'bool', int: 'int', float: 'float', str: 'str'}

DEFAULT_POINT_SIZE = 0.03
"""The point size, as a fraction of the image, of a point category ``point_sizes_by_id`` does not name."""


def _from_template(name: str, *, notes: str, minimum: float | None = None, maximum: float | None = None,
                   expert: bool = True) -> Hyperparameter:
    """A hyperparameter of ``hyp_det.yaml``, its type and default taken from the template."""
    default = _TEMPLATE[name]
    return Hyperparameter(name=name, type=_TYPES[type(default)], default=default, minimum=minimum, maximum=maximum,
                          notes=notes, expert=expert)


def _probability(name: str, notes: str) -> Hyperparameter:
    return _from_template(name, notes=notes, minimum=0, maximum=1)


def _pairs(name: str, notes: str, *, expert: bool = True) -> Hyperparameter:
    """A ``"key:value,..."`` hyperparameter of ``hyp_det.yaml``, checked by the node library before a training starts."""
    return Hyperparameter(name=name, type='pairs', default=_TEMPLATE[name], notes=notes, expert=expert)


HYPERPARAMETERS = [
    Hyperparameter(name='resolution', type='int', default=832, minimum=2 * MIN_STRIDE,
                   notes="Square input size in pixels, a multiple of the model's largest stride"),
    _from_template('epochs', minimum=1, expert=False,
                   notes='Maximum number of epochs; early stopping usually ends the training earlier'),
    Hyperparameter(name='patience', type='int', default=300, minimum=0,
                   notes='Epochs without improvement before the training stops early, 0 disables early stopping'),
    Hyperparameter(name=REQUESTED_BATCH_SIZE, type='int', default=0, minimum=0,
                   notes='Upper bound for the measured batch size, 0 means 128'),
    # on or off rather than a probability, see yolov5_format.FLIP_PROBABILITY; the template keeps 0.0 for off
    Hyperparameter(name='fliplr', type='bool', default=False, notes='Flip half of the images horizontally'),
    Hyperparameter(name='flipud', type='bool', default=False, notes='Flip half of the images vertically'),
    _pairs('flip_label_pairs', 'Categories swapped by a horizontal flip, as "uuid:uuid,..."', expert=False),
    _probability('detect_nms_conf_thres', 'Minimum confidence of a detection, in validation and in detection'),
    _probability('detect_nms_iou_thres',
                 'IoU above which the weaker of two detections is dropped, in validation and in detection'),
    _from_template('lr0', notes='Initial learning rate', minimum=0, expert=False),
    _from_template('lrf', notes='Final learning rate, as a fraction of lr0', minimum=0),
    _from_template('momentum', notes='SGD momentum, Adam beta1', minimum=0, maximum=1),
    _from_template('weight_decay', notes='Weight decay of the optimizer', minimum=0),
    _from_template('warmup_epochs', minimum=0,
                   notes='Epochs over which the learning rate warms up, fractions are fine'),
    _from_template('warmup_momentum', notes='Momentum at the start of the warmup', minimum=0, maximum=1),
    _from_template('warmup_bias_lr', notes='Learning rate of the biases at the start of the warmup', minimum=0),
    _from_template('box', notes='Box loss gain', minimum=0),
    _from_template('cls', notes='Class loss gain', minimum=0),
    _from_template('cls_pw', notes='Class BCE loss positive weight', minimum=0),
    _from_template('obj', notes='Objectness loss gain', minimum=0),
    _from_template('obj_pw', notes='Objectness BCE loss positive weight', minimum=0),
    _from_template('iou_t', notes='Unused by this YOLOv5 version; matching uses anchor_t', minimum=0, maximum=1),
    _from_template('anchor_t', notes='Largest size ratio between a label and an anchor that still matches', minimum=0),
    _from_template('fl_gamma', notes='Focal loss gamma, 0 disables it', minimum=0),
    _probability('hsv_h', 'Hue augmentation, as a fraction'),
    _probability('hsv_s', 'Saturation augmentation, as a fraction'),
    _probability('hsv_v', 'Value augmentation, as a fraction'),
    _from_template('degrees', notes='Image rotation, +/- degrees'),
    _probability('translate', 'Image translation, +/- fraction'),
    # below 1, or the drawn scale factor 1 +/- scale can reach zero; upstream evolves it within 0.9 too
    _from_template('scale', notes='Image scale, +/- gain', minimum=0, maximum=0.9),
    _from_template('shear', notes='Image shear, +/- degrees'),
    _from_template('perspective', notes='Image perspective, +/- fraction, 0 to 0.001', minimum=0, maximum=0.001),
    _probability('mosaic', 'Probability of combining four images into one'),
    _probability('mixup', 'Probability of blending two mosaic images, only applied when mosaic is'),
    _probability('copy_paste', 'Copy-paste of segments; has no effect on box and point labels'),
    _from_template('reset_points', notes='Resize the point labels to point_sizes_by_id'),
    _pairs('point_sizes_by_id',
           f'Point size per category as a fraction of the image, "uuid:size,...", used by reset_points; '
           f'{DEFAULT_POINT_SIZE} if unset'),
    COMPUTE_VALIDATION_SET_HYPERPARAMETER,
]
"""What this trainer accepts, reported to the loop."""
