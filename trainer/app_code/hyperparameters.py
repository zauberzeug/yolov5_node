"""What this trainer accepts from the loop, declared once.

Until now these knobs took a detour: the loop's values were written into ``hyp_det.yaml`` by
``set_hyperparameters_in_file`` and read back out again by the trainer. That round-trip only ever
copied keys the template already had, so a knob missing from the template was dropped in silence,
and the template doubled as the place its default was stated.

The knobs below are read straight from the loop's values instead. Everything still in
``hyp_det.yaml`` is a hyperparameter the vendored yolov5 training reads for itself, which is why
that file keeps its own defaults and :func:`merge_into_yaml` keeps writing into it.
"""

from learning_loop_node.trainer.hyperparameters import (
                                                       DETECT_NMS_CONF_THRES,
                                                       DETECT_NMS_IOU_THRES,
                                                       Float,
                                                       IdValueMap,
                                                       Int,
                                                       Parameter,
)

HYPERPARAMETERS: list[Parameter] = [
    Int('resolution', required=True, description='Training and inference image size.'),
    Int('epochs', default=2000, minimum=1,
        description='Passed to the training as --epochs; early stopping usually ends a run first.'),
    Float('detect_nms_conf_thres', default=DETECT_NMS_CONF_THRES, minimum=0.0, maximum=1.0,
          description='Minimum confidence, for validation scores as well as the detection pass.'),
    Float('detect_nms_iou_thres', default=DETECT_NMS_IOU_THRES, minimum=0.0, maximum=1.0,
          description='Maximum IoU between two detections of one category.'),
    IdValueMap('point_sizes_by_id', value_kind='float',
               description='Per-category point size, as a fraction of the image size.'),
    IdValueMap('flip_label_pairs',
               description='Category uuid pairs whose labels swap on a horizontal flip.'),
]
"""The knobs this node reads itself.

`patience` is deliberately absent: the training is started with a fixed one, so a value
configured in the loop is not read. It therefore shows up in the warning
:func:`merge_into_yaml` logs, which is the truth and the point.
"""

DETECTION_HYPERPARAMETERS = [p for p in HYPERPARAMETERS if p.name != 'resolution']
"""What the detection pass reads. Auto-detections run against a finished model, so requiring the
training resolution here would refuse a detection pass over a training that never set one."""

NAMES = tuple(parameter.name for parameter in HYPERPARAMETERS)
"""Handled here rather than in the yaml template -- see :func:`merge_into_yaml`'s ``ignore``."""
