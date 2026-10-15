"""What the training in `train_det.py` is made of, where the batch-size probe has to measure the same.

`train_det.py` and :mod:`.batch_size_calculation` both read it from here.
"""

NOMINAL_BATCH_SIZE = 64
"""The batch `train_det.py` accumulates gradients up to (its `nbs`)."""

VAL_PAD = 0.5
"""The padding, in strides, of the validation loader's rectangular batches."""


def scale_loss_weights(hyp: dict, *, layers: int, categories: int, img_size: int) -> None:
    """Scale the box, class and objectness gains in ``hyp`` to the detection layers, classes and image size."""
    hyp['box'] *= 3 / layers
    hyp['cls'] *= categories / 80 * 3 / layers
    hyp['obj'] *= (img_size / 640) ** 2 * 3 / layers
