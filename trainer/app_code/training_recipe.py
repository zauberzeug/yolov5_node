"""What the training in `train_det.py` is made of, where the batch-size probe and the final evaluation in
`eval_det.py` have to do the same.

`train_det.py`, `eval_det.py` and :mod:`.batch_size_calculation` all read it from here.
"""
from learning_loop_node.trainer.hyperparameters import parse_pairs
from torch.utils.data import DataLoader

from .yolov5.utils.dataloaders import create_dataloader
from .yolov5.utils.general import colorstr

NOMINAL_BATCH_SIZE = 64
"""The batch `train_det.py` accumulates gradients up to (its `nbs`)."""

VAL_PAD = 0.5
"""The padding, in strides, of the validation loader's rectangular batches."""


def scale_loss_weights(hyp: dict, *, layers: int, categories: int, img_size: int) -> None:
    """Scale the box, class and objectness gains in ``hyp`` to the detection layers, classes and image size."""
    hyp['box'] *= 3 / layers
    hyp['cls'] *= categories / 80 * 3 / layers
    hyp['obj'] *= (img_size / 640) ** 2 * 3 / layers


def evaluation_loader(path: str, name: str, *, img_size: int, batch_size: int,  # noqa: PLR0913
                      stride: int, single_cls: bool, hyp: dict, workers: int,
                      point_sizes_by_id: dict[int, float],
                      cache: str | None = None) -> DataLoader:
    """The loader of the images a model is validated or tested on.

    :param name: The set, ``'val'`` or ``'test'``, as the log names it.
    :param batch_size: The batch size of the training; the loader takes half of it.
    """
    return create_dataloader(path, img_size, batch_size // 2, stride, single_cls, hyp=hyp, cache=cache, rect=True,
                             rank=-1, workers=workers * 2, pad=VAL_PAD, prefix=colorstr(f'{name}: '),
                             point_sizes_by_id=point_sizes_by_id)[0]


def parse_point_sizes(argument: str) -> dict[int, float]:
    """The point size per category index from ``--point_sizes_by_id``, e.g. ``"1:0.2,3:0.3"``."""
    return {int(index): float(size) for index, size in parse_pairs(str(argument or ''))}


def parse_flip_label_pairs(argument: str) -> list[tuple[int, int]]:
    """The category indices a horizontal flip swaps from ``--flip_label_pairs``, e.g. ``"1:2,3:4"``."""
    return [(int(left), int(right)) for left, right in parse_pairs(str(argument or ''))]

