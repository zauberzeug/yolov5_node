"""Choosing the training batch size by probing.

`train_det.py` calls :func:`measure` when it is given ``--batch-size -1``, on the model it has just
built and before it builds anything that depends on the batch size. The probe runs a step
resembling the one the training runs: EMA copy, three-group optimizer, the real ``ComputeLoss``,
backward, clipping, optimizer step.

Validation is measured as well, as ``train_det.py`` runs it between epochs: the EMA copy at
``batch_size // 2``, on the padded rectangular shape of the validation loader, in half precision
under AMP, with the loss and NMS of ``val.run``.
"""
import logging
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from learning_loop_node.trainer.batch_size import MIN_TRAIN_STEPS_PER_EPOCH
from learning_loop_node.trainer.cuda import ProbeStep, measure_batch_size
from learning_loop_node.trainer.exceptions import CriticalError

from .training_recipe import NOMINAL_BATCH_SIZE, VAL_PAD, scale_loss_weights
from .yolov5.models.yolo import Model
from .yolov5.utils.general import non_max_suppression
from .yolov5.utils.loss import ComputeLoss
from .yolov5.utils.torch_utils import ModelEMA, smart_inference_mode, smart_optimizer

logger = logging.getLogger(__name__)

PROBE = 'batch-size probe'

MIN_BATCH_SIZE = 2
"""Smallest batch a training may use; validation runs at half of it."""

DEFAULT_MAX_BATCH_SIZE = 128
"""Upper bound for a training that sets no `max_batch_size`."""

MIN_STRIDE = 32
"""The smallest grid size `train_det.py` uses, whatever the model."""


class InvalidResolutionError(CriticalError):
    pass


def check_resolution(img_size: object, stride: int) -> None:
    """The input size must be a multiple of the model's largest stride and at least twice that stride."""
    if not isinstance(img_size, int) or isinstance(img_size, bool) or img_size < 2 * stride or img_size % stride:
        raise InvalidResolutionError(
            f'invalid resolution {img_size!r}: must be a multiple of {stride} and at least {2 * stride}')


@dataclass(kw_only=True, slots=True)
class TrainingSetup:
    """What `train_det.py` has settled before it needs a batch size, and the probe mirrors."""

    model: Model
    """The training's model, with its weights; the probe holds it off the card while it runs on a copy."""
    device: torch.device
    amp: bool
    hyp: dict
    img_size: int
    """A multiple of `grid_size`, see :func:`check_resolution`."""
    grid_size: int
    categories: int
    optimizer: str
    """The `--optimizer` `train_det.py` builds."""


def measure(setup: TrainingSetup, *, train_path: str, max_batch_size: int) -> int:
    """Return the largest batch size the training fits into, within ``max_batch_size``.

    :param train_path: The `train/` folder `yolov5_format` filled.
    :param max_batch_size: The bound the training asked for; 0 means :data:`DEFAULT_MAX_BATCH_SIZE`.
    :raises InsufficientMemoryError: If not even :data:`MIN_BATCH_SIZE` fits.
    """
    sample_count = _train_sample_count(train_path)
    targets_per_image = _targets_per_sample(train_path)
    batch_size = measure_batch_size(lambda: TrainingStep(setup, targets_per_image),
                                    max_batch_size=max_batch_size or DEFAULT_MAX_BATCH_SIZE,
                                    sample_count=max(sample_count, NOMINAL_BATCH_SIZE * MIN_TRAIN_STEPS_PER_EPOCH),
                                    probe=PROBE, minimum=MIN_BATCH_SIZE)
    logger.info('%s: training at %d px with batch size %d', PROBE, setup.img_size, batch_size)
    return batch_size


class TrainingStep(ProbeStep):
    """One training cycle, as close to what :mod:`train_det` runs as a synthetic batch gets.

    Built once by `measure_batch_size` and run at several batch sizes: model copy, EMA copy and
    optimizer state are resident before the first batch, and only the activations scale with it.
    The training's own model waits on the CPU meanwhile; :meth:`release` puts it back.
    """

    def __init__(self, setup: TrainingSetup, targets_per_image: int) -> None:
        """:param targets_per_image: Boxes per synthetic image, see :func:`_targets_per_sample`."""
        self.training_model = setup.model.cpu()
        self.device = setup.device
        self.amp = setup.amp
        self.img_size = setup.img_size
        self.val_img_size = math.ceil(setup.img_size / setup.grid_size + VAL_PAD) * setup.grid_size
        self.categories = setup.categories
        self.targets_per_image = targets_per_image

        hyp = dict(setup.hyp)
        scale_loss_weights(hyp, layers=setup.model.model[-1].nl, categories=setup.categories, img_size=setup.img_size)

        self.model = deepcopy(setup.model).to(self.device)
        self.model.hyp = hyp
        self.ema = ModelEMA(self.model)
        self.optimizer = smart_optimizer(self.model, setup.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)  # resident from the first trial on, see `zero_gradients`
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        self.compute_loss = ComputeLoss(self.model)

    def train_step(self, batch_size: int) -> str:
        images = torch.rand(batch_size, 3, self.img_size, self.img_size, device=self.device)
        targets = self._targets(batch_size)

        with torch.amp.autocast('cuda', enabled=self.amp):
            loss, _ = self.compute_loss(self.model(images), targets)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.zero_gradients()
        self.ema.update(self.model)
        return f'{self.img_size} px, {self.targets_per_image} boxes per image, {"mixed precision" if self.amp else "float32"}'

    @smart_inference_mode()
    def val_step(self, batch_size: int) -> str:
        """One validation batch: square images, the largest shape the rectangular loader pads to."""
        val_batch_size = max(batch_size // 2, 1)
        model = self.ema.ema
        if self.amp:
            model.half()
        images = torch.rand(val_batch_size, 3, self.val_img_size, self.val_img_size, device=self.device)
        preds, train_out = model(images.half() if self.amp else images)
        self.compute_loss(train_out, self._targets(val_batch_size))
        non_max_suppression(preds, conf_thres=0.001, iou_thres=0.6, multi_label=True)
        model.float()
        return f'validation at {val_batch_size}, {self.val_img_size} px'

    def zero_gradients(self) -> None:
        """Clear the gradients but keep their memory, so every trial starts from the same state."""
        self.optimizer.zero_grad(set_to_none=False)

    def on_out_of_memory(self) -> None:
        self.zero_gradients()

    def release(self) -> None:
        del self.compute_loss, self.scaler, self.optimizer, self.ema, self.model
        self.training_model.to(self.device)

    def _targets(self, batch_size: int) -> torch.Tensor:
        """``[image_index, class, cx, cy, w, h]`` per box, normalised, as the dataloader yields."""
        count = batch_size * self.targets_per_image
        targets = torch.zeros(count, 6, device=self.device)
        targets[:, 0] = torch.arange(batch_size, device=self.device).repeat_interleave(self.targets_per_image)
        targets[:, 1] = torch.randint(0, self.categories, (count,), device=self.device)
        targets[:, 2:4] = torch.rand(count, 2, device=self.device) * 0.6 + 0.2  # centres, away from the border
        targets[:, 4:6] = torch.rand(count, 2, device=self.device) * 0.2 + 0.05
        return targets


def _train_sample_count(train_path: str) -> int:
    """How many images the training will see per epoch.

    `yolov5_format` symlinks them into `train/` as `<id>.jpg`, with the labels beside them as
    `<id>.txt`.
    """
    return sum(1 for path in Path(train_path).iterdir() if path.suffix == '.jpg')


def _targets_per_sample(train_path: str) -> int:
    """The most boxes a training sample can carry: a mosaic of four images as dense as the densest.

    `yolov5_format` writes one line per box or point into `train/<id>.txt`.
    """
    return 4 * max((len(path.read_text().splitlines()) for path in Path(train_path).glob('*.txt')), default=0)
