"""Choosing the training batch size by probing, rather than by estimating one.

The training runs in a subprocess (:mod:`train_det`), so the probe builds its own model here and
measures a step resembling the one that subprocess runs: EMA copy, three-group SGD, mixed
precision on the same condition, the real ``ComputeLoss``, backward, clipping, optimizer step.

The probe runs in a subprocess of its own (``probe_batch_size.py``), never in the node: a CUDA
context lives as long as its process, so a probe in the node would leave one behind for the whole
training. ``train_det.py`` would then run beside two contexts where the probe measured beside one,
and the difference comes out of the safety margin. It would also block the node's event loop for
as long as the probe runs.

Validation is measured as well, as ``train_det.py`` runs it between epochs: the EMA copy at
``batch_size // 2``, on the padded rectangular shape of the validation loader, in half precision
under AMP, with the loss and NMS of ``val.run``.
"""
import logging
import math
from pathlib import Path
from typing import cast

import torch
import yaml
from learning_loop_node.trainer.batch_size import MIN_TRAIN_STEPS_PER_EPOCH
from learning_loop_node.trainer.cuda import ProbeStep, free_cuda_memory, limit_cuda_memory, measure_batch_size

from .yolov5.models.yolo import Model
from .yolov5.utils.downloads import attempt_download
from .yolov5.utils.general import check_amp, check_img_size, init_seeds, intersect_dicts, non_max_suppression
from .yolov5.utils.loss import ComputeLoss
from .yolov5.utils.torch_utils import ModelEMA, smart_inference_mode, smart_optimizer

logger = logging.getLogger(__name__)

PROBE = 'batch-size probe'

MIN_BATCH_SIZE = 2
"""Smallest batch a training may use, because validation halves it."""

DEFAULT_MAX_BATCH_SIZE = 128
"""Upper bound for a training that sets no `max_batch_size`."""

NOMINAL_BATCH_SIZE = 64
"""The batch `train_det.py` accumulates gradients up to, so a smaller one adds no optimizer steps."""

VAL_PAD = 0.5
"""The padding, in strides, `train_det.py` gives the validation loader's rectangular batches."""


def calc(training_path: str, model_file: str, *, img_size: int,
         max_batch_size: int, vram_limit_gb: float) -> int:
    """Return the largest batch size a training step fits into, within ``max_batch_size``.

    Initialises CUDA in the calling process, so call it only where that process ends with the
    probe (``probe_batch_size.py``), never in the node.

    :param training_path: The training folder, holding the `dataset.yaml` and `hyp.yaml` that
        `train_det.py` reads.
    :param img_size: The `resolution` hyperparameter, rounded here as `train_det.py` rounds it.
    :param max_batch_size: The bound the training asked for; 0 means :data:`DEFAULT_MAX_BATCH_SIZE`. The caller
        reports the size this returns as `batch_size`.
    :param vram_limit_gb: Gigabytes of the card this training may use; 0 means the whole card.
        The cap set here is also the budget the probe measures against. `train_det.py` is given
        the same number, because the cap does not survive a spawn.
    :raises InsufficientMemoryError: If not even :data:`MIN_BATCH_SIZE` fits.
    """
    sample_count = _train_sample_count(training_path)
    targets_per_image = _targets_per_sample(training_path)

    with open(f'{training_path}/hyp.yaml') as f:
        hyp = yaml.safe_load(f)
    with open(f'{training_path}/dataset.yaml') as f:
        dataset = yaml.safe_load(f)

    attempt_download(model_file)  # Download pretrained yolov5 model from ultralytics to .pt
    ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
    img_size = _training_img_size(ckpt['model'], img_size)

    init_seeds(1, deterministic=True)  # NOTE: as train_det.py seeds itself by default, so both use the same kernels
    torch.cuda.init()
    limit_cuda_memory(vram_limit_gb)
    free_cuda_memory()

    batch_size = measure_batch_size(lambda: TrainingStep(ckpt, hyp, dataset.get('nc'), img_size, targets_per_image),
                                    max_batch_size=max_batch_size or DEFAULT_MAX_BATCH_SIZE,
                                    sample_count=max(sample_count, NOMINAL_BATCH_SIZE * MIN_TRAIN_STEPS_PER_EPOCH),
                                    probe=PROBE, minimum=MIN_BATCH_SIZE)
    logger.info('%s: training at %d px with batch size %d', PROBE, img_size, batch_size)
    return batch_size


class TrainingStep(ProbeStep):
    """One training cycle, as close to what :mod:`train_det` runs as a synthetic batch gets.

    Built once by `measure_batch_size` and run at several batch sizes: model, EMA copy and
    optimizer state are resident before the first batch, and only the activations scale with it.

    :param img_size: Already rounded, see :func:`_training_img_size`.
    :param targets_per_image: Boxes per synthetic image, see :func:`_targets_per_sample`.
    """

    def __init__(self, ckpt: dict, hyp: dict, categories: int, img_size: int, targets_per_image: int) -> None:
        self.categories = categories
        self.img_size = img_size
        self.targets_per_image = targets_per_image
        self.device = torch.device('cuda', 0)

        self.model = Model(ckpt['model'].yaml, ch=3, nc=categories, anchors=hyp.get('anchors')).to(self.device)
        gs = max(int(self.model.stride.max()), 32)  # type: ignore[union-attr]  # grid size (max stride)
        self.val_img_size = math.ceil(img_size / gs + VAL_PAD) * gs
        # NOTE: the checkpoint's weights, as `train_det.py` transfers them; `check_amp` compares
        # detections, and a randomly initialised model detects nothing either way
        exclude = ['anchor'] if hyp.get('anchors') else []
        weights = intersect_dicts(ckpt['model'].float().state_dict(), self.model.state_dict(), exclude=exclude)
        self.model.load_state_dict(weights, strict=False)
        del weights
        self.amp = check_amp(self.model)  # before the EMA and optimizer: `check_amp` copies the model

        hyp = dict(hyp)
        nl = self.model.model[-1].nl  # detection layers
        hyp['box'] *= 3 / nl
        hyp['cls'] *= categories / 80 * 3 / nl
        hyp['obj'] *= (self.img_size / 640) ** 2 * 3 / nl

        self.model.nc = categories
        self.model.hyp = hyp
        self.model.train()

        self.ema = ModelEMA(self.model)
        self.optimizer = smart_optimizer(self.model, 'SGD', hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
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
        """Clear the gradients but keep their memory, so every trial starts from the same state.

        `train_det.py` accumulates over ``round(64 / batch_size)`` steps, so from its second step
        on every forward runs with the gradients resident.
        """
        self.optimizer.zero_grad(set_to_none=False)

    def on_out_of_memory(self) -> None:
        self.zero_gradients()

    def release(self) -> None:
        del self.compute_loss, self.scaler, self.optimizer, self.ema, self.model
        free_cuda_memory()

    def _targets(self, batch_size: int) -> torch.Tensor:
        """``[image_index, class, cx, cy, w, h]`` per box, normalised, as the dataloader yields."""
        count = batch_size * self.targets_per_image
        targets = torch.zeros(count, 6, device=self.device)
        targets[:, 0] = torch.arange(batch_size, device=self.device).repeat_interleave(self.targets_per_image)
        targets[:, 1] = torch.randint(0, self.categories, (count,), device=self.device)
        targets[:, 2:4] = torch.rand(count, 2, device=self.device) * 0.6 + 0.2  # centres, away from the border
        targets[:, 4:6] = torch.rand(count, 2, device=self.device) * 0.2 + 0.05
        return targets


def _training_img_size(model: Model, img_size: int) -> int:
    """The resolution as `train_det.py` rounds `--img` to the model's stride, so 600 is 608."""
    gs = max(int(model.stride.max()), 32)  # type: ignore[union-attr]  # grid size (max stride)
    return cast(int, check_img_size(img_size, gs, floor=gs * 2))  # an int in, an int out


def _train_sample_count(training_path: str) -> int:
    """How many images the training will see per epoch.

    `yolov5_format` symlinks them into `train/` as `<id>.jpg`, with the labels beside them as
    `<id>.txt`.
    """
    return sum(1 for path in (Path(training_path) / 'train').iterdir() if path.suffix == '.jpg')


def _targets_per_sample(training_path: str) -> int:
    """The most boxes a training sample can carry: a mosaic of four images as dense as the densest.

    The loss allocates per target, so a dense dataset needs memory a sparse one does not.
    `yolov5_format` writes one line per box or point into `train/<id>.txt`.
    """
    label_files = (Path(training_path) / 'train').glob('*.txt')
    return 4 * max((len(path.read_text().splitlines()) for path in label_files), default=0)
