"""Choosing the training batch size by probing, rather than by estimating one.

The training runs in a subprocess (:mod:`train_det`), so the probe builds its own model here and
measures a step resembling the one that subprocess runs: EMA copy, three-group SGD, mixed
precision on the same condition, the real ``ComputeLoss``, backward, clipping, optimizer step.

The probe runs in a subprocess of its own (``probe_batch_size.py``), never in the node: a CUDA
context lives as long as its process, so a probe in the node would leave one behind for the whole
training. ``train_det.py`` would then run beside two contexts where the probe measured beside one,
and the difference comes out of the safety margin. It would also block the node's event loop for
as long as the probe runs.

Validation is not measured: ``train_det.py`` validates at ``batch_size // 2``, in half precision
and without gradients, so the training step is the peak.
"""
import logging
import os
import sys
from pathlib import Path
from typing import cast

import torch
import yaml
from learning_loop_node.trainer.cuda import free_cuda_memory, limit_cuda_memory, measure_batch_size

from .yolov5.models.yolo import Model
from .yolov5.utils.downloads import attempt_download
from .yolov5.utils.general import check_amp, check_img_size, intersect_dicts
from .yolov5.utils.loss import ComputeLoss
from .yolov5.utils.torch_utils import ModelEMA, smart_optimizer

YOLOV5_ROOT = Path(__file__).resolve().parent / 'yolov5'
"""What `train_det.py` puts on `sys.path`, because yolov5 imports its own packages by bare name."""

logger = logging.getLogger(__name__)

PROBE = 'batch-size probe'

MIN_BATCH_SIZE = 2
"""Smallest batch a training may use, because validation halves it."""

TARGETS_PER_IMAGE = 8
"""Boxes per synthetic image; the loss allocates per target, so this is not free."""


def calc(training_path: str, model_file: str, *, img_size: int,
         max_batch_size: int, vram_limit_gb: float) -> int:
    """Return the largest batch size a training step fits into, within ``max_batch_size``.

    Initialises CUDA in the calling process, so call it only where that process ends with the
    probe (``probe_batch_size.py``), never in the node.

    :param training_path: The training folder, holding the `dataset.yaml` and `hyp.yaml` that
        `train_det.py` reads.
    :param img_size: The `resolution` hyperparameter, rounded here as `train_det.py` rounds it.
    :param max_batch_size: The bound the training asked for; 0 lets the card decide. The caller
        reports the size this returns as `batch_size`.
    :param vram_limit_gb: Gigabytes of the card this training may use; 0 means the whole card.
        `train_det.py` is given the same number, because the cap set here does not survive a spawn.
    :raises InsufficientMemoryError: If not even :data:`MIN_BATCH_SIZE` fits.
    """
    sample_count = _train_sample_count(training_path)
    os.chdir('/tmp')  # NOTE: attempt_download writes the weights into the working directory

    with open(f'{training_path}/hyp.yaml') as f:
        hyp = yaml.safe_load(f)
    with open(f'{training_path}/dataset.yaml') as f:
        dataset = yaml.safe_load(f)

    attempt_download(model_file)  # Download pretrained yolov5 model from ultralytics to .pt

    torch.cuda.init()
    limit_cuda_memory(vram_limit_gb)
    free_cuda_memory()

    step = TrainingStep(model_file, training_path, hyp, dataset.get('nc'), img_size)
    try:
        batch_size = measure_batch_size(step, batch_size=max_batch_size,
                                        sample_count=sample_count, probe=PROBE,
                                        minimum=MIN_BATCH_SIZE, vram_limit_gb=vram_limit_gb,
                                        on_out_of_memory=step.zero_gradients)
    finally:
        step.release()

    logger.info('%s: training at %d px with batch size %d', PROBE, step.img_size, batch_size)
    return batch_size


class TrainingStep:
    """One training step, as close to what :mod:`train_det` runs as a synthetic batch gets.

    Built once and called at several batch sizes: model, EMA copy and optimizer state are resident
    before the first batch, and only the activations scale with it.
    """

    def __init__(self, model_file: str, training_path: str, hyp: dict, categories: int, img_size: int) -> None:
        self.categories = categories
        self.device = torch.device('cuda', 0)

        try:
            ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
        except FileNotFoundError:
            ckpt = torch.load(f'{training_path}/{model_file}', map_location='cpu', weights_only=False)
        self.model = Model(ckpt['model'].yaml, ch=3, nc=categories, anchors=hyp.get('anchors')).to(self.device)
        # NOTE: the checkpoint's weights, as `train_det.py` transfers them; `check_amp` compares
        # detections, and a randomly initialised model detects nothing either way
        exclude = ['anchor'] if hyp.get('anchors') else []
        weights = intersect_dicts(ckpt['model'].float().state_dict(), self.model.state_dict(), exclude=exclude)
        self.model.load_state_dict(weights, strict=False)
        del ckpt, weights
        self.amp = _amp_enabled(self.model)  # before the EMA and optimizer: `check_amp` copies the model

        gs = max(int(self.model.stride.max()), 32)  # type: ignore[union-attr]  # grid size (max stride)
        # as `train_det.py` rounds `--img`, so 600 is 608; an int in, an int out
        self.img_size = cast(int, check_img_size(img_size, gs, floor=gs * 2))

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

    def __call__(self, batch_size: int) -> str:
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
        return f'{self.img_size} px, {"mixed precision" if self.amp else "float32"}'

    def zero_gradients(self) -> None:
        """Clear the gradients but keep their memory, so every trial starts from the same state.

        `train_det.py` accumulates over ``round(64 / batch_size)`` steps, so from its second step
        on every forward runs with the gradients resident. Also the handler for a trial that ran
        out of memory.
        """
        self.optimizer.zero_grad(set_to_none=False)

    def release(self) -> None:
        del self.compute_loss, self.scaler, self.optimizer, self.ema, self.model
        free_cuda_memory()

    def _targets(self, batch_size: int) -> torch.Tensor:
        """``[image_index, class, cx, cy, w, h]`` per box, normalised, as the dataloader yields."""
        count = batch_size * TARGETS_PER_IMAGE
        targets = torch.zeros(count, 6, device=self.device)
        targets[:, 0] = torch.arange(batch_size, device=self.device).repeat_interleave(TARGETS_PER_IMAGE)
        targets[:, 1] = torch.randint(0, self.categories, (count,), device=self.device)
        targets[:, 2:4] = torch.rand(count, 2, device=self.device) * 0.6 + 0.2  # centres, away from the border
        targets[:, 4:6] = torch.rand(count, 2, device=self.device) * 0.2 + 0.05
        return targets


def _train_sample_count(training_path: str) -> int:
    """How many images the training will see per epoch.

    `yolov5_format` symlinks them into `train/` as `<id>.jpg`, with the labels beside them as
    `<id>.txt`.
    """
    return sum(1 for path in (Path(training_path) / 'train').iterdir() if path.suffix == '.jpg')


def _amp_enabled(model: Model) -> bool:
    """Whether `train_det.py` will train in mixed precision, asked the way it asks.

    `check_amp` reaches for the yolov5 packages by bare name, hence the `sys.path` entry. A check
    that cannot run answers no, so the probe measures float32 -- more memory than AMP needs, never
    less.
    """
    if str(YOLOV5_ROOT) not in sys.path:
        sys.path.append(str(YOLOV5_ROOT))
    try:
        return bool(check_amp(model))
    except Exception:  # pylint: disable=broad-except
        logger.exception('%s: could not determine whether AMP is usable; measuring in float32', PROBE)
        return False
