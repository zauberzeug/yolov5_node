"""Choosing the training batch size by probing, rather than by estimating one.

The training runs in a subprocess (:mod:`train_det`), so the probe builds its own model here and
measures a step resembling the one that subprocess runs: the EMA copy, the three-group SGD
optimizer, AMP with a gradient scaler, the real ``ComputeLoss``, backward, gradient clipping and an
optimizer step. Everything it allocates is released before the subprocess starts.

Validation is deliberately not measured. ``train_det.py`` validates at ``batch_size // 2``, in half
precision and without gradients, so the training step is the peak. That same halving is why the
probe is given a :data:`MIN_BATCH_SIZE`: a batch of one would validate with a batch of zero.
"""
import logging
import os
from collections.abc import MutableMapping
from typing import Any

import torch
import yaml
from learning_loop_node.trainer.batch_size import BATCH_SIZE
from learning_loop_node.trainer.cuda import free_cuda_memory, measure_batch_size

from .yolov5.models.yolo import Model
from .yolov5.utils.downloads import attempt_download
from .yolov5.utils.loss import ComputeLoss
from .yolov5.utils.torch_utils import ModelEMA, smart_optimizer

PROBE = 'batch-size probe'
"""Names the probe in the log, so its lines are greppable next to the training's own."""

MIN_BATCH_SIZE = 2
"""Smallest batch a training may use, because validation halves it."""

TARGETS_PER_IMAGE = 8
"""Boxes per synthetic image; the loss allocates per target, so this is not free."""


async def calc(training_path: str, model_file: str, hyp_path: str, img_size: int,
               hyperparameters: MutableMapping[str, Any]) -> int:
    """Return the largest power-of-two batch size a training step fits into.

    :param training_path: The training folder, which is where `yolov5_format` wrote `dataset.yaml`.
    :param hyperparameters: The training's hyperparameters, which `measure_batch_size` reads the
        bound out of; the caller reports the size this returns.
    :raises InsufficientMemoryError: If not even :data:`MIN_BATCH_SIZE` fits.
    """
    os.chdir('/tmp')  # NOTE: attempt_download writes the weights into the working directory

    with open(hyp_path) as f:
        hyp = yaml.safe_load(f)
    with open(f'{training_path}/dataset.yaml') as f:
        dataset = yaml.safe_load(f)

    attempt_download(model_file)  # Download pretrained yolov5 model from ultralytics to .pt

    torch.cuda.init()
    free_cuda_memory()

    step = TrainingStep(model_file, training_path, hyp, dataset.get('nc'), img_size)
    try:
        batch_size = measure_batch_size(step, batch_size=int(hyperparameters.get(BATCH_SIZE, 0) or 0),
                                        probe=PROBE, minimum=MIN_BATCH_SIZE,
                                        on_out_of_memory=step.drop_gradients)
    finally:
        step.release()

    logging.info('%s: training at %d px with batch size %d', PROBE, img_size, batch_size)
    return batch_size


class TrainingStep:
    """One training step, as close to what :mod:`train_det` runs as a synthetic batch gets.

    Built once and called at several batch sizes, the way a training builds its model once and then
    steps: the model, the EMA copy and the optimizer state are already resident when the first
    batch arrives, and only the activations scale with the batch size.
    """

    def __init__(self, model_file: str, training_path: str, hyp: dict, categories: int, img_size: int) -> None:
        self.img_size = img_size
        self.categories = categories
        self.device = torch.device('cuda', 0)

        try:
            ckpt = torch.load(model_file, map_location=self.device, weights_only=False)
        except FileNotFoundError:
            ckpt = torch.load(f'{training_path}/{model_file}', map_location=self.device, weights_only=False)
        self.model = Model(ckpt['model'].yaml, ch=3, nc=categories, anchors=hyp.get('anchors')).to(self.device)
        del ckpt

        hyp = dict(hyp)
        nl = self.model.model[-1].nl  # detection layers; the loss gains are scaled per layer
        hyp['box'] *= 3 / nl
        hyp['cls'] *= categories / 80 * 3 / nl
        hyp['obj'] *= (img_size / 640) ** 2 * 3 / nl

        self.model.nc = categories
        self.model.hyp = hyp
        self.model.train()

        self.ema = ModelEMA(self.model)
        self.optimizer = smart_optimizer(self.model, 'SGD', hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
        self.scaler = torch.amp.GradScaler('cuda')
        self.compute_loss = ComputeLoss(self.model)

    def __call__(self, batch_size: int) -> str:
        images = torch.rand(batch_size, 3, self.img_size, self.img_size, device=self.device)
        targets = self._targets(batch_size)

        with torch.amp.autocast('cuda'):
            loss, _ = self.compute_loss(self.model(images), targets)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update(self.model)
        return f'{self.img_size} px'

    def drop_gradients(self) -> None:
        """Release what a failed step left behind, so the next trial starts from the same state."""
        self.optimizer.zero_grad(set_to_none=True)

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
