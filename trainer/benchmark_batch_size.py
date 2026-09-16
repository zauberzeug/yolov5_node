"""Compare the estimated batch size against a measured one, and check whether the estimate holds.

The trainer picks its batch size in :mod:`app_code.batch_size_calculation`, which never runs a
step: it reads a size off ``torchinfo.summary()`` and compares it against the free memory. The
library offers the alternative -- :func:`learning_loop_node.trainer.cuda.probe_batch_size`, which
doubles until a real step runs out of memory.

This script runs both against the same model and resolution and prints what each picks, plus
whether the estimate's pick survives one real training step. It changes nothing about the
trainer; it produces the table the decision needs.

Run it inside the trainer image on a GPU box, where the dependencies already are::

    docker run --rm --device nvidia.com/gpu=all -v "$PWD:/bench" -w /app \
        zauberzeug/yolov5-trainer:latest python /bench/benchmark_batch_size.py

A step is built to resemble :mod:`train_det` rather than to be cheap: the model, its EMA copy,
the three-group SGD optimizer, AMP with a gradient scaler, the real ``ComputeLoss``, backward,
gradient clipping and an optimizer step. Images and targets are synthetic -- memory does not
depend on what the pixels show, only on their shape.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from learning_loop_node.trainer.batch_size import find_batch_size, smaller_pot
from learning_loop_node.trainer.cuda import free_cuda_memory, measured_fits, reserve_margin

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app_code import batch_size_calculation
from app_code.yolov5.models.yolo import Model
from app_code.yolov5.utils.loss import ComputeLoss
from app_code.yolov5.utils.torch_utils import ModelEMA, smart_optimizer

DEFAULT_RESOLUTIONS = [416, 640, 800, 1024]
TARGETS_PER_IMAGE = 8
"""Boxes per synthetic image; the loss allocates per target, so this is not free."""


@dataclass
class ModelSpec:
    """What it takes to rebuild the model under test, unchanged across the sweep."""

    weights: str
    hyp: dict
    categories: int


@dataclass
class Row:
    """One resolution, measured three ways."""

    resolution: int
    estimated: int | None
    estimated_error: str | None
    estimated_holds: bool | None
    """Whether one real step at ``estimated`` fits -- the question the estimate never asks."""
    probed: int | None
    probed_error: str | None


class TrainingStep:
    """One training step, as close to ``train_det.py`` as a synthetic batch gets.

    Built once and called at several batch sizes, the way a training builds its model once and
    then steps: the model, the EMA copy and the optimizer state are already resident when the
    first batch arrives, and only the activations scale with the batch size.
    """

    def __init__(self, spec: ModelSpec, img_size: int, device: torch.device) -> None:
        self.device = device
        self.img_size = img_size
        self.nc = nc = spec.categories
        hyp = spec.hyp

        ckpt = torch.load(spec.weights, map_location=device, weights_only=False)
        self.model = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)
        del ckpt

        nl = self.model.model[-1].nl  # detection layers, the loss gains are scaled per layer
        hyp = dict(hyp)
        hyp['box'] *= 3 / nl
        hyp['cls'] *= nc / 80 * 3 / nl
        hyp['obj'] *= (img_size / 640) ** 2 * 3 / nl

        self.model.nc = nc
        self.model.hyp = hyp
        self.model.train()

        self.ema = ModelEMA(self.model)
        self.optimizer = smart_optimizer(self.model, 'SGD', hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.compute_loss = ComputeLoss(self.model)

    def __call__(self, batch_size: int) -> str:
        images = torch.rand(batch_size, 3, self.img_size, self.img_size, device=self.device)
        targets = self._targets(batch_size)

        with torch.cuda.amp.autocast(True):
            loss, _ = self.compute_loss(self.model(images), targets)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.ema.update(self.model)
        return f'{self.img_size} px'

    def _targets(self, batch_size: int) -> torch.Tensor:
        """``[image_index, class, cx, cy, w, h]`` per box, normalised, as the dataloader yields."""
        count = batch_size * TARGETS_PER_IMAGE
        targets = torch.zeros(count, 6, device=self.device)
        targets[:, 0] = torch.arange(batch_size, device=self.device).repeat_interleave(TARGETS_PER_IMAGE)
        targets[:, 1] = torch.randint(0, self.nc, (count,), device=self.device)
        targets[:, 2:4] = torch.rand(count, 2, device=self.device) * 0.6 + 0.2  # centres, away from the border
        targets[:, 4:6] = torch.rand(count, 2, device=self.device) * 0.2 + 0.05
        return targets

    def drop_gradients(self) -> None:
        """Release what a failed step left behind, so the next trial starts from the same state."""
        self.optimizer.zero_grad(set_to_none=True)

    def release(self) -> None:
        del self.compute_loss, self.scaler, self.optimizer, self.ema, self.model
        free_cuda_memory()


def estimate(training_path: str, model_file: str, *, hyp_path: str, dataset_path: str, resolution: int) -> int:
    """What the trainer picks today."""
    return asyncio.run(
        batch_size_calculation.calc(training_path, model_file, hyp_path, dataset_path, resolution))


def probe(spec: ModelSpec, resolution: int, *, limit: int, vram_limit_gb: float) -> int:
    """What a probe measures.

    Composed rather than calling :func:`probe_batch_size`, because the model outlives the trials
    here and a failed trial has to drop its gradients before the next one -- the case the
    library's ``on_out_of_memory`` hook exists for.
    """
    step = TrainingStep(spec, resolution, torch.device('cuda', 0))
    margin = reserve_margin(vram_limit_gb, probe=f'probe {resolution}px')
    try:
        fits = measured_fits(step, probe=f'probe {resolution}px', on_out_of_memory=step.drop_gradients)
        return find_batch_size(fits, limit=smaller_pot(limit))
    finally:
        del margin
        step.release()


def holds(spec: ModelSpec, resolution: int, batch_size: int) -> bool:
    """Whether one real step at ``batch_size`` fits -- with no margin, the benefit of the doubt."""
    step = TrainingStep(spec, resolution, torch.device('cuda', 0))
    try:
        return measured_fits(step, probe=f'verify {resolution}px', on_out_of_memory=step.drop_gradients)(batch_size)
    finally:
        step.release()


def write_dataset_yaml(path: Path, nc: int) -> None:
    """The estimate reads only ``nc`` out of the dataset description."""
    path.write_text(yaml.safe_dump({'nc': nc, 'names': [f'class_{i}' for i in range(nc)]}))


def measure(args: argparse.Namespace, workdir: Path) -> list[Row]:
    spec = ModelSpec(weights=args.model, hyp=yaml.safe_load(Path(args.hyp).read_text()),
                     categories=args.categories)
    dataset_path = workdir / 'dataset.yaml'
    write_dataset_yaml(dataset_path, args.categories)

    rows: list[Row] = []
    for resolution in args.resolutions:
        logging.info('=== %d px ===', resolution)
        row = Row(resolution=resolution, estimated=None, estimated_error=None,
                  estimated_holds=None, probed=None, probed_error=None)

        try:
            row.estimated = estimate(str(workdir), args.model, hyp_path=args.hyp,
                                     dataset_path=str(dataset_path), resolution=resolution)
        except Exception as exc:  # the comparison continues without this cell
            row.estimated_error = f'{type(exc).__name__}: {exc}'
        free_cuda_memory()

        if row.estimated is not None:
            try:
                row.estimated_holds = holds(spec, resolution, row.estimated)
            except Exception as exc:
                row.estimated_error = f'verification failed -- {type(exc).__name__}: {exc}'
        free_cuda_memory()

        try:
            row.probed = probe(spec, resolution, limit=args.limit, vram_limit_gb=args.vram_limit_gb)
        except Exception as exc:
            row.probed_error = f'{type(exc).__name__}: {exc}'
        free_cuda_memory()

        rows.append(row)
        logging.info('%d px: estimated %s (holds: %s), probed %s',
                     resolution, row.estimated, row.estimated_holds, row.probed)
    return rows


def print_table(rows: list[Row]) -> None:
    def cell(value: object) -> str:
        return {None: '-', True: 'yes', False: 'NO'}.get(value, str(value))  # type: ignore[arg-type]

    print()
    print(f'| {"resolution":>10} | {"estimated":>9} | {"step fits":>9} | {"probed":>6} |')
    print(f'|{"-" * 12}|{"-" * 11}|{"-" * 11}|{"-" * 8}|')
    for row in rows:
        print(f'| {row.resolution:>10} | {cell(row.estimated):>9} | {cell(row.estimated_holds):>9} '
              f'| {cell(row.probed):>6} |')
    print()
    for row in rows:
        for label, error in (('estimate', row.estimated_error), ('probe', row.probed_error)):
            if error:
                print(f'{row.resolution} px, {label}: {error}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--resolutions', type=int, nargs='+', default=DEFAULT_RESOLUTIONS)
    parser.add_argument('--model', default='yolov5s.pt', help='weights the trainer starts from')
    parser.add_argument('--categories', type=int, default=10, help='number of classes to build the head for')
    parser.add_argument('--hyp', default=str(Path(__file__).resolve().parent / 'hyp_det.yaml'))
    parser.add_argument('--limit', type=int, default=256, help='upper bound of the probe search')
    parser.add_argument('--vram-limit-gb', type=float, default=0, help='0 means the whole card')
    parser.add_argument('--json', help='write the rows to this file as well')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not torch.cuda.is_available():
        raise SystemExit('no CUDA device -- this benchmark only says something on a GPU box')
    logging.info('%s, %.1f GB', torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1024**3)

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        rows = measure(args, Path(tmp))
    with contextlib.suppress(OSError):
        os.chdir(cwd)  # the estimate chdirs to /tmp and never comes back

    print_table(rows)
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(row) for row in rows], indent=2))


if __name__ == '__main__':
    main()
