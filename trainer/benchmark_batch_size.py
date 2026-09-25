"""Re-measure the batch-size decision that `app_code.batch_size_calculation` now makes by probing.

The trainer used to read `Estimated Total Size (MB)` off `torchinfo.summary()` and compare it
against 95 % of the free memory, without ever running a step. :func:`legacy_estimate` keeps that
method alive here, gone from the trainer, so the comparison can be re-run.

Three numbers per resolution: what the estimate picked, whether that pick survives a real training
step, and what the probe picks now.

Run it inside the trainer image on a GPU box. The image is built without the dev group, so
`torchinfo`, which only this script needs, is installed first::

    docker run --rm --device nvidia.com/gpu=all -v "$PWD/benchmark_batch_size.py:/app/benchmark_batch_size.py" \
        -w /app zauberzeug/yolov5-trainer:latest \
        sh -c 'uv pip install --python /uv_venv/bin/python torchinfo && python /app/benchmark_batch_size.py'

Run it on an idle card. Two of these in parallel measure each other's memory, not the model's.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from learning_loop_node.helpers.misc import get_free_memory_mb
from learning_loop_node.trainer.batch_size import MIN_TRAIN_STEPS_PER_EPOCH, REQUESTED_BATCH_SIZE
from learning_loop_node.trainer.cuda import free_cuda_memory, measured_fits
from torchinfo import Verbosity, summary

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app_code.batch_size_calculation import TrainingStep, calc
from app_code.yolov5.models.yolo import Model
from app_code.yolov5.utils.downloads import attempt_download

logger = logging.getLogger(__name__)

DEFAULT_RESOLUTIONS = [320, 416, 640, 800, 1024]
LEGACY_CANDIDATES = [128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1]
"""The fixed, descending list the estimate walked; note the non-powers of two."""

LEGACY_FRACTION = 0.95
"""Share of the free memory the estimate allowed itself."""


@dataclass(kw_only=True, slots=True)
class Spec:
    """What it takes to rebuild the model under test, unchanged across the sweep."""

    weights: str
    hyp: dict
    categories: int
    training_path: str


@dataclass(kw_only=True, slots=True)
class Row:
    """One resolution, measured three ways."""

    resolution: int
    estimated: int | None
    estimated_holds: bool | None
    """Whether one real step at `estimated` fits."""
    probed: int | None
    error: str | None = None


def legacy_estimate(spec: Spec, resolution: int) -> int | None:
    """The method this branch replaced: the first candidate whose torchinfo size fits the card."""
    ckpt = torch.load(spec.weights, map_location='cuda', weights_only=False)
    model = Model(ckpt['model'].yaml, ch=3, nc=spec.categories, anchors=spec.hyp.get('anchors')).to('cuda')
    del ckpt
    budget_mb = get_free_memory_mb() * LEGACY_FRACTION
    try:
        for batch_size in LEGACY_CANDIDATES:
            try:
                stats = summary(model, input_size=(batch_size, 3, resolution, resolution), verbose=Verbosity.QUIET)
            except RuntimeError:
                continue
            size_mb = float(str(stats).split('Estimated Total Size (MB): ')[1].split('\n')[0])
            if size_mb < budget_mb:
                logger.info('estimate: %d px, batch size %d, %.0f of %.0f MB', resolution, batch_size,
                            size_mb, budget_mb)
                return batch_size
        return None
    finally:
        del model
        free_cuda_memory()


def holds(spec: Spec, resolution: int, batch_size: int) -> bool:
    """Whether one real step at `batch_size` fits, measured without a safety margin."""
    step = TrainingStep(spec.weights, spec.training_path, spec.hyp, spec.categories, resolution)
    try:
        return measured_fits(step, probe=f'verify {resolution}px', on_out_of_memory=step.zero_gradients)(batch_size)
    finally:
        step.release()


def measure(args: argparse.Namespace, workdir: Path) -> list[Row]:
    spec = Spec(weights=args.model, hyp=yaml.safe_load(Path(args.hyp).read_text()),
                categories=args.categories, training_path=str(workdir))
    write_dataset_yaml(workdir / 'dataset.yaml', args.categories)
    shutil.copy(args.hyp, workdir / 'hyp.yaml')  # where `calc` reads it, as in a training folder
    write_train_folder(workdir / 'train', args.limit * MIN_TRAIN_STEPS_PER_EPOCH)
    attempt_download(spec.weights)

    rows: list[Row] = []
    for resolution in args.resolutions:
        logger.info('=== %d px ===', resolution)
        row = Row(resolution=resolution, estimated=None, estimated_holds=None, probed=None)
        try:
            row.estimated = legacy_estimate(spec, resolution)
            if row.estimated is not None:
                row.estimated_holds = holds(spec, resolution, row.estimated)
            free_cuda_memory()
            row.probed = calc(spec.training_path, spec.weights, img_size=resolution, max_batch_size=args.limit)
        except Exception as exc:  # the sweep continues with the next resolution
            row.error = f'{type(exc).__name__}: {exc}'
        free_cuda_memory()
        rows.append(row)
        logger.info('%d px: estimated %s (holds: %s), probed %s',
                    resolution, row.estimated, row.estimated_holds, row.probed)
    return rows


def write_dataset_yaml(path: Path, categories: int) -> None:
    """Only `nc` is read back out of the dataset description."""
    path.write_text(yaml.safe_dump({'nc': categories, 'names': [f'class_{i}' for i in range(categories)]}))


def write_train_folder(path: Path, count: int) -> None:
    """Enough empty `.jpg` entries that the dataset bound `calc` derives never binds.

    The files stay empty; the probe's images are synthetic and nothing opens them.
    """
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (path / f'{i}.jpg').touch()


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
        if row.error:
            print(f'{row.resolution} px: {row.error}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--resolutions', type=int, nargs='+', default=DEFAULT_RESOLUTIONS)
    parser.add_argument('--model', default='yolov5s.pt', help='weights the trainer starts from')
    parser.add_argument('--categories', type=int, default=10, help='number of classes to build the head for')
    parser.add_argument('--hyp', default=str(Path(__file__).resolve().parent / 'hyp_det.yaml'))
    parser.add_argument('--limit', type=int, default=512,
                        help=f'upper bound, passed to the probe as the {REQUESTED_BATCH_SIZE} hyperparameter')
    parser.add_argument('--json', help='write the rows to this file as well')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if not torch.cuda.is_available():
        raise SystemExit('no CUDA device -- this benchmark only says something on a GPU box')
    logger.info('%s, %.1f GB', torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1024**3)

    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        rows = measure(args, Path(tmp))
    with contextlib.suppress(OSError):
        os.chdir(cwd)  # calc chdirs to /tmp for attempt_download and never comes back

    print_table(rows)
    if args.json:
        Path(args.json).write_text(json.dumps([asdict(row) for row in rows], indent=2))


if __name__ == '__main__':
    main()
