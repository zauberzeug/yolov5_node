"""Measure the training batch size in a process of its own, and write it to a JSON file.

Started by the node before `train_det.py`, through the same `Executor`. The process — and the CUDA
context it opened — ends with the probe, so the training runs beside no context but its own, as
the probe did. See :mod:`app_code.batch_size_calculation`.
"""
import argparse
import json
import logging
from pathlib import Path

from learning_loop_node.trainer.cuda import add_vram_limit_argument

from app_code.batch_size_calculation import calc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-path', required=True,
                        help='training folder holding dataset.yaml, hyp.yaml and train/')
    parser.add_argument('--weights', required=True, help='the checkpoint train_det.py starts from')
    parser.add_argument('--img', type=int, required=True, help='the resolution hyperparameter')
    parser.add_argument('--max-batch-size', type=int, default=0, help='upper bound; 0 lets the card decide')
    add_vram_limit_argument(parser)
    parser.add_argument('--output', required=True, help='JSON file the settled batch size is written to')
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parse_args()
    batch_size = calc(args.training_path, args.weights, img_size=args.img,
                      max_batch_size=args.max_batch_size, vram_limit_gb=args.vram_limit_gb)
    Path(args.output).write_text(json.dumps({'batch_size': batch_size}))


if __name__ == '__main__':
    main()
