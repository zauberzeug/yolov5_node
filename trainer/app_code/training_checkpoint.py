import json
import os
from pathlib import Path
from typing import Any

import torch

from .model_files import delete_newer_epochs


def save_best(checkpoint: dict[str, Any], folder: Path, confusion_matrices: dict[str, Any]) -> None:
    """Save the EMA weights as `best.pt` and, with metrics, as the `epoch<n>.pt` the node publishes.

    Must run before `last.pt` of the same epoch is written.
    """
    best = folder / 'best.pt'
    atomic_save({'epoch': checkpoint['epoch'], 'best_fitness': checkpoint['best_fitness'],
                 'model': checkpoint['ema'], 'ema': None, 'updates': None, 'optimizer': None,
                 'opt': checkpoint['opt'], 'date': checkpoint['date']}, best)
    if confusion_matrices:
        path = folder / f"epoch{checkpoint['epoch']}.pt"
        metrics = path.with_suffix('.json.tmp')
        metrics.write_text(json.dumps(confusion_matrices))
        metrics.replace(path.with_suffix('.json'))
        _atomic_link(best, path)


def discard_unfinished_epochs(folder: Path, last_epoch: int) -> None:
    """Remove the best checkpoints of an epoch whose `last.pt` was never written."""
    delete_newer_epochs(folder, last_epoch)
    best = folder / 'best.pt'
    if best.exists() and torch.load(best, map_location='cpu', weights_only=False)['epoch'] > last_epoch:
        best.unlink()


def restore_training_state(checkpoint: dict[str, Any], scheduler: Any, scaler: Any, stopper: Any) -> None:
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    if 'early_stopping' in checkpoint:
        stopper.best_epoch = checkpoint['early_stopping']['best_epoch']
        stopper.best_fitness = checkpoint['early_stopping']['best_fitness']
        stopper.possible_stop = checkpoint['early_stopping']['possible_stop']
    else:
        stopper.best_epoch = checkpoint['epoch']
        stopper.best_fitness = checkpoint['best_fitness']


def atomic_save(checkpoint: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_link(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    temporary.replace(destination)
