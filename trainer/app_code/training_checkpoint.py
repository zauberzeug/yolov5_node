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


def resume_fields(scheduler: Any, scaler: Any, stopper: Any, last_opt_step: int, stopped_early: bool) -> dict[str, Any]:
    """The runtime state `last.pt` needs beyond upstream's checkpoint to continue at the next epoch."""
    return {
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'early_stopping': {'best_epoch': stopper.best_epoch, 'best_fitness': stopper.best_fitness,
                           'possible_stop': stopper.possible_stop},
        'last_opt_step': last_opt_step,
        'stopped_early': stopped_early,
    }


def finished(checkpoint: dict[str, Any], epochs: int) -> bool:
    return bool(checkpoint.get('stopped_early')) or checkpoint['epoch'] + 1 >= epochs


def resume_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The part of the checkpoint `restore_training_state` needs once the checkpoint itself is released."""
    keys = ('epoch', 'best_fitness', 'scheduler', 'scaler', 'early_stopping', 'last_opt_step')
    return {key: checkpoint[key] for key in keys if key in checkpoint}


def restore_training_state(state: dict[str, Any], scheduler: Any, scaler: Any, stopper: Any,
                           default_last_opt_step: int) -> int:
    """Restore what `resume_fields` saved and return the last optimizer step.

    Checkpoints of older trainer versions lack these fields; their early stopping then starts at the saved epoch.
    """
    if 'scheduler' in state:
        scheduler.load_state_dict(state['scheduler'])
    if 'scaler' in state:
        scaler.load_state_dict(state['scaler'])
    if 'early_stopping' in state:
        stopper.best_epoch = state['early_stopping']['best_epoch']
        stopper.best_fitness = state['early_stopping']['best_fitness']
        stopper.possible_stop = state['early_stopping']['possible_stop']
    else:
        stopper.best_epoch = state['epoch']
        stopper.best_fitness = state['best_fitness']
    return state.get('last_opt_step', default_last_opt_step)


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
