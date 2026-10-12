"""What `train_det.py` leaves in its result folder for the trainer node to read.

`train_det.py` writes through :func:`report_batch_size` and :func:`reporting`; the node reads
through :class:`TrainingRun`. The file names and their format stay inside this module.
"""
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from learning_loop_node.trainer.exceptions import CriticalError, InsufficientMemoryError

from .batch_size_calculation import InvalidResolutionError

BATCH_SIZE_FILE = 'batch_size.json'
FAILURE_FILE = 'failure.json'


class NoGpuError(CriticalError):
    pass


_KINDS: dict[type[Exception], str] = {
    InvalidResolutionError: 'invalid_resolution',
    InsufficientMemoryError: 'insufficient_memory',
    NoGpuError: 'no_gpu',
}
_ERRORS: dict[str, Callable[[str], CriticalError]] = {
    'invalid_resolution': InvalidResolutionError,
    'insufficient_memory': lambda _: CriticalError('graphics card is too small for even the smallest batch size'),
    'no_gpu': NoGpuError,
}


def report_batch_size(result_dir: Path, batch_size: int) -> None:
    _write_json(result_dir / BATCH_SIZE_FILE, {'batch_size': batch_size})


@contextmanager
def reporting(result_dir: Path) -> Iterator[None]:
    """Record a failure that no retry can fix, then let it propagate."""
    try:
        yield
    except tuple(_KINDS) as e:
        kind = next(kind for error_type, kind in _KINDS.items() if isinstance(e, error_type))
        _write_json(result_dir / FAILURE_FILE, {'kind': kind, 'message': str(e)})
        raise


class TrainingRun:
    """The result folder of a `train_det.py` run, as the trainer node sees it."""

    def __init__(self, result_dir: Path) -> None:
        """:param result_dir: The folder `train_det.py` writes to, ``--project``/``--name``."""
        self.result_dir = result_dir

    @property
    def batch_size(self) -> int | None:
        """The batch size the probe settled on; None until it has."""
        content = _read_json(self.result_dir / BATCH_SIZE_FILE)
        return None if content is None else content['batch_size']

    def clear_failure(self) -> None:
        """Forget the failure of an earlier run; call before starting the next one."""
        (self.result_dir / FAILURE_FILE).unlink(missing_ok=True)

    def raise_failure(self) -> None:
        """Raise the failure `train_det.py` recorded, if any.

        :raises InvalidResolutionError: If the resolution does not suit the model's largest stride.
        :raises CriticalError: If not even the smallest batch fits on the graphics card.
        :raises NoGpuError: If `train_det.py` found no graphics card to train on.
        """
        failure = _read_json(self.result_dir / FAILURE_FILE)
        if failure is not None:
            raise _ERRORS[failure['kind']](failure['message'])


def _write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(json.dumps(content))
    os.replace(temporary, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
