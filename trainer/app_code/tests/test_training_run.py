from pathlib import Path

import pytest
from learning_loop_node.trainer.exceptions import CriticalError, InsufficientMemoryError

from .. import training_run
from ..batch_size_calculation import InvalidResolutionError


def test_batch_size_is_unknown_until_reported(tmp_path: Path):
    run = training_run.TrainingRun(tmp_path / 'result')
    assert run.batch_size is None
    batch_size = 32

    training_run.report_batch_size(tmp_path / 'result', batch_size)

    assert run.batch_size == batch_size


def test_invalid_resolution_reaches_the_node(tmp_path: Path):
    with pytest.raises(InvalidResolutionError):
        with training_run.reporting(tmp_path):
            raise InvalidResolutionError('invalid resolution 600: must be a multiple of 32 and at least 64')

    with pytest.raises(InvalidResolutionError, match='invalid resolution 600: must be a multiple of 32 and at least 64'):
        training_run.TrainingRun(tmp_path).raise_failure()


def test_insufficient_memory_ends_the_training(tmp_path: Path):
    with pytest.raises(InsufficientMemoryError):
        with training_run.reporting(tmp_path):
            raise InsufficientMemoryError('batch size 2 does not fit in memory')

    with pytest.raises(CriticalError, match='too small for even the smallest batch size'):
        training_run.TrainingRun(tmp_path).raise_failure()


def test_missing_gpu_ends_the_training(tmp_path: Path):
    with pytest.raises(training_run.NoGpuError):
        with training_run.reporting(tmp_path):
            raise training_run.NoGpuError('no graphics card available')

    with pytest.raises(training_run.NoGpuError, match='no graphics card available'):
        training_run.TrainingRun(tmp_path).raise_failure()


def test_other_errors_are_left_to_the_log(tmp_path: Path):
    with pytest.raises(ValueError):
        with training_run.reporting(tmp_path):
            raise ValueError('a bug')

    training_run.TrainingRun(tmp_path).raise_failure()


def test_a_cleared_failure_is_not_raised_again(tmp_path: Path):
    with pytest.raises(InvalidResolutionError):
        with training_run.reporting(tmp_path):
            raise InvalidResolutionError('invalid resolution 600')
    run = training_run.TrainingRun(tmp_path)

    run.clear_failure()

    run.raise_failure()
