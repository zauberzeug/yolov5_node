import logging
import os
from pathlib import Path


def get_all_weightfiles(training_path: Path) -> list[Path]:
    return _weightfiles_in(training_path / 'result/weights')


def resume_checkpoint(training_path: Path) -> Path | None:
    """The checkpoint an interrupted training continues from; older trainings only have the published one."""
    weights = training_path / 'result/weights'
    for path in (weights / 'last.pt', weights / 'published/latest.pt'):
        if path.exists():
            return path
    return None


def epoch_from_weightfile(weightfile: Path) -> int:
    try:
        number = weightfile.name[5:-3]
        if number == '':
            return 0
        return int(number)
    except ValueError:
        return 0


def delete_older_epochs(training_path: Path, weightfile: Path) -> None:
    all_weightfiles = get_all_weightfiles(training_path)

    target_epoch = epoch_from_weightfile(weightfile)
    for f in all_weightfiles:
        if epoch_from_weightfile(f) < target_epoch:
            _try_remove(f)
            delete_json_for_weightfile(f)


def delete_newer_epochs(weights_folder: Path, epoch: int) -> None:
    for f in _weightfiles_in(weights_folder):
        if epoch_from_weightfile(f) > epoch:
            _try_remove(f)
            delete_json_for_weightfile(f)


def delete_json_for_weightfile(weightfile: Path) -> None:
    _try_remove(weightfile.with_suffix('.json'))


def _weightfiles_in(weights_folder: Path) -> list[Path]:
    path = weights_folder.absolute()
    if not path.exists():
        return []
    return [path / f for f in os.listdir(path) if 'epoch' in f and f.endswith('.pt')]


def _try_remove(file: Path) -> None:
    try:
        os.remove(file)
    except Exception:
        logging.exception('could not remove %s', file)


def get_new(training_path: Path) -> Path | None:
    all_weightfiles = get_all_weightfiles(training_path)
    if all_weightfiles:
        all_weightfiles.sort(key=epoch_from_weightfile)
        return all_weightfiles[-1]
    return None
