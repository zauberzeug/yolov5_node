import logging
import os
from pathlib import Path
from typing import List, Optional, Union


def get_best(training_path: Path) -> Optional[Path]:
    path = training_path / 'result/weights'
    if not path.exists():
        return None
    weightfiles = [path / f for f in os.listdir(path) if 'best' in f and f.endswith('.pt')]
    if len(weightfiles) == 0:
        return None
    return weightfiles[0]


def get_all_weightfiles(training_path: Path) -> List[Path]:
    path = (training_path / 'result/weights').absolute()
    if not path.exists():
        return []
    weightfiles = [path / f for f in os.listdir(path) if 'epoch' in f and f.endswith('.pt')]
    return weightfiles


def _epoch_from_weightfile(weightfile: Path) -> int:
    number = weightfile.name[5:-3]
    if number == '':
        return 0
    return int(number)


def delete_older_epochs(training_path: Path, weightfile: Path):
    all_weightfiles = get_all_weightfiles(training_path)

    target_epoch = _epoch_from_weightfile(weightfile)
    for f in all_weightfiles:
        if _epoch_from_weightfile(f) < target_epoch:
            _try_remove(f)
            delete_json_for_weightfile(f)


def delete_json_for_weightfile(weightfile: Path):
    _try_remove(weightfile.with_suffix('.json'))


def _try_remove(file: Path):
    try:
        os.remove(file)
    except Exception:
        logging.exception(f'could not remove {file}')


def get_new(training_path: Path) -> Union[Path, None]:
    all_weightfiles = get_all_weightfiles(training_path)
    if all_weightfiles:
        all_weightfiles.sort(key=_epoch_from_weightfile)
        return all_weightfiles[-1]
    return None
