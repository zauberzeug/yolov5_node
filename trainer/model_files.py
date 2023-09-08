import logging
import os
from typing import List, Union


def get_all_weightfiles(training_path: str) -> List[str]:
    path = training_path + '/result/weights'
    if not os.path.isdir(path):
        return []
    weightfiles = [os.path.join(path, f) for f in os.listdir(path) if 'epoch' in f and f.endswith('.pt')]
    return weightfiles


def _epoch_from_weightfile(weightfile: str) -> int:
    return int(weightfile.split('epoch')[-1].split('.pt')[0])


def delete_older_epochs(training_path: str, weightfile: str):
    all_weightfiles = get_all_weightfiles(training_path)

    target_epoch = _epoch_from_weightfile(weightfile)
    for f in all_weightfiles:
        if _epoch_from_weightfile(f) < target_epoch:
            _try_remove(f)
            delete_json_for_weightfile(f)


def delete_json_for_weightfile(weightfile: str):
    _try_remove(weightfile.replace('.pt', '.json'))


def _try_remove(file: str):
    try:
        os.remove(file)
    except Exception:
        logging.exception(f'could not remove {file}')


def get_new(training_path: str) -> Union[str, None]:
    all_weightfiles = get_all_weightfiles(training_path)
    if all_weightfiles:
        all_weightfiles.sort(key=_epoch_from_weightfile)
        return all_weightfiles[-1]
    return None
