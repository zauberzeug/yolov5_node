import os
from typing import Union


def get_all(training_path: str):
    path = training_path + '/result/weights'
    if not os.path.isdir(path):
        return []
    weightfiles = [os.path.join(path, f) for f in os.listdir(path) if 'epoch' in f and f.endswith('.pt')]
    return weightfiles


def _epoch_from_weightfile(weightfile: str) -> int:
    return int(weightfile.split('epoch')[-1].split('.pt')[0])


def delete_older_epochs(training_path: str, weightfile: str):
    all = get_all(training_path)

    target_epoch = _epoch_from_weightfile(weightfile)
    for f in all:
        if _epoch_from_weightfile(f) < target_epoch:
            _try_remove(f)
            delete_json_for_weightfile(f)


def delete_json_for_weightfile(weightfile: str):
    _try_remove(weightfile.replace('.pt', '.json'))


def _try_remove(file: str):
    try:
        os.remove(file)
    except:
        pass


def get_new(training_path: str) -> Union[str, None]:
    all = get_all(training_path)
    if all:
        all.sort(key=_epoch_from_weightfile)
        return all[-1]
    return None
