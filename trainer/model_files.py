from typing import Union
import os


def get_best(training_path: str):
    path = training_path + '/result/weights'
    if not os.path.isdir(path):
        return []
    weightfiles = [os.path.join(path, f) for f in os.listdir(path) if 'best' in f and f.endswith('.pt')]
    return weightfiles


def delete_json_for_weightfile(weightfile: str):
    _try_remove(weightfile.replace('.pt', '.json'))


def _try_remove(file: str):
    try:
        os.remove(file)
    except:
        pass


def get_new(training_path: str) -> Union[str, None]:
    best = get_best(training_path)
    if best:
        return best[0]
    return None
