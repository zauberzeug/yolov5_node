import os
from pathlib import Path
from typing import Optional


def get_best(training_path: Path) -> Optional[Path]:
    path = training_path / 'result/weights'
    if not path.exists():
        return None
    weightfiles = [path / f for f in os.listdir(path) if 'best' in f and f.endswith('.pt')][0]
    return weightfiles


def delete_json_for_weightfile(weightfile: Path):
    _try_remove(weightfile.with_suffix('.json'))


def _try_remove(filepath: Path):
    try:
        os.remove(filepath)
    except OSError:
        pass
