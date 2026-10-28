from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app_code import image_cache
from app_code.image_cache import CacheBudget
from app_code.yolov5.utils.dataloaders import LoadImagesAndLabels


def test_caches_share_half_of_the_memory_available_before_the_first() -> None:
    budget = CacheBudget(lambda: 100)
    assert budget.claim(40)
    assert not budget.claim(20)
    assert budget.claim(10)
    assert not budget.claim(1)


def test_available_memory_is_read_once() -> None:
    reads = iter([100, 1000])
    budget = CacheBudget(lambda: next(reads))
    assert budget.claim(50)
    assert not budget.claim(1)


def test_cgroup_limit_caps_the_available_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'memory.max').write_text(f'{8 << 30}\n')
    (tmp_path / 'memory.current').write_text(f'{6 << 30}\n')
    monkeypatch.setattr(image_cache, 'CGROUP_MEMORY_MAX', tmp_path / 'memory.max')
    monkeypatch.setattr(image_cache, 'CGROUP_MEMORY_CURRENT', tmp_path / 'memory.current')
    monkeypatch.setattr(image_cache.psutil, 'virtual_memory', lambda: SimpleNamespace(available=28 << 30))
    assert image_cache.available_memory() == 2 << 30


@pytest.mark.parametrize('limit', ['max\n', None])
def test_without_a_cgroup_limit_the_host_memory_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       limit: str | None) -> None:
    if limit is not None:
        (tmp_path / 'memory.max').write_text(limit)
        (tmp_path / 'memory.current').write_text('0\n')
    monkeypatch.setattr(image_cache, 'CGROUP_MEMORY_MAX', tmp_path / 'memory.max')
    monkeypatch.setattr(image_cache, 'CGROUP_MEMORY_CURRENT', tmp_path / 'memory.current')
    monkeypatch.setattr(image_cache.psutil, 'virtual_memory', lambda: SimpleNamespace(available=28 << 30))
    assert image_cache.available_memory() == 28 << 30


@pytest.mark.parametrize(('available', 'cached'), [(10 << 20, True), (1 << 20, False)])
def test_dataset_is_cached_only_within_the_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  available: int, cached: bool) -> None:
    files = []
    for i in range(4):
        files.append(str(tmp_path / f'{i}.jpg'))
        cv2.imwrite(files[-1], np.zeros((480, 640, 3), dtype=np.uint8))
    monkeypatch.setattr(image_cache, 'budget', CacheBudget(lambda: available))
    dataset = SimpleNamespace(n=len(files), im_files=files, img_size=640)

    assert LoadImagesAndLabels.check_cache_ram(dataset) is cached  # 4 x 0.9 MB, plus the 10 % margin
