"""How much memory `train_det.py` may fill with decoded images under ``--cache ram``.

The cache holds every image decoded and resized to ``--img`` for the whole training. The caches
of all sets of one run together stay within :data:`RAM_SHARE` of the memory that was available
before the first of them, and a set that does not fit is loaded from disk instead.
"""
from collections.abc import Callable
from pathlib import Path

import psutil

RAM_SHARE = 0.5
"""Share of the available memory the image caches of one training take together at most."""

CGROUP_MEMORY_MAX = Path('/sys/fs/cgroup/memory.max')
CGROUP_MEMORY_CURRENT = Path('/sys/fs/cgroup/memory.current')
CGROUP_MEMORY_STAT = Path('/sys/fs/cgroup/memory.stat')


class CacheBudget:
    """The bytes the image caches of one `train_det.py` run may still take."""

    def __init__(self, available: Callable[[], int] | None = None) -> None:
        """:param available: Read once, when the first cache asks; defaults to :func:`available_memory`."""
        self._available = available or available_memory
        self._remaining: int | None = None

    @property
    def remaining(self) -> int:
        if self._remaining is None:
            self._remaining = int(self._available() * RAM_SHARE)
        return self._remaining

    def claim(self, required_bytes: float) -> bool:
        """Take ``required_bytes`` from the budget, or leave it untouched and return ``False``."""
        if required_bytes > self.remaining:
            return False
        self._remaining = self.remaining - int(required_bytes)
        return True


def available_memory() -> int:
    """Bytes this process can still allocate: the host's available memory, capped by a cgroup v2 limit."""
    available = psutil.virtual_memory().available
    headroom = _cgroup_headroom()
    return available if headroom is None else min(available, headroom)


def _cgroup_headroom() -> int | None:
    """Bytes left below the container's memory limit, counting inactive page cache as free; ``None`` without a limit."""
    try:
        limit = CGROUP_MEMORY_MAX.read_text().strip()
        current = int(CGROUP_MEMORY_CURRENT.read_text())
        stat = dict(line.split() for line in CGROUP_MEMORY_STAT.read_text().splitlines())
    except (OSError, ValueError):
        return None
    if limit == 'max':
        return None
    used = current - int(stat.get('inactive_file', 0))
    return max(0, int(limit) - used)


budget = CacheBudget()
