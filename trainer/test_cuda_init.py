import signal
import sys
import time
from typing import Any

import torch


def signal_handler(sig: int, frame: Any) -> None:
    print('\n\nSignal received:', sig, flush=True)
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("CUDA available: ", torch.cuda.is_available(), flush=True)

    torch.cuda.init()

    print("CUDA version: ", torch.version.cuda, flush=True)
    print("CUDA device count: ", torch.cuda.device_count(), flush=True)
    print("CUDA device name: ", torch.cuda.get_device_name(0), flush=True)

    print("CUDA initialized: ", torch.cuda.is_initialized())
    print("CUDA current device: ", torch.cuda.current_device())
    print("CUDA device properties: ", torch.cuda.get_device_properties(0))
    print("CUDA memory allocated: ", torch.cuda.memory_allocated())

    time.sleep(600)

    torch.cuda.empty_cache()
