import asyncio
import logging
import os

import torch
import yaml
from learning_loop_node.helpers.misc import get_free_memory_mb
from learning_loop_node.trainer.exceptions import CriticalError
from torch.multiprocessing import Process, Queue, set_start_method
from torchinfo import Verbosity, summary

from .yolov5.models.yolo import Model
from .yolov5.utils.downloads import attempt_download


async def calc(training_path: str, model_file: str, hyp_path: str, dataset_path: str, img_size: int,
               init_clear_cuda: bool = True) -> int:

    os.chdir('/tmp')

    with open(hyp_path) as f:
        hyp = yaml.safe_load(f)
    with open(dataset_path) as f:
        dataset = yaml.safe_load(f)

    attempt_download(model_file)  # Download pretrained yolov5 model from ultralytics to .pt

    if init_clear_cuda:
        torch.cuda.init()
        torch.cuda.empty_cache()
    device = torch.device('cuda', 0)

    free_mem_mb = get_free_memory_mb()
    fraction = 0.95
    free_mem_mb *= fraction

    try:
        ckpt = torch.load(model_file, map_location=device)
    except FileNotFoundError:
        ckpt = torch.load(f'{training_path}/{model_file}', map_location=device)

    model = Model(ckpt['model'].yaml, ch=3, nc=dataset.get('nc'), anchors=hyp.get('anchors')).to(device)  # create

    best_batch_size = None
    for batch_size in [128, 96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 2, 1]:
        try:
            stats = summary(model, input_size=(batch_size, 3, img_size, img_size), verbose=Verbosity.QUIET)
        except RuntimeError as e:
            logging.error(f'Got RuntimeError for batch_size {batch_size} and image_size {img_size}: {str(e)}')
            continue

        estimated_model_size_mb = float(str(stats).split('Estimated Total Size (MB): ')[1].split('\n')[0])
        logging.info(f'Model size for batch size {batch_size} and image size {img_size}: {estimated_model_size_mb} mb')

        if estimated_model_size_mb < free_mem_mb:
            logging.info(f'batch size {batch_size} and image size {img_size} fits to {free_mem_mb} mb')
            best_batch_size = batch_size
            break

    model = model.cpu()  # TODO WTF??
    del model, ckpt

    if init_clear_cuda:
        torch.cuda.empty_cache()

    if not best_batch_size:
        logging.error('Did not find best matching batch size')
        raise CriticalError('Did not find best matching batch size')
    return best_batch_size


# -------------------------------- BELOW IMPLEMENTATION RESULTS IN  RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method

async def calc_on_thread(training_path: str, model_file: str, hyp_path: str, dataset_path: str, img_size: int) -> int:
    logging.info('Calculating best batch size on thread.....')
    set_start_method('spawn')

    queue = Queue()  # type: Queue[int]
    p = Process(target=_calc_batch_size, args=(queue, training_path, model_file, hyp_path, dataset_path, img_size))
    p.start()

    try:
        while p.is_alive():
            await asyncio.sleep(1)
            logging.warning('Still calculating best batch size')
    except asyncio.CancelledError:
        logging.warning('Training cancelled during batch size calculation')
        p.kill()
        raise

    p.join()
    if p.exitcode != 0:
        raise Exception('calc_batch_size failed')

    batch_size = queue.get()
    logging.info(f'best batch size is {batch_size}')
    return batch_size


def _calc_batch_size(
        queue: Queue, training_path: str, model_file: str, hyp_path: str, dataset_path: str, img_size: int) -> None:
    best_batch_size = calc(training_path, model_file, hyp_path, dataset_path, img_size, init_clear_cuda=False)
    queue.put(best_batch_size)
