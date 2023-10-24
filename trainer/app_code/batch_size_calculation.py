import asyncio
import logging
import os
from multiprocessing import Process, Queue

import torch
import yaml
from learning_loop_node.helpers.misc import get_free_memory_mb
from torchinfo import Verbosity, summary

from .yolov5.models.yolo import Model
from .yolov5.utils.downloads import attempt_download


async def calc(training_path: str, model_file: str, hyp_path: str, dataset_path: str, img_size: int) -> int:
    queue = Queue()
    p = Process(target=_calc_batch_size, args=(queue, training_path, model_file, hyp_path, dataset_path, img_size))
    p.start()

    try:
        while p.is_alive():
            await asyncio.sleep(1)
            logging.warning('still calculating best batch size')
    except asyncio.CancelledError:
        logging.warning('the training was cancelled during batch size calculation')
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
    logging.info('calc_batch_size.....')

    os.chdir('/tmp')

    with open(hyp_path) as f:
        hyp = yaml.safe_load(f)
    with open(dataset_path) as f:
        dataset = yaml.safe_load(f)

    # Try to download pretrained model. Its ok when its a 'Continued' training.
    attempt_download(model_file)

    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    logging.error(f'{t}, {r}, {a}')
    free_mem = get_free_memory_mb()
    fraction = 0.95
    free_mem *= fraction
    logging.info(f'{fraction:.0%} of free memory ({free_mem}) in use')

    device = torch.device('cuda', 0)
    torch.cuda.empty_cache()
    try:
        ckpt = torch.load(model_file, map_location=device)
    except FileNotFoundError:
        # Continue Training
        ckpt = torch.load(f'{training_path}/{model_file}', map_location=device)

    model = Model(ckpt['model'].yaml, ch=3, nc=dataset.get('nc'), anchors=hyp.get('anchors')).to(device)  # create

    best_batch_size = None
    for batch_size in [128, 64, 32, 16, 8, 4, 2, 1]:
        try:
            stats = summary(model, input_size=(batch_size, 3, img_size, img_size), verbose=Verbosity.QUIET)
        except RuntimeError as e:
            logging.error(f'Got RuntimeError for batch_size {batch_size} and image_size {img_size}: {str(e)}')
            continue
        estimated_total_size = str(stats).split('Estimated Total Size (MB): ')[1]
        estimated_total_size = float(estimated_total_size.split('\n')[0])

        logging.info(
            f'estimated_total_size for batch_size {batch_size} and image_size {img_size} is {estimated_total_size}')

        if estimated_total_size < free_mem:
            logging.info(f'batch_size {batch_size} and image_size {img_size} is good')
            best_batch_size = batch_size
            break
        else:
            logging.info(
                f'batch_size {batch_size} and image_size {img_size} is too big to fit in available memory {free_mem} ')

    model = model.cpu()
    del model
    del ckpt
    torch.cuda.empty_cache()

    if best_batch_size:
        queue.put(best_batch_size)
        logging.info(f'found best matching batch size:  {best_batch_size}')
    else:
        logging.error('Did not find best matching batch size')
        raise Exception('Did not find best matching batch size')
