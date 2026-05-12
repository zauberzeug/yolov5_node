from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Literal, final

import numpy as np
from learning_loop_node import DetectorLogic, DetectorLogicFactory
from learning_loop_node.data_classes import (
    BoxDetection,
    ImageMetadata,
    ImagesMetadata,
    ModelInformation,
    PointDetection,
)
from learning_loop_node.enums import CategoryType
from typing_extensions import override

import yolov5


@final
@dataclass(frozen=True)
class Yolov5DetectorParams(DetectorLogicFactory):
    weight_type: Literal['FP16', 'FP32', 'INT8']
    iou_threshold: float
    conf_threshold: float

    @property
    @override
    def model_format(self) -> str:
        return 'yolov5_wts'

    @override
    async def build(self, model_info: ModelInformation) -> Yolov5Detector:
        return await asyncio.to_thread(Yolov5Detector, model_info, self)


_LOG = logging.getLogger('Yolov5Detector')
_LOG.setLevel(logging.INFO)


@final
class Yolov5Detector(DetectorLogic):
    MIN_BOX_SIZE = 2

    def __init__(self, model_info: ModelInformation, params: Yolov5DetectorParams) -> None:
        self.model_info = model_info

        if not isinstance(model_info.resolution, int) or model_info.resolution <= 0:
            raise RuntimeError("resolution must be an integer > 0")

        engine_file = _create_engine(model_info.resolution,
                                     len(model_info.categories),
                                     model_info.model_size,
                                     f'{model_info.model_root_path}/model.wts',
                                     params.weight_type)

        ctypes.CDLL('/tensorrtx/yolov5/build/libmyplugins.so')

        try:
            self.yolov5 = yolov5.YoLov5TRT(engine_file, params.iou_threshold, params.conf_threshold)
        except RuntimeError as e:
            if 'TensorRT version mismatch' in str(e) or 'deserialize' in str(e):
                _LOG.error('TensorRT engine compatibility issue: %s', e)
                if os.path.isfile(engine_file):
                    _LOG.info('Removing incompatible engine file: %s', engine_file)
                    os.remove(engine_file)
            raise

        for _ in range(8):
            warmup = yolov5.WarmUpThread(self.yolov5)
            warmup.start()
            warmup.join()

        _LOG.info('Yolov5Detector initialized successfully')

    @staticmethod
    def clip_box(x1: float, y1: float, width: float, height: float, img_width: int, img_height: int) -> tuple[int, int, int, int]:  # noqa: PLR0913
        """
        Clips a box defined by top-left corner (x1, y1), width, and height
        to stay within image boundaries (img_width, img_height).
        Returns the clipped (x1, y1, width, height) as ints.
        """

        x2 = x1 + width
        y2 = y1 + height

        # Clip coordinates
        clipped_x1 = round(max(0.0, x1))
        clipped_y1 = round(max(0.0, y1))
        clipped_x2 = round(min(float(img_width), x2))
        clipped_y2 = round(min(float(img_height), y2))

        # Recalculate dimensions
        clipped_width = clipped_x2 - clipped_x1
        clipped_height = clipped_y2 - clipped_y1

        # Ensure width and height are non-negative
        clipped_width = max(clipped_width, 0)
        clipped_height = max(clipped_height, 0)

        return clipped_x1, clipped_y1, clipped_width, clipped_height

    @staticmethod
    def clip_point(x: float, y: float, img_width: int, img_height: int) -> tuple[float, float]:
        x = min(max(0, x), img_width)
        y = min(max(0, y), img_height)
        return x, y

    @override
    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        try:
            t = time.time()
            results, inference_ms = self.yolov5.infer(image)
            _LOG.debug('took %f s, overall %f s', inference_ms, time.time() - t)
            return self._collect_detections(results, image.shape[0], image.shape[1])

        except Exception as e:
            raise RuntimeError('Error during inference') from e

    @override
    def batch_evaluate(self, images: list[np.ndarray]) -> ImagesMetadata:
        if len(images) == 0:
            return ImagesMetadata([])

        shape = images[0].shape

        try:
            t = time.time()
            results, total_ms = self.yolov5.infer_batch(images)

            detections = [self._collect_detections(result, shape[0], shape[1]) for result in results]

            _LOG.debug('batch infer took %f s, overall %f s', total_ms, time.time() - t)
            return ImagesMetadata(items=detections)

        except Exception as e:
            raise RuntimeError('Error during inference') from e

    def _collect_detections(self, detections: list[yolov5.Detection], im_height: int, im_width: int) -> ImageMetadata:
        image_metadata = ImageMetadata()
        skipped_detections = []

        for detection in detections:
            x, y, w, h, category_idx, probability = detection
            category = self.model_info.categories[category_idx]
            if w <= self.MIN_BOX_SIZE or h <= self.MIN_BOX_SIZE:  # skip very small boxes.
                skipped_detections.append((category.name, detection))
                continue
            if category.type == CategoryType.Box:
                clipped_x1, clipped_y1, clipped_w, clipped_h = self.clip_box(
                    x, y, w, h, im_width, im_height)
                image_metadata.box_detections.append(
                    BoxDetection(category_name=category.name,
                                 x=clipped_x1,
                                 y=clipped_y1,
                                 width=clipped_w,
                                 height=clipped_h,
                                 category_id=category.id,
                                 model_name=self.model_info.version,
                                 confidence=probability))
            elif category.type == CategoryType.Point:
                cx, cy = x + w/2, y + h/2
                cx, cy = self.clip_point(cx, cy, im_width, im_height)
                image_metadata.point_detections.append(
                    PointDetection(category_name=category.name,
                                   x=cx,
                                   y=cy,
                                   category_id=category.id,
                                   model_name=self.model_info.version,
                                   confidence=probability))
        if skipped_detections:
            log_msg = '\n'.join([str(d) for d in skipped_detections])
            _LOG.warning('Removed %d small detections from result: \n%s', len(skipped_detections), log_msg)

        return image_metadata


def _create_engine(resolution: int, cat_count: int, model_variant: str | None, wts_file: str, weight_type: str) -> str:
    engine_file = os.path.dirname(wts_file) + '/model.engine'
    if os.path.isfile(engine_file):
        _LOG.info('Engine at %s already exists, skipping conversion', engine_file)
        return engine_file
    _LOG.info('Building Engine (%s to %s)', wts_file, engine_file)
    _LOG.info('Resolution: %s', resolution)
    _LOG.info('Weight_type: %s', weight_type)
    _LOG.info('Num Categories: %d', cat_count)
    _LOG.info('Model_variant: %s', model_variant)

    os.chdir('/tensorrtx/yolov5/build')

    # Adapt resolution
    with open('../src/config.h', 'r+') as f:
        content = f.read()
        if weight_type == 'INT8':
            _LOG.info('using INT8')
            content = content.replace('#define USE_FP16', '#define USE_INT8')
        elif weight_type == 'FP32':
            _LOG.info('using FP32')
            content = content.replace('#define USE_FP16', '#define USE_FP32')
        else:
            _LOG.info('using FP16')

        content = re.sub(r'(kNumClass =) \d*', r'\1 ' + str(cat_count), content)
        content = re.sub(r'(kInput[HW] =) \d*', r'\1 ' + str(resolution), content)
        f.seek(0)
        f.truncate()
        f.write(content)

    if not os.path.isfile('Makefile'):
        _LOG.info('Running cmake for tensorrtx/yolov5')
        subprocess.run(
            'cmake '
            '-DCMAKE_CUDA_FLAGS="--diag-suppress=997 -Xcompiler=-Wno-deprecated-declarations" '
            '-DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" ..',
            shell=True, check=True)

    _LOG.info('Making tensorrtx/yolov5')
    subprocess.run('make -j6 -Wno-deprecated-declarations', shell=True, check=True)

    _LOG.info('Building engine file with tensorrtx/yolov5_det')
    model_variant = model_variant or 's6'

    subprocess.run(f'./yolov5_det -s {wts_file} {engine_file} {model_variant}', shell=True, check=True)
    return engine_file
