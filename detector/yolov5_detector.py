from __future__ import annotations

import asyncio
import ctypes
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, final

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

        lib_config = _LibConfig(
            resolution=model_info.resolution,
            cat_count=len(model_info.categories),
            weight_type=params.weight_type,
        )
        _build_module_lib(lib_config)
        engine_file = _create_engine(f'{model_info.model_root_path}/model.wts',
                                     model_info.model_size or 's6')

        ctypes.CDLL(str(_LIB_FILE))

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


_BUILD_DIR = Path('/tensorrtx/yolov5/build')
_LIB_FILE = _BUILD_DIR / 'libmyplugins.so'
_DET_BIN = _BUILD_DIR / 'yolov5_det'
_LIB_CONFIG_FILE = _BUILD_DIR / 'build.json'


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class _LibConfig:
    resolution: int
    cat_count: int
    weight_type: Literal['FP16', 'FP32', 'INT8']

    def save(self, path: Path) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> _LibConfig | None:
        if not path.exists():
            return None
        try:
            with open(path, encoding='utf-8') as f:
                data: dict[str, Any] = json.load(f)
            return cls(**data)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            return None


def _build_module_lib(config: _LibConfig) -> None:
    existing = _LibConfig.load(_LIB_CONFIG_FILE)
    if _LIB_FILE.is_file() and _DET_BIN.is_file() and existing == config:
        _LOG.info('Module lib already built with matching config, skipping')
        return
    if _LIB_FILE.is_file():
        _LOG.info('Module lib config changed, rebuilding')

    _LOG.info('Building module lib (%s)', config)
    os.chdir(_BUILD_DIR)

    with open('../src/config.h', 'r+') as f:
        content = f.read()
        content = re.sub(r'#define USE_(FP16|FP32|INT8)', f'#define USE_{config.weight_type}', content)
        content = re.sub(r'(kNumClass =) \d*', r'\1 ' + str(config.cat_count), content)
        content = re.sub(r'(kInput[HW] =) \d*', r'\1 ' + str(config.resolution), content)
        f.seek(0)
        f.truncate()
        f.write(content)

    if not Path('Makefile').is_file():
        _LOG.info('Running cmake for tensorrtx/yolov5')
        subprocess.run(
            'cmake '
            '-DCMAKE_CUDA_FLAGS="--diag-suppress=997 -Xcompiler=-Wno-deprecated-declarations" '
            '-DCMAKE_CXX_FLAGS="-Wno-deprecated-declarations" ..',
            shell=True, check=True)

    _LOG.info('Making tensorrtx/yolov5')
    subprocess.run('make -j6 -Wno-deprecated-declarations', shell=True, check=True)

    config.save(_LIB_CONFIG_FILE)


def _create_engine(wts_file: str, model_variant: str) -> str:
    engine_file = Path(wts_file).parent / 'model.engine'
    if engine_file.is_file():
        _LOG.info('Engine at %s already exists, skipping conversion', engine_file)
        return str(engine_file)

    _LOG.info('Building engine %s from %s', engine_file, wts_file)
    os.chdir(_BUILD_DIR)
    subprocess.run(f'{_DET_BIN} -s {wts_file} {engine_file} {model_variant}', shell=True, check=True)
    return str(engine_file)
