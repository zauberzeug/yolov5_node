import ctypes
import logging
import os
import re
import subprocess
import time

import numpy as np
from learning_loop_node.data_classes import (
    BoxDetection,
    ImageMetadata,
    ImagesMetadata,
    PointDetection,
)
from learning_loop_node.detector.detector_logic import DetectorLogic
from learning_loop_node.enums import CategoryType

import yolov5


class Yolov5Detector(DetectorLogic):
    MIN_BOX_SIZE = 2

    def __init__(self) -> None:
        super().__init__('yolov5_wts')
        self.yolov5: yolov5.YoLov5TRT | None = None
        self.weight_type = os.getenv('WEIGHT_TYPE', 'FP16')
        assert self.weight_type in ['FP16', 'FP32', 'INT8'], 'WEIGHT_TYPE must be one of FP16, FP32, INT8'
        self.log = logging.getLogger('Yolov5Detector')
        self.log.setLevel(logging.INFO)
        self.iou_threshold = float(os.getenv('IOU_THRESHOLD', '0.45'))
        self.conf_threshold = float(os.getenv('CONF_THRESHOLD', '0.2'))

    def init(self) -> None:
        if self.model_info is None:
            raise RuntimeError('Model info not initialized. Call load_model_info_and_init_model() first.')
        if not isinstance(self.model_info.resolution, int) or self.model_info.resolution <= 0:
            raise RuntimeError("resolution must be an integer > 0")

        engine_file = self._create_engine(self.model_info.resolution,
                                          len(self.model_info.categories),
                                          self.model_info.model_size,
                                          f'{self.model_info.model_root_path}/model.wts')

        ctypes.CDLL('/tensorrtx/yolov5/build/libmyplugins.so')
        if self.yolov5 is not None:
            self.yolov5.destroy()
            self.yolov5 = None
            self.log.info('destroyed old yolov5 instance')

        try:
            self.yolov5 = yolov5.YoLov5TRT(engine_file, self.iou_threshold, self.conf_threshold)

        except RuntimeError as e:
            if 'TensorRT version mismatch' in str(e) or 'deserialize' in str(e):
                self.log.error('TensorRT engine compatibility issue: %s', e)
                if os.path.isfile(engine_file):
                    self.log.info('Removing incompatible engine file: %s', engine_file)
                    os.remove(engine_file)
            raise

        for _ in range(8):
            warmup = yolov5.warmUpThread(self.yolov5)
            warmup.start()
            warmup.join()

        self.log.info('Yolov5Detector initialized successfully')

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

    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        if self.yolov5 is None:
            raise RuntimeError('init() must be executed first. Maybe loading the engine failed?!')
        if self.model_info is None:
            raise RuntimeError('model_info must be set before calling evaluate()')

        try:
            t = time.time()
            results, inference_ms = self.yolov5.infer(image)
            self.log.debug('took %f s, overall %f s', inference_ms, time.time() - t)
            return self._collect_detections(results, image.shape[0], image.shape[1])

        except Exception as e:
            raise RuntimeError('Error during inference') from e

    def batch_evaluate(self, images: list[np.ndarray]) -> ImagesMetadata:
        if self.yolov5 is None:
            raise RuntimeError('init() must be executed first. Maybe loading the engine failed?!')
        if self.model_info is None:
            raise RuntimeError('model_info must be set before calling evaluate()')

        if len(images) == 0:
            return []

        shape = images[0].shape

        try:
            t = time.time()
            results = self.yolov5.infer_batch(images)

            total_ms = 0
            detections = []
            for result, inference_ms in results:
                total_ms += inference_ms
                detections.append(self._collect_detections(result, shape[0], shape[1]))

            self.log.debug('batch infer took %f s, overall %f s', total_ms, time.time() - t)
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
            self.log.warning('Removed %d small detections from result: \n%s', len(skipped_detections), log_msg)

        return image_metadata

    def _create_engine(self, resolution: int, cat_count: int, model_variant: str | None, wts_file: str) -> str:
        engine_file = os.path.dirname(wts_file) + '/model.engine'
        if os.path.isfile(engine_file):
            self.log.info('Engine at %s already exists, skipping conversion', engine_file)
            return engine_file
        self.log.info('Building Engine (%s to %s)', wts_file, engine_file)
        self.log.info('Resolution: %s', resolution)
        self.log.info('Weight_type: %s', self.weight_type)
        self.log.info('Num Categories: %d', cat_count)
        self.log.info('Model_variant: %s', model_variant)

        # NOTE cmake and initial building is done in Dockerfile (to speeds things up)
        os.chdir('/tensorrtx/yolov5/build')

        # Adapt resolution
        with open('../src/config.h', 'r+') as f:
            content = f.read()
            if self.weight_type == 'INT8':
                self.log.info('using INT8')
                content = content.replace('#define USE_FP16', '#define USE_INT8')
            elif self.weight_type == 'FP32':
                self.log.info('using FP32')
                content = content.replace('#define USE_FP16', '#define USE_FP32')
            else:
                self.log.info('using FP16')

            content = re.sub(r'(kNumClass =) \d*', r'\1 ' + str(cat_count), content)
            content = re.sub(r'(kInput[HW] =) \d*', r'\1 ' + str(resolution), content)
            f.seek(0)
            f.truncate()
            f.write(content)

        self.log.info('Making tensorrtx/yolov5')
        subprocess.run('make -j6 -Wno-deprecated-declarations', shell=True, check=True)

        self.log.info('Building engine file with tensorrtx/yolov5_det')
        model_variant = model_variant or 's6'

        subprocess.run(f'./yolov5_det -s {wts_file} {engine_file} {model_variant}', shell=True, check=True)
        return engine_file
