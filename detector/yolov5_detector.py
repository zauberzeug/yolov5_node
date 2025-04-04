import ctypes
import logging
import os
import re
import subprocess
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import yolov5
from learning_loop_node.data_classes import BoxDetection, ImageMetadata, PointDetection
from learning_loop_node.detector.detector_logic import DetectorLogic
from learning_loop_node.enums import CategoryType


class Yolov5Detector(DetectorLogic):

    def __init__(self) -> None:
        super().__init__('yolov5_wts')
        self.yolov5: Optional[yolov5.YoLov5TRT] = None
        self.weight_type = os.getenv('WEIGHT_TYPE', 'FP16')
        assert self.weight_type in ['FP16', 'FP32', 'INT8'], 'WEIGHT_TYPE must be one of FP16, FP32, INT8'
        self.log = logging.getLogger('Yolov5Detector')
        self.log.setLevel(logging.INFO)
        self.iou_threshold = float(os.getenv('IOU_THRESHOLD', '0.45'))
        self.conf_threshold = float(os.getenv('CONF_THRESHOLD', '0.2'))

    def init(self) -> None:
        assert self.model_info is not None, 'model_info must be set before calling init()'
        resolution = self.model_info.resolution
        assert resolution is not None
        engine_file = self._create_engine(resolution,
                                          len(self.model_info.categories),
                                          f'{self.model_info.model_root_path}/model.wts')
        ctypes.CDLL('/tensorrtx/yolov5/build/libmyplugins.so')
        if self.yolov5 is not None:
            self.yolov5.destroy()
            self.yolov5 = None
            self.log.info('destroyed old yolov5 instance')

        self.yolov5 = yolov5.YoLov5TRT(engine_file, self.iou_threshold, self.conf_threshold)
        for _ in range(3):
            warmup = yolov5.warmUpThread(self.yolov5)
            warmup.start()
            warmup.join()

    @staticmethod
    def clip_box(
            x1: float, y1: float, width: float, height: float, img_width: int, img_height: int) -> Tuple[
            int, int, int, int]:
        '''Clips a box defined by top-left corner (x1, y1), width, and height
           to stay within image boundaries (img_width, img_height).
           Returns the clipped (x1, y1, width, height) as ints.
        '''
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
        if clipped_width < 0:
            clipped_width = 0
        if clipped_height < 0:
            clipped_height = 0

        return clipped_x1, clipped_y1, clipped_width, clipped_height

    @staticmethod
    def clip_point(x: float, y: float, img_width: int, img_height: int) -> Tuple[float, float]:
        x = min(max(0, x), img_width)
        y = min(max(0, y), img_height)
        return x, y

    def evaluate(self, image: bytes) -> ImageMetadata:
        assert self.yolov5 is not None, 'init() must be executed first. Maybe loading the engine failed?!'
        assert self.model_info is not None, 'model_info must be set before calling evaluate()'

        image_metadata = ImageMetadata()
        try:
            t = time.time()
            cv_image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            im_height, im_width, _ = cv_image.shape
            results, inference_ms = self.yolov5.infer(cv_image)
            skipped_detections = []
            self.log.debug('took %f s, overall %f s', inference_ms, time.time() - t)
            for detection in results:
                x, y, w, h, category_idx, probability = detection
                category = self.model_info.categories[category_idx]
                if w <= 2 or h <= 2:  # skip very small boxes.
                    skipped_detections.append((category.name, detection))
                    continue
                if category.type == CategoryType.Box:
                    clipped_x1, clipped_y1, clipped_w, clipped_h = self.clip_box(x, y, w, h, im_width, im_height)
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
        except Exception:
            self.log.exception('inference failed')
        return image_metadata

    def _create_engine(self, resolution: int, cat_count: int, wts_file: str) -> str:
        engine_file = os.path.dirname(wts_file) + '/model.engine'
        if os.path.isfile(engine_file):
            self.log.info('%s already exists, skipping conversion', engine_file)
            return engine_file
        self.log.info('converting %s to %s', wts_file, engine_file)

        # NOTE cmake and inital building is done in Dockerfile (to speeds things up)
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

            content = re.sub('(kNumClass =) \d*', r'\1 ' +
                             str(cat_count), content)
            content = re.sub('(kInput[HW] =) \d*',
                             r'\1 ' + str(resolution), content)
            f.seek(0)
            f.truncate()
            f.write(content)

        subprocess.run('make -j6 -Wno-deprecated-declarations',
                       shell=True, check=True)
        self.log.warning('currently we assume a Yolov5 s6 model;\
            parameterization of the variant (s, s6, m, m6, ...) still needs to be done')
        # TODO parameterize variant "s6"
        subprocess.run(
            f'./yolov5_det -s {wts_file} {engine_file} s6', shell=True, check=True)
        return engine_file
