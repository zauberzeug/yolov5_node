import ctypes
import logging
import os
import re
import subprocess
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from learning_loop_node.data_classes import (BoxDetection, ImageMetadata,
                                             PointDetection)
from learning_loop_node.detector.detector_logic import DetectorLogic
from learning_loop_node.enums import CategoryType

import yolov5


class Yolov5Detector(DetectorLogic):

    def __init__(self) -> None:
        super().__init__('yolov5_wts')
        self.yolov5: Optional[yolov5.YoLov5TRT] = None
        self.weight_type = os.getenv('WEIGHT_TYPE', 'FP16')
        assert self.weight_type in ['FP16', 'FP32', 'INT8'], 'WEIGHT_TYPE must be one of FP16, FP32, INT8'
        self.log = logging.getLogger('Yolov5Detector')
        self.log.setLevel(logging.INFO)

    def init(self) -> None:
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

        self.yolov5 = yolov5.YoLov5TRT(engine_file)
        for _ in range(3):
            warmup = yolov5.warmUpThread(self.yolov5)
            warmup.start()
            warmup.join()

    @staticmethod
    def clip_box(
            x: float, y: float, width: float, height: float, img_width: int, img_height: int) -> Tuple[
            float, float, float, float]:
        '''make sure the box is within the image
            x,y is the center of the box
        '''
        left = max(0, x - 0.5 * width)
        top = max(0, y - 0.5 * height)
        right = min(img_width, x + 0.5 * width)
        bottom = min(img_height, y + 0.5 * height)

        x = 0.5 * (left + right)
        y = 0.5 * (top + bottom)
        width = right - left
        height = bottom - top

        return x, y, width, height

    @staticmethod
    def clip_point(x: float, y: float, img_width: int, img_height: int) -> Tuple[float, float]:
        x = min(max(0, x), img_width)
        y = min(max(0, y), img_height)
        return x, y

    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        assert self.yolov5 is not None, 'init() must be executed first. Maybe loading the engine failed?!'
        image_metadata = ImageMetadata()
        try:
            t = time.time()
            cv_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
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
                    x, y, w, h = self.clip_box(x, y, w, h, im_width, im_height)
                    image_metadata.box_detections.append(
                        BoxDetection(category_name=category.name,
                                     x=round(x),
                                     y=round(y),
                                     width=round(x+w)-round(x),
                                     height=round(y+h)-round(y),
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
                logging.warning(
                    f'Removed {len(skipped_detections)} small detections from result: \n{log_msg}')
        except Exception:
            self.log.exception('inference failed')
        return image_metadata

    def _create_engine(self, resolution: int, cat_count: int, wts_file: str) -> str:
        engine_file = os.path.dirname(wts_file) + '/model.engine'
        if os.path.isfile(engine_file):
            logging.info(f'{engine_file} already exists, skipping conversion')
            return engine_file
        logging.info(f'converting {wts_file} to {engine_file}')

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
