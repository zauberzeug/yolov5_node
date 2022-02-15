from typing import Any, List
from learning_loop_node import ModelInformation, Detector
from learning_loop_node.detector import Detections, BoxDetection, PointDetection
from learning_loop_node.data_classes import Category, CategoryType
import logging
import os
import subprocess
import re
import yolov5
import ctypes
import cv2
import numpy as np
import time


class Yolov5Detector(Detector):

    def __init__(self) -> None:
        super().__init__('yolov5_wts')

    def init(self,  model_info: ModelInformation, model_root_path: str):
        self.model_info = model_info
        engine_file = self._create_engine(
            model_info.resolution,
            len(model_info.categories),
            f'{model_root_path}/model.wts'
        )
        ctypes.CDLL('/tensorrtx/yolov5/build/libmyplugins.so')
        self.yolov5 = yolov5.YoLov5TRT(engine_file)
        for i in range(3):
            warmup = yolov5.warmUpThread(self.yolov5)
            warmup.start()
            warmup.join()

    def evaluate(self, image: List[np.uint8]) -> Detections:
        detections = Detections()
        try:
            t = time.time()
            results, inference_ms = self.yolov5.infer(cv2.imdecode(image, cv2.IMREAD_COLOR))
            skipped_detections = []
            logging.info(f'took {inference_ms} s, overall {time.time() -t} s')
            for detection in results:
                x, y, w, h, category, probability = detection
                category = self.model_info.categories[category]
                if w <= 2 or h <= 2:  # skip very small boxes.
                    skipped_detections.append((category.name, detection))
                    continue

                if category.type == CategoryType.Box:
                    detections.box_detections.append(BoxDetection(
                        category.name, x, y, w, h, self.model_info.version, probability
                    ))
                elif category.type == CategoryType.Point:
                    cx, cy = (np.average([x, x + w]), np.average([y, y + h]))
                    detections.point_detections.append(PointDetection(
                        category.name, int(cx), int(cy), self.model_info.version, probability
                    ))
            if skipped_detections:
                log_msg = '\n'.join([str(d) for d in skipped_detections])
                logging.warning(f'Removed very small detections from inference result (count={len(skipped_detections)}): \n{log_msg}')
        except Exception as e:
            logging.exception('inference failed')
        return detections

    def _create_engine(self, resolution: int, cat_count: int, wts_file: str) -> str:
        engine_file = os.path.dirname(wts_file) + '/model.engine'
        if os.path.isfile(engine_file):
            logging.info(f'{engine_file} already exists, skipping conversion')
            return engine_file

        # NOTE cmake and inital building is done in Dockerfile (to speeds things up)
        os.chdir('/tensorrtx/yolov5/build')
        # Adapt resolution
        with open('../yololayer.h', 'r+') as f:
            content = f.read()
            content = re.sub('(CLASS_NUM =) \d*', r'\1 ' + str(cat_count), content)
            content = re.sub('(INPUT_[HW] =) \d*', r'\1 ' + str(resolution), content)
            f.seek(0)
            f.truncate()
            f.write(content)
        subprocess.run('make -j6 -Wno-deprecated-declarations', shell=True)
        logging.warning('currently we assume a Yolov5 s6 model;\
            parameterization of the variant (s, s6, m, m6, ...) still needs to be done')
        subprocess.run(f'./yolov5 -s {wts_file} {engine_file} s6', shell=True)  # TODO parameterize variant "s6"
        return engine_file
