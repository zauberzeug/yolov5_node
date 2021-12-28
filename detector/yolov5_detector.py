from typing import Any, List
from learning_loop_node import ModelInformation
from learning_loop_node import Detector
from learning_loop_node.detector import Detections
import logging
import os
import subprocess
import re
import yolov5
import ctypes
import cv2
import numpy as np


class Yolov5Detector(Detector):

    def __init__(self) -> None:
        super().__init__('yolov5_wts')

    def init(self,  model_info: ModelInformation, model_root_path: str):
        self.model_info = model_info
        engine_file = self._create_engine(model_info.resolution, f'{model_root_path}/model.wts')
        ctypes.CDLL('/tensorrtx/yolov5/build/libmyplugins.so')
        self.yolov5 = yolov5.YoLov5TRT(engine_file)
        for i in range(3):
            warmup = yolov5.warmUpThread(self.yolov5)
            warmup.start()
            warmup.join()

    def evaluate(self, image: List[np.uint8]) -> Detections:
        detections = Detections()
        try:
            result, time = self.yolov5.infer([cv2.imdecode(image, cv2.IMREAD_COLOR)])
            logging.info(f'took {time} ms')
        except Exception as e:
            logging.exception('inference failed')

        return detections

    def _create_engine(self, resolution: int, wts_file: str) -> str:
        engine_file = os.path.dirname(wts_file) + '/model.engine'
        if os.path.isfile(engine_file):
            logging.info(f'{engine_file} already exists, skipping conversion')
            return engine_file

        # NOTE cmake and inital building is done in Dockerfile (to speeds things up)
        os.chdir('/tensorrtx/yolov5/build')
        # Adapt resolution
        with open('../yololayer.h', 'r+') as f:
            content = f.read()
            content = re.sub('(INPUT_[HW] =) \d*', r'\1 ' + str(resolution), content)
            f.seek(0)
            f.truncate()
            f.write(content)
        subprocess.run('make -j6 -Wno-deprecated-declarations', shell=True)
        subprocess.run(f'./yolov5 -s {wts_file} {engine_file} s6', shell=True)  # TODO parameterize variant "s6"
        return engine_file
