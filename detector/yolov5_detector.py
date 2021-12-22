from typing import Any
from learning_loop_node import ModelInformation
from learning_loop_node import Detector
from learning_loop_node.detector import Detections
import logging
import os
import subprocess
import re


class Yolov5Detector(Detector):

    def __init__(self) -> None:
        super().__init__('yolov5_wts')

    def init(self,  model_info: ModelInformation, model_root_path: str):
        model_file = f'{model_root_path}/model.wts'
        self._build(model_info.resolution)

    def _build(self, resolution: int):
        os.chdir('/tensorrtx/yolov5/build')  # NOTE cmake and inital building is done in Dockerfile (to speeds things up)
        # Adapt resolution
        with open('../yololayer.h', 'r+') as f:
            content = f.read()
            content = re.sub('(INPUT_[HW] =) \d*', r'\1 ' + str(resolution), content)
            f.seek(0)
            f.truncate()
            f.write(content)
        subprocess.run('make -j6 -Wno-deprecated-declarations', shell=True)

    def evaluate(self, image: Any) -> Detections:
        detections = Detections()
        try:
            pass
        except Exception as e:
            logging.exception('inference failed')

        return detections
