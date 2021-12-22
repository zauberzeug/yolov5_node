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
        self._create_engine(model_info.resolution, f'{model_root_path}/model.wts')

    def _create_engine(self, resolution: int, wts_file: str):
        engine_file = os.path.dirname(wts_file) + '/model.engine'
        if os.path.isfile(engine_file):
            logging.info(f'{engine_file} already exists, skipping conversion')
            return

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

    def evaluate(self, image: Any) -> Detections:
        detections = Detections()
        try:
            pass
        except Exception as e:
            logging.exception('inference failed')

        return detections
