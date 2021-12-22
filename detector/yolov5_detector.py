from typing import Any
from learning_loop_node import ModelInformation
from learning_loop_node import Detector
from learning_loop_node.detector import Detections
import logging


class Yolov5Detector(Detector):

    def __init__(self) -> None:
        super().__init__('yolov5_wts')

    def init(self,  model_info: ModelInformation, model_root_path: str):
        model_file = f'{model_root_path}/model.rt'

    def evaluate(self, image: Any) -> Detections:
        detections = Detections()
        try:
            pass
        except Exception as e:
            logging.exception('inference failed')

        return detections
