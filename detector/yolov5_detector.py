import shutil
from typing import Any, List
from learning_loop_node import ModelInformation, Detector
from learning_loop_node.detector import Detections, BoxDetection, PointDetection
from learning_loop_node.data_classes import Category, CategoryType
import logging
import os
import subprocess
import re
import numpy as np
import time
import torch
import sys
from utils.torch_utils import select_device, time_sync
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression


class Yolov5Detector(Detector):

    def __init__(self) -> None:
        super().__init__('yolov5_pytorch')

    def init(self,  model_info: ModelInformation, model_root_path: str):
        weightfile = f'{model_root_path}/model.pt'
        if not os.path.exists(f'{model_root_path}/model.engine'):
            shutil.copy('/data/model.engine', f'{model_root_path}/model.engine')
            # subprocess.run(
            #     f'python3 /yolov5/export.py --device 0 --half --weights {weightfile} --include engine', shell=True)
        self.device = select_device('0')

        self.model = DetectMultiBackend(
            f'{model_root_path}/model.engine', device=self.device)
        logging.info(model_info.resolution)
        imgz = (model_info.resolution, model_info.resolution)
        self.imgz = check_img_size(imgz, s=self.model.stride)
        logging.info(self.imgz)
        self.half = self.model.engine and self.device.type != 'cpu'
        self.model.warmup(imgsz=(1, 3, *self.imgz), half=self.half)

    def evaluate(self, image: List[np.uint8]) -> Detections:
        detections = Detections()
        try:
            start = time_sync()
            im = torch.from_numpy(im).to(self.device)
            results = self.model(im, augment=False, visualize=False)
            skipped_detections = []
            end = time_sync()
            results = non_max_suppression(
                results, conf_thresh=0.2, iou_thresh=0.4)
            logging.info(f'took {end-start} s, overall {time.time() -start} s')
            for detection in results:
                logging.info(detection)
        #         x, y, w, h, category, probability = detection
        #         category = self.model_info.categories[category]
        #         if w <= 2 or h <= 2:  # skip very small boxes.
        #             skipped_detections.append((category.name, detection))
        #             continue

        #         if category.type == CategoryType.Box:
        #             detections.box_detections.append(BoxDetection(
        #                 category.name, x, y, w, h, self.model_info.version, probability
        #             ))
        #         elif category.type == CategoryType.Point:
        #             cx, cy = (np.average([x, x + w]), np.average([y, y + h]))
        #             detections.point_detections.append(PointDetection(
        #                 category.name, int(cx), int(
        #                     cy), self.model_info.version, probability
        #             ))
        #     if skipped_detections:
        #         log_msg = '\n'.join([str(d) for d in skipped_detections])
        #         logging.warning(
        #             f'Removed very small detections from inference result (count={len(skipped_detections)}): \n{log_msg}')
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
            content = re.sub('(CLASS_NUM =) \d*', r'\1 ' +
                             str(cat_count), content)
            content = re.sub('(INPUT_[HW] =) \d*',
                             r'\1 ' + str(resolution), content)
            f.seek(0)
            f.truncate()
            f.write(content)
        subprocess.run('make -j6 -Wno-deprecated-declarations', shell=True)
        logging.warning('currently we assume a Yolov5 s6 model;\
            parameterization of the variant (s, s6, m, m6, ...) still needs to be done')
        # TODO parameterize variant "s6"
        subprocess.run(f'./yolov5 -s {wts_file} {engine_file} s6', shell=True)
        return engine_file
