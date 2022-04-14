import shutil
from typing import Any, List, Tuple
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
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
import cv2
from utils.augmentations import letterbox
import os
import yaml


class Yolov5Detector(Detector):

    def __init__(self) -> None:
        super().__init__('yolov5_pytorch')

    def init(self,  model_info: ModelInformation):
        self.model_info = model_info
        weightfile = f'{self.model_info.model_root_path}/model.pt'
        # get yaml file. Older models may not sent dataset.yaml
        data = f'{self.model_info.model_root_path}/dataset.yaml' if os.path.exists(
            f'{self.model_info.model_root_path}/dataset.yaml') else self._create_dataset_yaml()
        if not os.path.exists(f'{self.model_info.model_root_path}/model.engine'):
            subprocess.run(
                f'python3 /yolov5/export.py --device 0 --half --weights {weightfile} --data {data} --imgsz {self.model_info.resolution} --include engine', shell=True)
        self.device = select_device('0')

        self.model = DetectMultiBackend(
            f'{self.model_info.model_root_path}/model.engine', device=self.device, data=data)
        imgz = (model_info.resolution, model_info.resolution)
        self.imgz = check_img_size(imgz, s=self.model.stride)
        self.half = self.model.engine and self.device.type != 'cpu'
        self.model.warmup(imgsz=(1, 3, *self.imgz), half=self.half)

    def evaluate(self, image: List[np.uint8]) -> Detections:
        detections = Detections()
        try:
            start = time_sync()
            im, im0 = self._preprocess_image(image)
            im = torch.from_numpy(im).to(self.device)
            im = im.half()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            results = self.model(im, augment=False, visualize=False)
            skipped_detections = []
            end = time_sync()
            results = non_max_suppression(results, conf_thres=0.2, iou_thres=0.4)[0]  # We have only one image
            if len(results):
                results[:, :4] = scale_coords(im.shape[2:], results[:, :4], im0.shape).round()
                for *xyxy, probability, cls in reversed(results):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    x, y, w, h = xywh
                    x = x - 0.5*w
                    y = y - 0.5*h
                    c = int(cls)
                    category = self.model_info.categories[c]

                    probability = probability.item()
                    if w <= 2 or h <= 2:  # skip very small boxes.
                        detection = (category.name, x, y, w, h, probability)
                        skipped_detections.append((category.name, detection))
                        continue
                    if category.type == CategoryType.Box:
                        detections.box_detections.append(BoxDetection(
                            category.name, x, y, w, h, self.model_info.version, probability))
                    elif category.type == CategoryType.Point:
                        cx, cy = (np.average([x, x + w]), np.average([y, y + h]))
                        detections.point_detections.append(PointDetection(
                            category.name, int(cx), int(cy), self.model_info.version, probability))

            logging.info(f'took {end-start} s, overall {time.time() -start} s')
            if skipped_detections:
                log_msg = '\n'.join([str(d) for d in skipped_detections])
                logging.warning(
                    f'Removed very small detections from inference result (count={len(skipped_detections)}): \n{log_msg}')
        except Exception as e:
            logging.exception('inference failed')
        logging.info(detections)
        return detections

    def _preprocess_image(self, image: List[np.uint8]) -> Tuple[np.ndarray, np.ndarray]:
        img0 = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img = letterbox(img0, self.imgz, stride=self.model.stride, scaleFill=True, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img, img0

    def _create_dataset_yaml(self) -> str:
        data = {
            'nc': len(self.model_info.categories),
            'names': [category.name for category in self.model_info.categories]
        }
        logging.info(data)
        with open(f'{self.model_info.model_root_path}/dataset.yaml', 'w') as f:
            yaml.dump(data, f)

        return f'{self.model_info.model_root_path}/dataset.yaml'
