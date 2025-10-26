import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from learning_loop_node.data_classes import (
    ClassificationDetection,
    ImageMetadata,
    ImagesMetadata,
)
from learning_loop_node.detector.detector_logic import DetectorLogic

IMAGENET_MEAN = 0.485, 0.456, 0.406
IMAGENET_STD = 0.229, 0.224, 0.225


def classify_transforms(size: Tuple[int, int] = (832, 832)):
    return T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class Yolov5Detector(DetectorLogic):

    def __init__(self) -> None:
        super().__init__('yolov5_pytorch')

    def init(self):
        assert self.model_info is not None, 'model_info must be set before calling init()'
        assert self.model_info.resolution is not None

        self.imgsz = (self.model_info.resolution, self.model_info.resolution)
        self.torch_transforms = classify_transforms(self.imgsz)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path=f'{self.model_info.model_root_path}/model.pt', force_reload=True)

    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        if self.model_info is None or self.model is None:
            return ImageMetadata()

        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        metadata = ImageMetadata()
        try:
            torch_image = self.torch_transforms(image)
            torch_image = torch_image.unsqueeze(0)

            torch_image = torch_image.cuda()
            results = self.model(torch_image)
            pred = F.softmax(results, dim=1)

            top_i = pred[0].argsort(0, descending=True)[:1].tolist()
            if top_i:
                category_index = top_i[0]
                category = [category for category in self.model_info.categories if category.name ==
                            self.model.names[category_index]][0]
                metadata.classification_detections.append(ClassificationDetection(
                    category.name, self.model_info.version, pred[0][category_index].item(), category.id
                ))

        except Exception:
            logging.exception('inference failed')
        return metadata

    def batch_evaluate(self, images: List[np.ndarray]) -> ImagesMetadata:
        raise NotImplementedError('batch_evaluate is not implemented for Yolov5Detector')
