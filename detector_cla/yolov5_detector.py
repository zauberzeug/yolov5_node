from typing import List
from learning_loop_node import ModelInformation, Detector
from learning_loop_node.detector import Detections
from learning_loop_node.detector.classification_detection import ClassificationDetection
import logging
import numpy as np
import torch
import cv2
import torch.nn.functional as F
import torchvision.transforms as T


IMAGENET_MEAN = 0.485, 0.456, 0.406
IMAGENET_STD = 0.229, 0.224, 0.225


def classify_transforms(size=832):
    return T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class Yolov5Detector(Detector):

    def __init__(self) -> None:
        super().__init__('yolov5_pytorch')

    def init(self,  model_info: ModelInformation):
        self.model_info = model_info
        self.imgsz = (model_info.resolution, model_info.resolution)
        self.torch_transforms = classify_transforms(self.imgsz)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path=f'{model_info.model_root_path}/model.pt', force_reload=True)

    def evaluate(self, image: List[np.uint8]) -> Detections:
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        detections = Detections()
        try:
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # Perform yolov5 preprocessing
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.torch_transforms(image)
            image = image.unsqueeze(0)

            image = torch.Tensor(image).cuda()
            results = self.model(image)
            pred = F.softmax(results, dim=1)

            top_i = pred[0].argsort(0, descending=True)[:1].tolist()
            if top_i:
                category_index = top_i[0]
                category = [category for category in self.model_info.categories if category.name ==
                            self.model.names[category_index]][0]
                detections.classification_detections.append(ClassificationDetection(
                    category.name, self.model_info.version, pred[0][category_index].item(), category.id
                ))

        except Exception as e:
            logging.exception('inference failed')
        return detections
