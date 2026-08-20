from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import final, override

import cv2  # type: ignore # pylint: disable=import-error
import numpy as np
import torch  # type: ignore # pylint: disable=import-error
from learning_loop_node import DetectorLogic, DetectorLogicFactory
from learning_loop_node.data_classes import (
    ImageMetadata,
    ImagesMetadata,
    ModelInformation,
)
from learning_loop_node.detector.postprocess import bbox_iou, detections_from_xyxy, to_image_metadata

MAX_DETECTIONS = 1000


@final
@dataclass(frozen=True)
class Yolov5DetectorParams(DetectorLogicFactory):
    iou_threshold: float
    conf_threshold: float

    @property
    @override
    def model_format(self) -> str:
        return 'yolov5_pytorch'

    @override
    async def build(self, model_info: ModelInformation) -> Yolov5Detector:
        return await asyncio.to_thread(Yolov5Detector, model_info, self)


@final
class Yolov5Detector(DetectorLogic):

    def __init__(self, model_info: ModelInformation, params: Yolov5DetectorParams) -> None:
        self.model_info = model_info
        self.log = logging.getLogger('Yolov5Detector')
        self.log.setLevel(logging.INFO)
        self.iou_threshold = params.iou_threshold
        self.conf_threshold = params.conf_threshold

        if not isinstance(model_info.resolution, int) or model_info.resolution <= 0:
            raise RuntimeError("model_info.resolution must be an integer > 0")
        self.input_size: int = model_info.resolution

        pt_file = f'{model_info.model_root_path}/model.pt'
        yolov5_path = os.path.join(os.path.dirname(__file__), 'app_code', 'yolov5')
        self.yolov5 = torch.hub.load(yolov5_path, 'custom', pt_file, source='local')

        if self.yolov5 is None:
            raise RuntimeError('Failed to load YOLOv5 model')

        self.yolov5.eval()

    @override
    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        try:
            t = time.time()
            input_image, origin_h, origin_w = self._preprocess_image(image)

            det = self.yolov5(torch.from_numpy(input_image))[0].numpy()[0]
            if len(det) == 0:
                return ImageMetadata()

            result_boxes, result_scores, result_classid = self._post_process(
                det, origin_h, origin_w, self.conf_threshold, self.iou_threshold)

            detections = detections_from_xyxy(
                labels=result_classid[:MAX_DETECTIONS].tolist(),
                boxes=result_boxes[:MAX_DETECTIONS].tolist(),
                scores=[round(float(score), 2) for score in result_scores[:MAX_DETECTIONS]],
            )
            self.log.debug('took %f s', time.time() - t)
            return to_image_metadata(detections, self.model_info, origin_h, origin_w)

        except Exception as e:
            raise RuntimeError('Error during inference') from e

    def _preprocess_image(self, image_raw):
        """
        description: resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            h: original height
            w: original width
        """
        input_size = self.input_size
        h, w, _ = image_raw.shape
        # Calculate widht and height and paddings
        r_w = input_size / w
        r_h = input_size / h
        if r_h > r_w:
            tw = input_size
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((input_size - th) / 2)
            ty2 = input_size - th - ty1
        else:
            tw = int(r_h * w)
            th = input_size
            tx1 = int((input_size - tw) / 2)
            tx2 = input_size - tw - tx1
            ty1 = ty2 = 0

        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image_raw, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
        image = image.astype(np.float32)
        image /= 255.0  # Normalize to [0,1]
        image = np.transpose(image, [2, 0, 1])  # HWC to CHW format:
        image = np.expand_dims(image, axis=0)  # CHW to NCHW format
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)

        return image, h, w

    def _post_process(self, pred, origin_h, origin_w, conf_thres, nms_thres):
        """
        description: postprocess the prediction
        param:
            pred:     A numpy likes [[cx,cy,w,h,conf, c0_prob, c1_prob, ...], 
                                     [cx,cy,w,h,conf, c0_prob, c1_prob, ...], ...] 
            origin_h:   height of original image
            origin_w:   width of original image
            conf_thres: confidence threshold
            nms_thres: iou threshold
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """

        num_classes = pred.shape[1] - 5

        # Do nms
        boxes = self._non_max_suppression(
            pred, origin_h, origin_w, conf_thres, nms_thres)
        if len(boxes) == 0:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

        result_boxes = boxes[:, :4]
        result_scores = boxes[:, 4]
        if num_classes > 1:
            result_classid = np.argmax(boxes[:, 5:], axis=1)
        else:
            # Without classes there is no classid to return
            result_classid = np.zeros(boxes.shape[0], dtype=int)
        return result_boxes, result_scores, result_classid

    def _non_max_suppression(self, pred, origin_h, origin_w, conf_thres, nms_thres):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: A numpy likes [[cx,cy,w,h,conf, c0_prob, c1_prob, ...], 
                                       [cx,cy,w,h,conf, c0_prob, c1_prob, ...], ...] 
            origin_h: original image height
            origin_w: original image width
            input_size: the input size of the model
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = pred[pred[:, 4] >= conf_thres]
        if len(boxes) == 0:
            return np.array([])
        num_classes = boxes.shape[1] - 5
        # Transform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = bbox_iou(np.expand_dims(
                boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            if num_classes > 1:
                label_match = np.argmax(boxes[:, 5:], axis=1) == np.argmax(boxes[0, 5:])
            else:
                label_match = np.ones(boxes.shape[0], dtype=bool)
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        input_size = self.input_size
        y = np.zeros_like(x)
        r_w = input_size / origin_w
        r_h = input_size / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - \
                (input_size - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - \
                (input_size - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - \
                (input_size - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - \
                (input_size - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    @override
    def batch_evaluate(self, images: list[np.ndarray]) -> ImagesMetadata:
        raise NotImplementedError('batch_evaluate is not implemented yet')
