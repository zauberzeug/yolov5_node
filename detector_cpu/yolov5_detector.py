import logging
import os
import time
from typing import Tuple

import cv2
import numpy as np
import torch  # type: ignore # pylint: disable=import-error
from learning_loop_node.data_classes import BoxDetection, ImageMetadata, PointDetection
from learning_loop_node.detector.detector_logic import DetectorLogic
from learning_loop_node.enums import CategoryType


class Yolov5Detector(DetectorLogic):

    def __init__(self) -> None:
        super().__init__('yolov5_pytorch')
        self.yolov5 = None
        self.input_size: int = 0
        self.log = logging.getLogger('Yolov5Detector')
        self.log.setLevel(logging.INFO)

    def init(self) -> None:
        resolution = self.model_info.resolution
        assert resolution is not None
        pt_file = f'{self.model_info.model_root_path}/model.pt'

        yolov5_path = os.path.join(
            os.path.dirname(__file__), 'app_code', 'yolov5')
        self.yolov5 = torch.hub.load(
            yolov5_path, 'custom', pt_file, source='local')
        assert self.yolov5 is not None
        self.yolov5.eval()

    @staticmethod
    def clip_box(
            x: float, y: float, width: float, height: float, img_width: int, img_height: int) -> Tuple[
            float, float, float, float]:
        '''make sure the box is within the image
            x,y is the center of the box
        '''
        left = max(0, x - 0.5 * width)
        top = max(0, y - 0.5 * height)
        right = min(img_width, x + 0.5 * width)
        bottom = min(img_height, y + 0.5 * height)

        x = 0.5 * (left + right)
        y = 0.5 * (top + bottom)
        width = right - left
        height = bottom - top

        return x, y, width, height

    @staticmethod
    def clip_point(x: float, y: float, img_width: int, img_height: int) -> Tuple[float, float]:
        x = min(max(0, x), img_width)
        y = min(max(0, y), img_height)
        return x, y

    def evaluate(self, image: np.ndarray) -> ImageMetadata:
        assert self.yolov5 is not None, 'init() must be executed first!'
        image_metadata = ImageMetadata()
        assert self.model_info.resolution, "input_size must be an integer > 0"
        self.input_size = self.model_info.resolution

        try:
            t = time.time()
            cv_image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # Returns BGR
            input_image, _, origin_h, origin_w = self._preprocess_image(cv_image)

            im_height = origin_h
            im_width = origin_w

            # Inference
            det = self.yolov5(torch.from_numpy(input_image))[0].numpy()[0]

            # NMS with lower confidence threshold
            conf_thres = 0.2  # Lowered from 0.25
            iou_thres = 0.4
            max_det = 1000

            if len(det):
                result_boxes, result_scores, result_classid = self._post_process(
                    det, origin_h, origin_w, conf_thres, iou_thres)

                for j, box in enumerate(result_boxes[:max_det]):
                    x, y, br_x, br_y = box
                    w = br_x - x
                    h = br_y - y
                    category_idx = int(result_classid[j])
                    category = self.model_info.categories[category_idx]
                    conf = round(float(result_scores[j]), 2)

                    if category.type == CategoryType.Box:
                        x, y, w, h = self.clip_box(
                            x, y, w, h, im_width, im_height)
                        image_metadata.box_detections.append(
                            BoxDetection(category_name=category.name,
                                         x=round(x),
                                         y=round(y),
                                         width=round(w),
                                         height=round(h),
                                         category_id=category.id,
                                         model_name=self.model_info.version,
                                         confidence=float(conf)))

                    elif category.type == CategoryType.Point:
                        cx, cy = x + w/2, y + h/2
                        cx, cy = self.clip_point(cx, cy, im_width, im_height)
                        image_metadata.point_detections.append(
                            PointDetection(category_name=category.name,
                                           x=cx,
                                           y=cy,
                                           category_id=category.id,
                                           model_name=self.model_info.version,
                                           confidence=float(conf)))

            self.log.debug('took %f s', time.time() - t)
            return image_metadata

        except Exception:
            self.log.exception('inference failed')
            return image_metadata

    def _preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        input_size = self.input_size
        image_raw = raw_bgr_image
        h, w, _ = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
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
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128))
        image = image.astype(np.float32)
        image /= 255.0  # Normalize to [0,1]
        image = np.transpose(image, [2, 0, 1])  # HWC to CHW format:
        image = np.expand_dims(image, axis=0)  # CHW to NCHW format
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)

        return image, image_raw, h, w

    def _post_process(self, pred, origin_h, origin_w, conf_thres, nms_thres):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [[cx,cy,w,h,conf,cls_id], [cx,cy,w,h,conf,cls_id], ...] 
            origin_h:   height of original image
            origin_w:   width of original image
            conf_thres: confidence threshold
            nms_thres: iou threshold
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """

        # Do nms
        boxes = self._non_max_suppression(
            pred, origin_h, origin_w, conf_thres, nms_thres)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def _non_max_suppression(self, pred, origin_h, origin_w, conf_thres, nms_thres):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: A numpy likes [[cx,cy,w,h,conf,cls_id], [cx,cy,w,h,conf,cls_id], ...] 
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
        # Trasform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(
                boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1].astype(int) == boxes[:, -1].astype(int)
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / \
                2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / \
                2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / \
                2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / \
                2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                              0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                              0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
            np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

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
