import logging
from dataclasses import asdict

import numpy as np
import torch
from fastapi.encoders import jsonable_encoder
from learning_loop_node.data_classes import Category, ModelInformation
from learning_loop_node.enums import CategoryType

from ..yolov5_detector import Yolov5Detector

# pylint: disable=protected-access


def _make_detector(input_size: int = 0) -> Yolov5Detector:
    """Create a Yolov5Detector without running the full __init__ (no model loading)."""
    det = object.__new__(Yolov5Detector)
    det.input_size = input_size
    return det


def test_postprocess_empty():
    boxes, scores, class_id = _make_detector()._post_process(np.empty(shape=(0, 8)), 100, 100, 0.2, 0.45)
    assert len(boxes) == 0
    assert len(scores) == 0
    assert len(class_id) == 0


def test_postprocess_conf_thresh_filtered_conf():
    data = np.array([[0, 0, 10, 10, 0.1, 0.8, 0.8]])
    boxes, scores, class_id = _make_detector()._post_process(data, 100, 100, 0.2, 0.45)
    assert len(boxes) == 0
    assert len(scores) == 0
    assert len(class_id) == 0


def test_postprocess_conf_thresh_filtered_iou():
    data = np.array(
        [[0.5, 0, 0.1, 0.1, 0.95],
         [0.5, 0, 0.1, 0.11, 0.9]]
    )
    boxes, scores, class_id = _make_detector(input_size=100)._post_process(data, 100, 100, 0.2, 0.45)
    assert len(boxes) == 1
    assert len(scores) == 1
    assert len(class_id) == 1


def test_postprocess_conf_thresh_not_filtered():
    data = np.array(
        [[0.5, 0, 0.1, 0.1, 0.95],
         [0, 0.5, 0.1, 0.1, 0.9]]
    )
    boxes, scores, class_id = _make_detector(input_size=100)._post_process(data, 100, 100, 0.2, 0.45)
    assert len(boxes) == 2
    assert len(scores) == 2
    assert len(class_id) == 2


def test_evaluate_returns_serializable_point_detections():
    """Point coordinates must be Python floats, not numpy scalars.

    The node library sends detections with fastapi's jsonable_encoder, which fails on numpy.float32.
    """
    input_size = 100
    det = _make_point_detector(input_size)

    image = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    metadata = det.evaluate(image)

    assert len(metadata.point_detections) == 1
    point = metadata.point_detections[0]
    assert type(point.x) is float
    assert type(point.y) is float
    assert type(point.confidence) is float
    assert point.category_name == 'Rechts'
    jsonable_encoder(asdict(metadata))  # must not raise


def test_batch_evaluate_matches_evaluate_per_image() -> None:
    det = _make_point_detector(input_size=100)
    images = [np.zeros((100, 100, 3), dtype=np.uint8), np.zeros((50, 200, 3), dtype=np.uint8)]

    assert ([item.point_detections for item in det.batch_evaluate(images).items]
            == [det.evaluate(image).point_detections for image in images])
    assert det.batch_evaluate([]).items == []


def _make_point_detector(input_size: int) -> Yolov5Detector:
    categories = [Category(id='a', name='Links', type=CategoryType.Point, point_size=10),
                  Category(id='b', name='Rechts', type=CategoryType.Point, point_size=10)]
    det = _make_detector(input_size=input_size)
    det.model_info = ModelInformation(id='m', host='', organization='', project='',
                                      version='1.0', categories=categories, resolution=input_size)
    det.log = logging.getLogger('test')
    det.conf_threshold = 0.2
    det.iou_threshold = 0.45
    # one prediction in the model output format [cx, cy, w, h, conf, class0, class1] as float32
    pred = torch.tensor([[[[50.0, 40.0, 10.0, 10.0, 0.95, 0.1, 0.9]]]], dtype=torch.float32)
    det.yolov5 = lambda _: pred
    return det
