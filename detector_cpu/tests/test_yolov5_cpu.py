import numpy as np

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
