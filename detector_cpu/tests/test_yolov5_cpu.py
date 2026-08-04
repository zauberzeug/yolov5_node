import numpy as np
from yolo_common.model_input import LetterboxGeometry

from ..yolov5_detector import Yolov5Detector

# pylint: disable=protected-access


def _make_detector(input_size: int = 0) -> Yolov5Detector:
    """Create a Yolov5Detector without running the full __init__ (no model loading)."""
    det = object.__new__(Yolov5Detector)
    det.input_size = input_size
    return det


def _geometry(size: int = 100) -> LetterboxGeometry:
    """A no-op letterbox: original image already matches the model input size."""
    return LetterboxGeometry(original_width=size, original_height=size,
                             resized_width=size, resized_height=size, pad_left=0, pad_top=0)


def test_postprocess_empty():
    boxes, scores, class_id = _make_detector()._post_process(np.empty(shape=(0, 8)), _geometry(), 0.2, 0.45)
    assert len(boxes) == 0
    assert len(scores) == 0
    assert len(class_id) == 0


def test_postprocess_conf_thresh_filtered_conf():
    data = np.array([[0, 0, 10, 10, 0.1, 0.8, 0.8]])
    boxes, scores, class_id = _make_detector()._post_process(data, _geometry(), 0.2, 0.45)
    assert len(boxes) == 0
    assert len(scores) == 0
    assert len(class_id) == 0


def test_postprocess_conf_thresh_filtered_iou():
    data = np.array(
        [[0.5, 0, 0.1, 0.1, 0.95],
         [0.5, 0, 0.1, 0.11, 0.9]]
    )
    boxes, scores, class_id = _make_detector(input_size=100)._post_process(data, _geometry(100), 0.2, 0.45)
    assert len(boxes) == 1
    assert len(scores) == 1
    assert len(class_id) == 1


def test_postprocess_conf_thresh_not_filtered():
    data = np.array(
        [[0.5, 0, 0.1, 0.1, 0.95],
         [0, 0.5, 0.1, 0.1, 0.9]]
    )
    boxes, scores, class_id = _make_detector(input_size=100)._post_process(data, _geometry(100), 0.2, 0.45)
    assert len(boxes) == 2
    assert len(scores) == 2
    assert len(class_id) == 2


def test_preprocess_letterboxes_to_the_model_input_size():
    image = np.full((50, 100, 3), 200, dtype=np.uint8)

    input_image, geometry = _make_detector(input_size=100)._preprocess_image(image)

    assert input_image.shape == (1, 3, 100, 100)
    assert input_image.dtype == np.float32
    assert geometry == LetterboxGeometry(original_width=100, original_height=50,
                                         resized_width=100, resized_height=50, pad_left=0, pad_top=25)
    assert np.allclose(input_image[0, :, :25, :], 114 / 255)  # padded with training gray, not 128
    assert np.allclose(input_image[0, :, 25:75, :], 200 / 255)
