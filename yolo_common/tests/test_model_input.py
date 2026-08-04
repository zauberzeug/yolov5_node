"""The letterbox contract: geometry, pad color, filter rule and box back-projection."""

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from yolo_common.model_input import PAD_COLOR, LetterboxGeometry, letterbox

VENDORED_AUGMENTATIONS = Path(__file__).parents[2] / 'trainer' / 'app_code' / 'yolov5' / 'utils' / 'augmentations.py'


def _random_image(height: int, width: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


# --- geometry ---

def test_tall_image_is_padded_left_and_right() -> None:
    image = np.full((64, 32, 3), 200, dtype=np.uint8)

    result, geometry = letterbox(image, 64, 64)

    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8
    assert geometry == LetterboxGeometry(original_width=32, original_height=64,
                                         resized_width=32, resized_height=64, pad_left=16, pad_top=0)
    assert (result[:, :16] == PAD_COLOR).all()
    assert (result[:, 16:48] == 200).all()
    assert (result[:, 48:] == PAD_COLOR).all()


def test_odd_padding_puts_the_smaller_half_first() -> None:
    """dh = 9.5 must split into top 9 / bottom 10 - the ±0.1 rounding from the vendored letterbox."""
    image = np.full((13, 32, 3), 200, dtype=np.uint8)

    result, geometry = letterbox(image, 32, 32)

    assert geometry.pad_top == 9
    assert (result[:9] == PAD_COLOR).all()
    assert (result[9:22] == 200).all()
    assert (result[22:] == PAD_COLOR).all()


def test_resized_extent_is_rounded_not_truncated() -> None:
    """20 * (16 / 30) = 10.67 must become 11, not 10 - int() truncation was the old detector bug."""
    image = _random_image(30, 20)

    result, geometry = letterbox(image, 16, 16)

    assert result.shape == (16, 16, 3)
    assert geometry.resized_width == 11
    assert geometry.resized_height == 16
    assert geometry.pad_left == 2  # dw = 2.5 splits into 2 / 3


def test_non_square_target_size() -> None:
    image = _random_image(10, 10)

    result, geometry = letterbox(image, 32, 64)

    assert result.shape == (32, 64, 3)
    assert geometry.resized_width == 32
    assert geometry.resized_height == 32
    assert geometry.pad_left == 16
    assert geometry.pad_top == 0


# --- bit parity with the vendored training letterbox ---

def _vendored_letterbox() -> Callable[..., Any]:
    """Extract the vendored function via AST because importing its module would drag in torch."""
    module = ast.parse(VENDORED_AUGMENTATIONS.read_text())
    function = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'letterbox')
    namespace: dict[str, Any] = {'cv2': cv2, 'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(VENDORED_AUGMENTATIONS), 'exec'), namespace)  # pylint: disable=exec-used
    return namespace['letterbox']


@pytest.mark.parametrize('size', [(13, 32), (100, 200), (7, 5), (640, 640), (33, 47), (479, 653)])
@pytest.mark.parametrize('target', [(64, 64), (48, 64), (640, 640)])
def test_letterbox_matches_the_vendored_training_letterbox(size: tuple[int, int], target: tuple[int, int]) -> None:
    image = _random_image(*size)

    ours, _ = letterbox(image, *target)
    vendored, _, _ = _vendored_letterbox()(image, target, color=(114, 114, 114), auto=False, scaleup=True)

    assert np.array_equal(ours, vendored)


# --- filter rule ---

def test_downscale_antialiases_one_pixel_lines() -> None:
    """INTER_AREA keeps the energy of 1px lines; plain INTER_LINEAR would drop them entirely."""
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    image[:, ::3] = 255

    result, _ = letterbox(image, 32, 32)

    assert result.mean() > 50  # lines average into ~85 gray instead of vanishing
    aliased = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
    assert aliased.mean() < 5  # the rule exists because LINEAR samples right past every line


def test_upscale_interpolates_instead_of_replicating_pixels() -> None:
    """INTER_LINEAR on upscale - INTER_AREA would degenerate to blocky nearest-neighbor."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, 4:] = 255

    result, geometry = letterbox(image, 64, 64)

    assert geometry.resized_width == 64
    assert np.any((result > 0) & (result < 255))  # gradient across the edge, not a hard step
    assert np.array_equal(result, cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR))


# --- box back-projection ---

def test_boxes_round_trip_between_original_and_letterbox_space() -> None:
    _, geometry = letterbox(_random_image(100, 200), 64, 64)

    boxes_original = np.array([[10.0, 20.0, 150.0, 90.0], [0.0, 0.0, 200.0, 100.0]], dtype=np.float32)
    forward = boxes_original.copy()
    forward[:, [0, 2]] = forward[:, [0, 2]] * geometry.resized_width / geometry.original_width + geometry.pad_left
    forward[:, [1, 3]] = forward[:, [1, 3]] * geometry.resized_height / geometry.original_height + geometry.pad_top

    assert np.allclose(geometry.boxes_to_original(forward), boxes_original, atol=1e-3)


def test_the_content_region_maps_back_to_the_full_image() -> None:
    _, geometry = letterbox(_random_image(13, 32), 32, 32)

    content_region = np.array([[0.0, 9.0, 32.0, 22.0]], dtype=np.float32)

    assert np.allclose(geometry.boxes_to_original(content_region), [[0.0, 0.0, 32.0, 13.0]])


def test_back_projection_does_not_mutate_the_input() -> None:
    geometry = LetterboxGeometry(original_width=100, original_height=100,
                                 resized_width=50, resized_height=50, pad_left=7, pad_top=7)
    boxes = np.array([[10.0, 10.0, 20.0, 20.0]], dtype=np.float32)

    geometry.boxes_to_original(boxes)

    assert np.array_equal(boxes, [[10.0, 10.0, 20.0, 20.0]])
