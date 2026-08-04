"""The single declaration of how images become YOLOv5 model input.

The model is trained on images letterboxed by the vendored ``trainer/app_code/yolov5`` code.
Both detectors must reproduce that letterbox exactly - any divergence (pad color, geometry,
resize filter) is silent train/deploy skew.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD_COLOR = 114
"""The gray the training letterbox pads with - detectors padding anything else feed the model unseen borders."""


def letterbox(image: np.ndarray, height: int, width: int) -> tuple[np.ndarray, LetterboxGeometry]:
    """Resize and center-pad a uint8 HWC image to ``(height, width)``, preserving aspect ratio.

    Geometry matches the vendored training letterbox bit for bit: ``int(round(shape * r))`` resize,
    padding split via ``int(round(pad / 2 -/+ 0.1))``, scale-up allowed.
    Filter rule: ``cv2.INTER_AREA`` when downscaling - a proper low-pass that antialiases thin
    structures - and ``cv2.INTER_LINEAR`` when upscaling, where AREA degenerates to nearest-neighbor.
    """
    original_height, original_width = image.shape[:2]
    ratio = min(height / original_height, width / original_width)
    resized_width = round(original_width * ratio)
    resized_height = round(original_height * ratio)
    if (resized_height, resized_width) != (original_height, original_width):
        interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
        image = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    pad_w = (width - resized_width) / 2
    pad_h = (height - resized_height) / 2
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(PAD_COLOR, PAD_COLOR, PAD_COLOR))
    geometry = LetterboxGeometry(original_width=original_width, original_height=original_height,
                                 resized_width=resized_width, resized_height=resized_height,
                                 pad_left=left, pad_top=top)
    return image, geometry


@dataclass(frozen=True, slots=True, kw_only=True)
class LetterboxGeometry:
    """What ``letterbox`` actually did to an image, so detections can be mapped back exactly."""

    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int

    def boxes_to_original(self, boxes: np.ndarray) -> np.ndarray:
        """Map ``(N, 4)`` ``[x1, y1, x2, y2]`` boxes from letterbox space back to original image pixels.

        Inverts the rounded resize and the actual pad split per axis, so boxes stay correct
        even where ``int(round())`` made the effective scale differ slightly between axes.
        """
        result = boxes.astype(np.float32)
        result[:, [0, 2]] = (result[:, [0, 2]] - self.pad_left) * (self.original_width / self.resized_width)
        result[:, [1, 3]] = (result[:, [1, 3]] - self.pad_top) * (self.original_height / self.resized_height)
        return result
