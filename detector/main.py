import os

import uvicorn
from learning_loop_node import DetectorNode

from yolov5_detector import Yolov5DetectorParams

reload = os.getenv("UVICORN_RELOAD", "FALSE").lower() == "true"
print(f'Uvicorn reload is set to: {reload}')

weight_type = os.getenv('WEIGHT_TYPE', 'FP16')
assert weight_type in ('FP16', 'FP32', 'INT8'), 'WEIGHT_TYPE must be one of FP16, FP32, INT8'

MAX_BUILDER_OPTIMIZATION_LEVEL = 5
builder_optimization_level = int(os.getenv('BUILDER_OPTIMIZATION_LEVEL', '4'))
assert 0 <= builder_optimization_level <= MAX_BUILDER_OPTIMIZATION_LEVEL, \
    f'BUILDER_OPTIMIZATION_LEVEL must be between 0 and {MAX_BUILDER_OPTIMIZATION_LEVEL}'

params = Yolov5DetectorParams(
    weight_type=weight_type,
    iou_threshold=float(os.getenv('IOU_THRESHOLD', '0.45')),
    conf_threshold=float(os.getenv('CONF_THRESHOLD', '0.2')),
    builder_optimization_level=builder_optimization_level,
)
node = DetectorNode(name='YOLOv5 Detector ' + os.uname()[1], detector_factory=params)

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=reload)
