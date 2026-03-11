import os

import uvicorn
from learning_loop_node import DetectorNode

from yolov5_detector import Yolov5DetectorParams

reload = os.getenv("UVICORN_RELOAD", "FALSE").lower() == "true"
print(f'Uvicorn reload is set to: {reload}')

params = Yolov5DetectorParams(
    iou_threshold=float(os.getenv('IOU_THRESHOLD', '0.45')),
    conf_threshold=float(os.getenv('CONF_THRESHOLD', '0.2')),
)
node = DetectorNode(name=os.uname()[1], detector_factory=params)

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=reload)
