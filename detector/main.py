import uvicorn
from learning_loop_node import DetectorNode
from yolov5_detector import Yolov5Detector
import logging
import os
import icecream
icecream.install()

logging.getLogger().setLevel(logging.INFO)

detector = Yolov5Detector()
node = DetectorNode(name=os.uname()[1], detector=detector)

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
