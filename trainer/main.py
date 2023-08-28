import logging
import os

import icecream
import uvicorn
from learning_loop_node import TrainerNode

from yolov5_trainer import Yolov5Trainer

icecream.install()

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

node = TrainerNode(name='Yolov5 Trainer ' + os.uname()[1], trainer=Yolov5Trainer())

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
