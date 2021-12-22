import uvicorn
from fastapi import APIRouter, Request, File, UploadFile
from fastapi.encoders import jsonable_encoder
from typing import Optional, List
import numpy as np
from fastapi_utils.tasks import repeat_every
from fastapi_socketio import SocketManager
from learning_loop_node import DetectorNode
import asyncio
import logging
from yolov5_detector import Yolov5Detector
import icecream
icecream.install()

logging.getLogger().setLevel(logging.INFO)


detector = Yolov5Detector()
node = DetectorNode(name='Yolov5 detector ' + os.uname()[1], detector=detector)

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
