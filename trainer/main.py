from learning_loop_node import TrainerNode
from yolov5_trainer import Yolov5Trainer
import uvicorn
import os
import icecream
icecream.install()

node = TrainerNode(name='Yolov5 Trainer ' + os.uname()[1], trainer=Yolov5Trainer())

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
