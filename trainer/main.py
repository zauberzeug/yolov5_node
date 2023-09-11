import logging
import os
from code.yolov5_trainer import Yolov5TrainerLogic

import icecream
import uvicorn
from dotenv import load_dotenv
from learning_loop_node import TrainerNode

icecream.install()


load_dotenv()
# check if env variable 'YOLOV5_MODE' is set to 'cla'


logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

trainer_logic = Yolov5TrainerLogic()
node = TrainerNode(name='Yolov5 Trainer ' + os.uname()[1], trainer_logic=trainer_logic)

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
