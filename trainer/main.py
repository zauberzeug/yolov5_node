import logging
import os

import icecream
import uvicorn
from learning_loop_node import TrainerNode

from app_code.yolov5_trainer import Yolov5TrainerLogic

icecream.install()


# check if env variable 'YOLOV5_MODE' is set to 'cla'
print(f'Uvicorn reload is set to: {os.getenv("UVICORN_RELOAD", "FALSE").lower() == "true"}')
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)

trainer_logic = Yolov5TrainerLogic()
node = TrainerNode(name='Yolov5 Trainer ' + os.uname()[1], trainer_logic=trainer_logic)

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on',
                reload=os.getenv('UVICORN_RELOAD', 'FALSE').lower() == 'true')
