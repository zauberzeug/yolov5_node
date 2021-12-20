from learning_loop_node.trainer.trainer_node import TrainerNode
from yolor_trainer import YolorTrainer
import uvicorn
import icecream

icecream.install()

node = TrainerNode(
    uuid='ad04045a-d9b8-414b-a41a-f0cb1f100069',
    name='Yolov5 Trainer',
    trainer=YolorTrainer()
)

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
