from typing import List, Optional
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.model import BasicModel
import yolov5_format
import os
import shutil
import json


class Yolov5Trainer(Trainer):

    def __init__(self) -> None:
        super().__init__(model_format='yolov5_pytorch')
        self.latest_epoch = 0

    async def start_training(self) -> None:
        resolution = 832
        yolov5_format.create_file_structure(self.training)
        batch_size = 32
        epochs = 1000
        if not os.path.isfile('hpy.yaml'):
            shutil.copy('/app/hyp.yaml', self.training.training_folder)
        cmd = f'WANDB_MODE=disabled python /yolov5/train.py --batch-size {batch_size} --img {resolution} --data dataset.yaml --save-period 2 --weights model.pt --project {self.training.training_folder} --name result --hyp hyp.yaml --epochs {epochs}'
        ic(cmd)
        self.executor.start(cmd)

    def get_error(self) -> str:
        if self.executor is None:
            return
        try:
            if 'CUDA Error: out of memory' in self.executor.get_log():
                return 'graphics card is out of memory'
        except:
            return

    def get_model_files(self, model_id) -> List[str]:
        return []  # tbd.

    def get_new_model(self) -> Optional[BasicModel]:
        path = 'result/weights'
        if not os.path.isdir(path):
            return
        weightfiles = [os.path.join(path, f) for f in os.listdir(path) if 'epoch' in f and f.endswith('.pt')]
        if not weightfiles:
            return
        weightfile = sorted(weightfiles)[0]
        # NOTE /yolov5 is patched to create confusion matrix json files
        with open(weightfile[:-3] + '.json') as f:
            return BasicModel(confusion_matrix=json.load(f), meta_information={'weightfile': weightfile})

    def on_model_published(self, basic_model: BasicModel, uuid: str) -> None:
        shutil.move(basic_model.meta_information['weightfile'], f'result/weights/{uuid}.pt')

    def stop_training(self) -> None:
        self.executor.stop()
