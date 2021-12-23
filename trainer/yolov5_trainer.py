from typing import List, Optional
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.model import BasicModel
import yolov5_format
import os
import shutil


class Yolov5Trainer(Trainer):

    def __init__(self) -> None:
        super().__init__(model_format='yolov5_pytorch')

    async def start_training(self) -> None:
        resolution = 832
        yolov5_format.create_file_structure(self.training)
        batch_size = 4  # batch size 1 takes already 6 GB on 1280x1280
        epochs = 10
        if not os.path.isfile('hpy.yaml'):
            shutil.copy('/app/hyp.yaml', self.training.training_folder)
        cmd = f'WANDB_MODE=disabled python /yolov5/train.py --batch-size {batch_size} --img {resolution} --data dataset.yaml --weights model.pt --project {self.training.training_folder} --name result --hyp hyp.yaml --epochs {epochs}'
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
        return None  # BasicModel(confusion_matrix={}, meta_information={})

    def on_model_published(self, basic_model: BasicModel, uuid: str) -> None:
        pass
        # tbd.

    def stop_training(self) -> None:
        self.executor.stop()
