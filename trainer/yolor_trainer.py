from typing import List, Optional
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.model import BasicModel
import yolor_format


class YolorTrainer(Trainer):

    def __init__(self) -> None:
        super().__init__(model_format='yolor')

    async def start_training(self) -> None:
        resolution = 800
        yolor_format.create_file_structure(self.training)

        batch_size = 4  # batch size 1 takes already 6 GB on 1280x1280
        epochs = 10
        # from https://github.com/WongKinYiu/yolor#training
        cmd = f'python /yolor/train.py --batch-size {batch_size} --img {resolution} {resolution} --data dataset.yaml --cfg model.yaml --weights model.pt --device 0 --name yolor --hyp /yolor/data/hyp.finetune.1280.yaml --epochs {epochs}'

        self.executor.start(cmd)

    def is_training_alive(self) -> bool:
        return self.executor.is_process_running()

    def get_model_files(self, model_id) -> List[str]:
        return []  # tbd.

    def get_new_model(self) -> Optional[BasicModel]:
        return None  # BasicModel(confusion_matrix={}, meta_information={})

    def on_model_published(self, basic_model: BasicModel, uuid: str) -> None:
        pass
        # tbd.

    def stop_training(self) -> None:
        self.executor.stop()
