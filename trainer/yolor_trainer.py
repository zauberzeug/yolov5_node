from typing import List, Optional
from learning_loop_node.trainer.trainer import Trainer
from learning_loop_node.trainer.capability import Capability
from learning_loop_node.trainer.model import BasicModel
import yolor


class YolorTrainer(Trainer):

    def __init__(self) -> None:
        super().__init__(capability=Capability.Box, model_format='yolor')

    async def start_training(self) -> None:
        yolor.start(self.training)

    def is_training_alive(self) -> bool:
        return True  # tbd.

    def get_model_files(self, model_id) -> List[str]:
        return []  # tbd.

    def get_new_model(self) -> Optional[BasicModel]:
        return None  # BasicModel(confusion_matrix={}, meta_information={})

    def on_model_published(self, basic_model: BasicModel, uuid: str) -> None:
        pass
        # tbd.

    def stop_training(self) -> None:
        pass
        # tbd.
