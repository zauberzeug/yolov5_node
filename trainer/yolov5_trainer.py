from typing import List, Optional
from learning_loop_node import GLOBALS
from learning_loop_node.trainer import Trainer, BasicModel
import yolov5_format
import os
import shutil
import json
from glob import glob


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

    def get_new_model(self) -> Optional[BasicModel]:
        path = self.training.training_folder + '/result/weights'
        if not os.path.isdir(path):
            return
        weightfiles = [os.path.join(path, f) for f in os.listdir(path) if 'epoch' in f and f.endswith('.pt')]
        if not weightfiles:
            return
        weightfile = sorted(weightfiles)[0]
        # NOTE /yolov5 is patched to create confusion matrix json files
        with open(weightfile[:-3] + '.json') as f:
            matrix = json.load(f)
            for category_name in list(matrix.keys()):
                matrix[self.training.data.categories[category_name]] = matrix.pop(category_name)

        return BasicModel(confusion_matrix=matrix, meta_information={'weightfile': weightfile})

    def on_model_published(self, basic_model: BasicModel, model_id: str) -> None:
        target = self.training.training_folder + f'/result/weights/{model_id}.pt'
        shutil.move(basic_model.meta_information['weightfile'], target)

    def get_model_files(self, model_id) -> List[str]:
        weightfile = glob(f'{GLOBALS.data_folder}/**/trainings/**/{model_id}.pt', recursive=True)[0]
        shutil.copy(weightfile, '/tmp/model.pt')
        training_path = '/'.join(weightfile.split('/')[:-2])
        formats = {}
        formats[self.model_format] = [weightfile, f'{training_path}/hyp.yaml']
        return formats
