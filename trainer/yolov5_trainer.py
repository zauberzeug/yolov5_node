from typing import List, Optional
from learning_loop_node import GLOBALS
from learning_loop_node.trainer import Trainer, BasicModel
from learning_loop_node.trainer.model import PretrainedModel
import yolov5_format
import os
import shutil
import json
from glob import glob
import subprocess


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
        cmd = f'WANDB_MODE=disabled python /yolov5/train.py --batch-size {batch_size} --img {resolution} --data dataset.yaml --weights model.pt --project {self.training.training_folder} --name result --hyp hyp.yaml --epochs {epochs} --clear'
        self.executor.start(cmd)

    async def start_training_from_scratch(self, identifier: str) -> None:
        if identifier == 'small':
            resolution = 832
            yolov5_format.create_file_structure(self.training)
            batch_size = 32
            epochs = 1000
            if not os.path.isfile('hpy.yaml'):
                shutil.copy('/app/hyp.yaml', self.training.training_folder)
            cmd = f'WANDB_MODE=disabled python /yolov5/train.py --batch-size {batch_size} --img {resolution} --data dataset.yaml --weights yolov5s.pt --project {self.training.training_folder} --name result --hyp hyp.yaml --epochs {epochs} --clear'
            import logging
            logging.info('going to start with command : ')
            logging.info(cmd)
            self.executor.start(cmd)
        else:
            raise ValueError(f"Pretrained model '{identifier}' is not supported.")
    

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
        def delete_old_weightfiles(model_id: str):
            path = self.training.training_folder + '/result/weights'
            if not os.path.isdir(path):
                return
            files = glob(path + '/*')
            [os.remove(f) for f in files if os.path.isfile(f)]
        path = self.training.training_folder + '/result/weights/published'
        if not os.path.isdir(path):
            os.mkdir(path)
        target = f'{path}/{model_id}.pt'
        shutil.move(basic_model.meta_information['weightfile'], target)
        delete_old_weightfiles(model_id)
        

    def get_model_files(self, model_id: str) -> List[str]:
        weightfile = glob(f'{GLOBALS.data_folder}/**/trainings/**/{model_id}.pt', recursive=True)[0]
        shutil.copy(weightfile, '/tmp/model.pt')
        training_path = '/'.join(weightfile.split('/')[:-4])
        subprocess.run(f'python3 /yolov5/gen_wts.py -w {weightfile} -o /tmp/model.wts', shell=True)
        return {
            self.model_format: ['/tmp/model.pt', f'{training_path}/hyp.yaml', f'{training_path}/model.json'],
            'yolov5_wts': ['/tmp/model.wts', f'{training_path}/model.json']
        }
    
    def clear_training_data(self, model_id: str) -> None:
        keep_files = ['last_training.log', 'best.pt'] # Note: Keep best.pt in case uploaded model was not best.
        keep_dirs = ['result', 'weights']
        weightfile = glob(f'{GLOBALS.data_folder}/**/trainings/**/{model_id}.pt', recursive=True)[0]
        training_path = '/'.join(weightfile.split('/')[:-4])
        for root, dirs, files in os.walk(training_path, topdown=False):
            for file in files:
                if file not in keep_files:
                    os.remove(os.path.join(root, file))
            for dir in dirs:
                if dir not in keep_dirs:
                    os.rmdir(os.path.join(root, dir))


    @property
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        return [
            PretrainedModel(name='small', label='Small', description='')
        ]
