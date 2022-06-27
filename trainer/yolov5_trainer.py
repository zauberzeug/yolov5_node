from asyncio import sleep
import logging
from typing import List, Optional, Tuple
from learning_loop_node import GLOBALS
from learning_loop_node.trainer import Trainer, BasicModel
from learning_loop_node.trainer.model import PretrainedModel
import yolov5_format
import os
import shutil
import json
from glob import glob
import subprocess
from learning_loop_node.trainer.executor import Executor
from learning_loop_node.model_information import ModelInformation
from learning_loop_node.detector.box_detection import BoxDetection
from learning_loop_node.detector.point_detection import PointDetection
import cv2
import asyncio
import re


class Yolov5Trainer(Trainer):

    def __init__(self) -> None:
        super().__init__(model_format='yolov5_pytorch')
        self.latest_epoch = 0
        self.epochs = 2000

    async def start_training(self, model: str = 'model.pt') -> None:
        resolution = self.training.data.hyperparameter.resolution
        yolov5_format.create_file_structure(self.training)
        batch_size = Yolov5Trainer.get_batch_size()
        patience = 300

        hyperparameter_path = f'{self.training.training_folder}/hyp.yaml'
        if not os.path.isfile(hyperparameter_path):
            shutil.copy('/app/hyp.yaml', hyperparameter_path)
        yolov5_format.update_hyp(hyperparameter_path, self.training.data.hyperparameter)

        cmd = f'WANDB_MODE=disabled python /yolov5/train.py --patience {patience} --batch-size {batch_size} --img {resolution} --data dataset.yaml --weights {model} --project {self.training.training_folder} --name result --hyp {hyperparameter_path} --epochs {self.epochs} --clear'
        with open(hyperparameter_path) as f:
            logging.info(f'running training with command :\n {cmd} \and hyperparameter\n{f.read()}')
        self.executor.start(cmd)

    async def start_training_from_scratch(self, id: str) -> None:
        await self.start_training(model=f'{id}.pt')

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
            categories = yolov5_format.category_lookup_from_training(self.training)
            for category_name in list(matrix.keys()):
                matrix[categories[category_name]] = matrix.pop(category_name)

        return BasicModel(confusion_matrix=matrix, meta_information={'weightfile': weightfile})

    def on_model_published(self, basic_model: BasicModel) -> None:
        def delete_old_weightfiles():
            path = self.training.training_folder + '/result/weights'
            if not os.path.isdir(path):
                return
            files = glob(path + '/*')
            [os.remove(f) for f in files if os.path.isfile(f)]
        path = self.training.training_folder + '/result/weights/published'
        if not os.path.isdir(path):
            os.mkdir(path)
        target = f'{path}/latest.pt'
        shutil.move(basic_model.meta_information['weightfile'], target)
        delete_old_weightfiles()

    def get_latest_model_files(self) -> List[str]:
        path = self.training.training_folder + '/result/weights/published'
        weightfile = f'{path}/latest.pt'
        shutil.copy(weightfile, '/tmp/model.pt')
        training_path = '/'.join(weightfile.split('/')[:-4])

        subprocess.run(f'python3 /yolov5/gen_wts.py -w {weightfile} -o /tmp/model.wts', shell=True)
        return {
            self.model_format: ['/tmp/model.pt', f'{training_path}/hyp.yaml'],
            'yolov5_wts': ['/tmp/model.wts']
        }

    async def _detect(self, model_information: ModelInformation, images:  List[str], model_folder: str) -> List:
        images_folder = f'/tmp/imagelinks_for_detecting'
        shutil.rmtree(images_folder, ignore_errors=True)
        os.makedirs(images_folder)
        for img in images:
            image_name = os.path.basename(img)
            os.symlink(img, f'{images_folder}/{image_name}')

        logging.info('start detections')
        shutil.rmtree('/yolov5/runs', ignore_errors=True)
        os.makedirs('/yolov5/runs')
        executor = Executor(images_folder)
        img_size = model_information.resolution

        cmd = f'python /yolov5/detect.py --weights {model_folder}/model.pt --source {images_folder} --img-size {img_size} --conf-thres 0.2 --save-txt --save-conf'
        executor.start(cmd)
        while executor.is_process_running():
            await sleep(1)

        if executor.return_code == 1:
            logging.error('Error during detecting.')

        detections = []
        logging.info('start parsing detections')
        labels_path = '/yolov5/runs/detect/exp/labels'
        detections = await asyncio.get_event_loop().run_in_executor(None, lambda: self._parse(labels_path, images_folder, model_information))

        return detections

    def _parse(self, labels_path, images_folder, model_information):
        detections = []
        if os.path.exists(labels_path):
            for filename in os.scandir(labels_path):
                uuid = os.path.splitext(os.path.basename(filename.path))[0]
                box_detections, point_detections = self._parse_file(model_information, images_folder, filename)
                detections.append({'image_id': uuid, 'box_detections': box_detections,
                                   'point_detections': point_detections})
        return detections

    def _parse_file(self, model_information, images_folder, filename) -> Tuple[BoxDetection, PointDetection]:
        uuid = os.path.splitext(os.path.basename(filename.path))[0]

        image_path = f'{images_folder}/{uuid}.jpg'
        img_height, img_width, _ = cv2.imread(image_path).shape
        with open(filename.path, 'r') as f:
            content = f.readlines()
        box_detections = []
        point_detections = []

        for line in content:
            c, x, y, w, h, probability = line.split(' ')
            c = int(c)
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            probability = float(probability) * 100

            category = model_information.categories[c]
            width = w*img_width
            height = h*img_height
            x = (x*img_width)-0.5*width
            y = (y*img_height)-0.5*height
            if(category.type == 'box'):
                box_detection = BoxDetection(category_name=category.name, x=x, y=y, width=width, height=height, net=model_information.version,
                                             confidence=probability, category_id=category.id)
                box_detections.append(box_detection)
            elif(category.type == 'point'):
                point_detection = PointDetection(category_name=category.name, x=x+width/2, y=y+height/2, net=model_information.version,
                                                 confidence=probability, category_id=category.id)
                point_detections.append(point_detection)
        return box_detections, point_detections

    async def clear_training_data(self, training_folder: str) -> None:
        # Note: Keep best.pt in case uploaded model was not best.
        keep_files = ['last_training.log', 'hyp.yaml', 'dataset.yaml', 'best.pt']
        keep_dirs = ['result', 'weights']
        for root, dirs, files in os.walk(training_folder, topdown=False):
            for file in files:
                if file not in keep_files:
                    os.remove(os.path.join(root, file))
            for dir in dirs:
                if dir not in keep_dirs:
                    os.rmdir(os.path.join(root, dir))

    @property
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        return [
            PretrainedModel(name='s6', label='YOLO v5 small', description='~5 fps on Jetson Nano'),
            # PretrainedModel(name='yolov5m', label='YOLO v5 medium', description='~2 fps on Jetson Nano'),
        ]

    @property
    def model_architecture(self):
        return 'yolov5'

    @property
    def progress(self) -> float:
        return self.get_progress_from_log()

    def get_progress_from_log(self) -> float:
        if self.epochs == 1:
            return 1.0  # NOTE: We would divide by 0 in this case
        lines = list(reversed(self.executor.get_log_by_lines()))
        for line in lines:
            if re.search(f'/{self.epochs -1}', line):
                found_line = line.split(' ')
                for item in found_line:
                    if f'/{self.epochs -1}' in item:
                        epoch_and_total_epochs = item.split('/')
                        epoch = epoch_and_total_epochs[0]
                        total_epochs = epoch_and_total_epochs[1]
                        progress = int(epoch) / int(total_epochs)
                        return progress

    @staticmethod
    def get_batch_size():
        return int(os.environ.get('BATCH_SIZE', '8'))
