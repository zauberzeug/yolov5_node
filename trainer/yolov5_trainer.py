from asyncio import sleep
import logging
from typing import Dict, List, Optional, Tuple, Union
from learning_loop_node import GLOBALS
from learning_loop_node.trainer import Trainer, BasicModel
from learning_loop_node.trainer.model import PretrainedModel
import yolov5_format
import os
import shutil
import json
from learning_loop_node.trainer.executor import Executor
from learning_loop_node.model_information import ModelInformation
from learning_loop_node.detector.classification_detection import ClassificationDetection
import cv2
import asyncio
import re
import model_files


class Yolov5Trainer(Trainer):

    def __init__(self) -> None:
        super().__init__(model_format='yolov5_pytorch')
        self.latest_epoch = 0
        self.epochs = 1000

    async def start_training_from_scratch(self, id: str) -> None:
        await self.start_training(model=f'yolov5{id}.pt')

    async def start_training(self, model: str = 'model.pt') -> None:
        yolov5_format.create_file_structure(self.training)
        if model == 'model.pt':
            model = f'{self.training.training_folder}/model.pt'
        await self._start(model)

    async def _start(self, model: str, additional_parameters: str = ''):
        resolution = self.training.data.hyperparameter.resolution

        # batch_size = await batch_size_calculation.calc(self.training.training_folder, model, hyperparameter_path, f'{self.training.training_folder}/dataset.yaml', resolution)
        batch_size = 4
        cmd = f'python /yolov5/classify/train.py --exist-ok --batch-size {batch_size} --img {resolution} --data {self.training.training_folder} --model {model} --project {self.training.training_folder} --name result --epochs {self.epochs} {additional_parameters}'
        self.executor.start(cmd)

    def can_resume(self) -> bool:
        path = f'{self.training.training_folder}/result/weights/published/latest.pt'
        return os.path.exists(path)

    async def resume(self) -> None:
        logging.info('resume called')
        await self._start(f'{self.training.training_folder}/result/weights/published/latest.pt')

    def get_error(self) -> str:
        if self.executor is None:
            return

        for line in self.executor.get_log_by_lines(since_last_start=True):
            if 'CUDA out of memory' in line:
                return 'graphics card is out of memory'
            if 'CUDA error: invalid device ordinal' in line:
                return 'graphics card not found'
        return None

    def get_new_model(self) -> Optional[BasicModel]:
        weightfile = model_files.get_new(self.training.training_folder)
        logging.info(weightfile)
        if not weightfile:
            return None
        # NOTE /yolov5 is patched to create confusion matrix json files
        with open(weightfile[:-3] + '.json') as f:
            matrix = json.load(f)
            categories = yolov5_format.category_lookup_from_training(self.training)
            for category_name in list(matrix.keys()):
                matrix[categories[category_name]] = matrix.pop(category_name)
        return BasicModel(confusion_matrix=matrix, meta_information={'weightfile': weightfile})

    def on_model_published(self, basic_model: BasicModel) -> None:
        path = self.training.training_folder + '/result/weights/published'
        if not os.path.isdir(path):
            os.mkdir(path)
        target = f'{path}/latest.pt'
        weightfile = basic_model.meta_information['weightfile']

        shutil.move(weightfile, target)
        model_files.delete_json_for_weightfile(weightfile)

    def get_latest_model_files(self) -> Union[List[str], Dict[str, List[str]]]:
        path = self.training.training_folder + '/result/weights/published'
        weightfile = f'{path}/latest.pt'
        shutil.copy(weightfile, '/tmp/model.pt')
        training_path = '/'.join(weightfile.split('/')[:-4])

        return {
            self.model_format: ['/tmp/model.pt', f'{training_path}/result/opt.yaml']
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

        cmd = f'python /yolov5/classify/predict.py --weights {model_folder}/model.pt --source {images_folder} --img-size {img_size} --save-txt'
        executor.start(cmd)
        while executor.is_process_running():
            await sleep(1)

        if executor.return_code == 1:
            logging.error(f'Error during detecting: {executor.get_log()}')
            raise Exception('Error during detecting')

        detections = []
        logging.info('start parsing detections')
        labels_path = '/yolov5/runs/predict-cls/exp/labels'
        detections = await asyncio.get_event_loop().run_in_executor(None, self._parse, labels_path, model_information)

        return detections

    def _parse(self, labels_path, model_information):
        detections = []
        if os.path.exists(labels_path):
            for filename in os.scandir(labels_path):
                logging.error(filename)
                uuid = os.path.splitext(os.path.basename(filename.path))[0]
                classification_detections = self._parse_file(model_information, filename)
                detections.append({'image_id': uuid, 'classification_detections': classification_detections})
        return detections

    def _parse_file(self, model_information, filename) -> List[ClassificationDetection]:
        with open(filename.path, 'r') as f:
            content = f.readlines()
        classification_detections = []

        logging.error(content)
        for line in content:
            logging.error(line)
            probability, c = line.split(' ')
            probability = float(probability) * 100
            c = c.strip()
            logging.error(model_information.categories)
            category = [category for category in model_information.categories if category.name == c][0]
            classification_detection = ClassificationDetection(
                category_name=category.name, model_name=model_information.version, confidence=probability, category_id=category.id)

            classification_detections.append(classification_detection)
        return classification_detections

    async def clear_training_data(self, training_folder: str) -> None:
        # Note: Keep best.pt in case uploaded model was not best.
        keep_files = ['last_training.log', 'last.pt']
        keep_dirs = ['result', 'weights']
        for root, dirs, files in os.walk(training_folder, topdown=False):
            for file in files:
                if file not in keep_files:
                    os.remove(os.path.join(root, file))
            for dir in dirs:
                if dir not in keep_dirs:
                    shutil.rmtree(os.path.join(root, dir))

    @property
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        return [
            PretrainedModel(name='s-cls', label='YOLO v5 small classification', description='~5 fps on Jetson Nano'),
            # PretrainedModel(name='yolov5m', label='YOLO v5 medium', description='~2 fps on Jetson Nano'),
        ]

    @property
    def model_architecture(self):
        return 'yolov5_cls'

    @property
    def progress(self) -> float:
        return self.get_progress_from_log()

    def get_progress_from_log(self) -> float:
        if self.epochs == 1:
            return 1.0  # NOTE: We would divide by 0 in this case
        lines = list(reversed(self.executor.get_log_by_lines()))
        for line in lines:
            if re.search(f'/{self.epochs}', line):
                found_line = line.split('/')
                if found_line:
                    epoch = int(found_line[0])
                    progress = int(epoch) / int(self.epochs)
                    return progress

    @staticmethod
    def infer_image(model_folder: str, image_path: str):
        '''
            Run this function from within the docker container.
            Example Usage
                python -c 'from yolov5_trainer import Yolov5Trainer; Yolov5Trainer.infer_image("/data/some_folder_with_model.pt_and_model.json","/data/img.jpg")

        '''

        trainer = Yolov5Trainer()
        model_information = ModelInformation.load_from_disk(model_folder)
        import asyncio

        detections = asyncio.get_event_loop().run_until_complete(
            trainer._detect(model_information, [image_path], model_folder))

        from fastapi.encoders import jsonable_encoder
        print(jsonable_encoder(detections))
