import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from asyncio import sleep
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple, Union

import cv2
from fastapi.encoders import jsonable_encoder
from learning_loop_node.data_classes import (BasicModel, BoxDetection,
                                             CategoryType, Detections,
                                             Hyperparameter, ModelInformation,
                                             PointDetection, PretrainedModel)
from learning_loop_node.trainer.executor import Executor
from learning_loop_node.trainer.trainer_logic import TrainerLogic

from . import batch_size_calculation, model_files, yolov5_format


class Yolov5TrainerLogic(TrainerLogic):

    def __init__(self) -> None:
        super().__init__(model_format='yolov5_pytorch')
        self.latest_epoch = 0
        self.epochs = 2000
        self.patience = 300

    async def start_training_from_scratch(self, identifier: str) -> None:
        await self.start_training(model=f'yolov5{identifier}.pt')

    @property
    def hyperparameter(self) -> Hyperparameter:
        assert self.is_initialized, 'Trainer is not initialized'
        assert self.training.data is not None, 'Training should have data'
        assert self.training.data.hyperparameter is not None, 'Training.data should have hyperparameter'
        return self.training.data.hyperparameter

    async def start_training(self, model: str = 'model.pt') -> None:
        yolov5_format.create_file_structure(self.training)

        hyperparameter_path = f'{self.training.training_folder}/hyp.yaml'
        if not os.path.isfile(hyperparameter_path):
            shutil.copy('/app/hyp.yaml', hyperparameter_path)

        yolov5_format.update_hyp(hyperparameter_path, self.hyperparameter)
        await self._start(model, " --clear")

    async def _start(self, model: str, additional_parameters: str = ''):
        resolution = self.hyperparameter.resolution

        hyperparameter_path = f'{self.training.training_folder}/hyp.yaml'
        self.try_replace_optimized_hyperparameter()
        batch_size = await batch_size_calculation.calc(self.training.training_folder, model, hyperparameter_path, f'{self.training.training_folder}/dataset.yaml', resolution)

        cmd = f'WANDB_MODE=disabled python /app/yolov5/train.py --exist-ok --patience {self.patience} --batch-size {batch_size} --img {resolution} --data dataset.yaml --weights {model} --project {self.training.training_folder} --name result --hyp {hyperparameter_path} --epochs {self.epochs} {additional_parameters}'

        with open(hyperparameter_path) as f:
            logging.info(f'running training with command :\n {cmd} \nand hyperparameter\n{f.read()}')
        self.executor.start(cmd)

    def can_resume(self) -> bool:
        path = f'{self.training.training_folder}/result/weights/published/latest.pt'
        return os.path.exists(path)

    async def resume(self) -> None:
        logging.info('resume called')
        await self._start(f'{self.training.training_folder}/result/weights/published/latest.pt')

    def try_replace_optimized_hyperparameter(self):
        optimized_hyp = f'{self.training.project_folder}/yolov5s6_evolved_hyperparameter.yaml'
        if os.path.exists(optimized_hyp):
            logging.info('Found optimized hyperparameter')
            shutil.copy(optimized_hyp,
                        f'{self.training.training_folder}/hyp.yaml')
        else:
            logging.warning('NO optimized hyperparameter found (!)')

    def get_executor_error_from_log(self) -> Optional[str]:
        if self._executor is None:
            return

        for line in self._executor.get_log_by_lines(since_last_start=True):
            if 'CUDA out of memory' in line:
                return 'graphics card is out of memory'
            if 'CUDA error: invalid device ordinal' in line:
                return 'graphics card not found'
        return None

    def get_new_model(self) -> Optional[BasicModel]:
        weightfile = model_files.get_new(self.training.training_folder)
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
        os.makedirs(path, exist_ok=True)

        assert basic_model.meta_information is not None, 'meta_information should not be set'
        weightfile = basic_model.meta_information['weightfile']

        shutil.move(weightfile, f'{path}/latest.pt')
        model_files.delete_json_for_weightfile(weightfile)
        model_files.delete_older_epochs(self.training.training_folder, weightfile)

    def get_latest_model_files(self) -> Union[List[str], Dict[str, List[str]]]:
        path = self.training.training_folder + '/result/weights/published'
        weightfile = f'{path}/latest.pt'
        shutil.copy(weightfile, '/tmp/model.pt')
        training_path = '/'.join(weightfile.split('/')[:-4])

        subprocess.run(f'python3 /yolov5/gen_wts.py -w {weightfile} -o /tmp/model.wts', shell=True, check=False)
        return {
            self.model_format: ['/tmp/model.pt', f'{training_path}/hyp.yaml'],
            'yolov5_wts': ['/tmp/model.wts']
        }

    async def _detect(self, model_information: ModelInformation, images:  List[str], model_folder: str) -> List[Detections]:
        images_folder = '/tmp/imagelinks_for_detecting'
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
            logging.error(f'Error during detecting: {executor.get_log()}')
            raise Exception('Error during detecting')

        detections = []
        logging.info('start parsing detections')
        labels_path = '/yolov5/runs/detect/exp/labels'
        detections = await asyncio.get_event_loop().run_in_executor(None, self._parse, labels_path, images_folder, model_information)

        return detections

    def _parse(self, labels_path: str, images_folder: str, model_information: ModelInformation) -> List[Detections]:
        detections = []
        if os.path.exists(labels_path):
            for filename in os.scandir(labels_path):
                uuid = os.path.splitext(os.path.basename(filename.path))[0]
                box_detections, point_detections = self._parse_file(model_information, images_folder, filename.path)
                detections.append(Detections(box_detections=box_detections,
                                  point_detections=point_detections, image_id=uuid))
        return detections

    def _parse_file(self, model_info: ModelInformation, images_folder: str, filename: str) -> Tuple[
            List[BoxDetection], List[PointDetection]]:
        uuid = os.path.splitext(os.path.basename(filename))[0]

        image_path = f'{images_folder}/{uuid}.jpg'
        img_height, img_width, _ = cv2.imread(image_path).shape
        with open(filename, 'r') as f:
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

            category = model_info.categories[c]
            width = w*img_width
            height = h*img_height
            x = (x*img_width)-0.5*width
            y = (y*img_height)-0.5*height
            if (category.type == CategoryType.Box):
                box_detection = BoxDetection(
                    category_name=category.name, x=x, y=y, width=width, height=height,
                    model_name=model_info.version, confidence=probability, category_id=category.id)
                box_detections.append(box_detection)
            elif (category.type == CategoryType.Point):
                point_detection = PointDetection(
                    category_name=category.name, x=x + width / 2, y=y + height / 2, model_name=model_info.version,
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
                    shutil.rmtree(os.path.join(root, dir))

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
        progress = 0.0
        for line in lines:
            if re.search(f'/{self.epochs -1}', line):
                found_line = line.split(' ')
                for item in found_line:
                    if f'/{self.epochs -1}' in item:
                        epoch_and_total_epochs = item.split('/')
                        epoch = epoch_and_total_epochs[0]
                        total_epochs = epoch_and_total_epochs[1]
                        progress = float(epoch) / float(total_epochs)
                        return progress
        return progress

    @staticmethod
    def infer_image(model_folder: str, image_path: str) -> None:
        '''
            Run this function from within the docker container.
            Example Usage
                python -c 'from yolov5_trainer import Yolov5Trainer; Yolov5Trainer.infer_image("/data/some_folder_with_model.pt_and_model.json","/data/img.jpg")

        '''

        trainer_logic = Yolov5TrainerLogic()
        model_information = ModelInformation.load_from_disk(model_folder)
        assert model_information is not None, 'model_information should not be None'

        detections = asyncio.get_event_loop().run_until_complete(
            trainer_logic._detect(model_information, [image_path], model_folder))  # pylint: disable=protected-access

        for detection in detections:
            print(jsonable_encoder(asdict(detection)))
