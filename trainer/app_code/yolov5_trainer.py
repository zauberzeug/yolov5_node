import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
from asyncio import sleep
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
from fastapi.encoders import jsonable_encoder
from learning_loop_node.data_classes import (BasicModel, BoxDetection,
                                             CategoryType,
                                             ClassificationDetection,
                                             Detections, Hyperparameter,
                                             ModelInformation, PointDetection,
                                             PretrainedModel)
from learning_loop_node.trainer.executor import Executor
from learning_loop_node.trainer.trainer_logic import TrainerLogic

from . import batch_size_calculation, model_files, yolov5_format


class Yolov5TrainerLogic(TrainerLogic):

    def __init__(self) -> None:
        self.is_cla = os.getenv('YOLOV5_MODE') == 'CLASSIFICATION'
        if not self.is_cla:
            assert os.getenv('YOLOV5_MODE') == 'DETECTION', 'YOLOV5_MODE should be `DETECTION` or `CLASSIFICATION`'
        super().__init__(model_format='yolov5_pytorch' if not self.is_cla else 'yolov5_cla_pytorch')

        logging.info(f'------ STARTING YOLOV5 TRAINER LOGIC WITH MODE {os.getenv("YOLOV5_MODE")} ------')
        self.latest_epoch = 0
        self.epochs = 1000 if self.is_cla else 2000
        self.patience = 300

    async def start_training_from_scratch(self, identifier: str) -> None:
        if self.is_cla:
            await self.start_training(model=f'yolov5_cla{identifier}.pt')
        else:
            await self.start_training(model=f'yolov5{identifier}.pt')

    @property
    def hyperparameter(self) -> Hyperparameter:
        assert self.is_initialized, 'Trainer is not initialized'
        assert self.training.data is not None, 'Training should have data'
        assert self.training.data.hyperparameter is not None, 'Training.data should have hyperparameter'
        return self.training.data.hyperparameter

    @property
    def training_folder(self) -> Path:
        assert self.training is not None
        assert self.training.training_folder is not None
        return Path(self.training.training_folder)

    async def start_training(self, model: str = 'model.pt') -> None:
        if self.is_cla:
            yolov5_format.create_file_structure_cla(self.training)
            if model == 'model.pt':
                model = f'{self.training.training_folder}/model.pt'
            # TODO check why hyps are not updated
            await self._start(model)
        else:
            yolov5_format.create_file_structure(self.training)

            hyperparameter_path = f'{self.training.training_folder}/hyp.yaml'
            if not os.path.isfile(hyperparameter_path):
                shutil.copy('/app/hyp.yaml', hyperparameter_path)

            yolov5_format.update_hyp(hyperparameter_path, self.hyperparameter)
            await self._start(model, " --clear")

    async def _start(self, model: str, additional_parameters: str = ''):
        resolution = self.hyperparameter.resolution

        if self.is_cla:
            batch_size = 4  # TODO check why batchsize is updated here
            cmd = f'WANDB_MODE=disabled python /app/app_code/yolov5/classify/train.py --exist-ok --batch-size {batch_size} --img {resolution} --data {self.training.training_folder} --model {model} --project {self.training.training_folder} --name result --epochs {self.epochs} --optimizer SGD {additional_parameters}'
        else:
            hyperparameter_path = f'{self.training.training_folder}/hyp.yaml'
            self.try_replace_optimized_hyperparameter()
            batch_size = await batch_size_calculation.calc(self.training.training_folder, model, hyperparameter_path, f'{self.training.training_folder}/dataset.yaml', resolution)
            cmd = f'WANDB_MODE=disabled python /app/app_code/yolov5/train.py --exist-ok --patience {self.patience} --batch-size {batch_size} --img {resolution} --data dataset.yaml --weights {model} --project {self.training.training_folder} --name result --hyp {hyperparameter_path} --epochs {self.epochs} {additional_parameters}'
            with open(hyperparameter_path) as f:
                logging.info(f'running training with command :\n {cmd} \nand hyperparameter\n{f.read()}')
        logging.info(f'running training with command :\n {cmd}')
        self.executor.start(cmd)

    def can_resume(self) -> bool:
        path = self.training_folder / 'result/weights/published/latest.pt'
        return path.exists()

    async def resume(self) -> None:
        logging.info('resume called')
        await self._start(str(self.training_folder / 'result/weights/published/latest.pt'))

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
        print(f'searching for new model in {self.training_folder}')
        if self.is_cla:
            weightfile = model_files.get_best(self.training_folder)
        else:
            weightfile = model_files.get_new(self.training_folder)
        if not weightfile:
            return None
        weightfile = str(weightfile.absolute())
        # NOTE /yolov5 is patched to create confusion matrix json files
        with open(str(weightfile)[:-3] + '.json') as f:
            matrix = json.load(f)
            categories = yolov5_format.category_lookup_from_training(self.training)
            for category_name in list(matrix.keys()):
                matrix[categories[category_name]] = matrix.pop(category_name)

        return BasicModel(confusion_matrix=matrix, meta_information={'weightfile': weightfile})

    def on_model_published(self, basic_model: BasicModel) -> None:
        path = (self.training_folder / 'result/weights/published').absolute()
        path.mkdir(parents=True, exist_ok=True)

        assert basic_model.meta_information is not None, 'meta_information should not be set'
        weightfile = basic_model.meta_information['weightfile']

        target = path / 'latest.pt'
        shutil.move(weightfile, target)
        model_files.delete_json_for_weightfile(Path(weightfile))
        # TODO why are the older epochs not deleted for cla model? .. ignored atm
        model_files.delete_older_epochs(Path(self.training.training_folder), Path(weightfile))

    def get_latest_model_files(self) -> Union[List[str], Dict[str, List[str]]]:
        path = (self.training_folder / 'result/weights/published').absolute()
        weightfile = f'{path}/latest.pt'
        if not os.path.isfile(weightfile):
            raise Exception(f'No model found at {weightfile}')
        shutil.copy(weightfile, '/tmp/model.pt')
        training_path = '/'.join(weightfile.split('/')[:-4])

        if self.is_cla:
            return {self.model_format: ['/tmp/model.pt', f'{training_path}/result/opt.yaml']}
        else:
            subprocess.run(
                f'python3 /app/app_code/yolov5/gen_wts.py -w {weightfile} -o /tmp/model.wts', shell=True, check=False)
            return {self.model_format: ['/tmp/model.pt', f'{training_path}/hyp.yaml'],
                    'yolov5_wts': ['/tmp/model.wts']}

    async def _detect(self, model_information: ModelInformation, images:  List[str], model_folder: str) -> List[Detections]:
        images_folder = '/tmp/imagelinks_for_detecting'
        shutil.rmtree(images_folder, ignore_errors=True)
        os.makedirs(images_folder)
        for img in images:
            image_name = os.path.basename(img)
            os.symlink(img, f'{images_folder}/{image_name}')

        logging.info('start detections')
        shutil.rmtree('/app/app_code/yolov5/runs', ignore_errors=True)
        os.makedirs('/app/app_code/yolov5/runs')
        executor = Executor(images_folder)
        img_size = model_information.resolution

        if self.is_cla:
            cmd = f'python /app/app_code/yolov5/classify/predict.py --weights {model_folder}/model.pt --source {images_folder} --img-size {img_size} --save-txt'
        else:
            cmd = f'python /app/app_code/yolov5/detect.py --weights {model_folder}/model.pt --source {images_folder} --img-size {img_size} --conf-thres 0.2 --save-txt --save-conf'
        executor.start(cmd)
        while executor.is_process_running():
            await sleep(1)

        if executor.return_code == 1:
            logging.error(f'Error during detecting: {executor.get_log()}')
            raise Exception('Error during detecting')

        detections = []
        logging.info('start parsing detections')
        labels_path = '/app/app_code/yolov5/runs/predict-cls/exp/labels' if self.is_cla else '/yolov5/runs/detect/exp/labels'
        detections = await asyncio.get_event_loop().run_in_executor(None, self._parse, labels_path, images_folder, model_information)

        return detections

    def _parse(self, labels_path: str, images_folder: str, model_information: ModelInformation) -> List[Detections]:
        detections = []
        if os.path.exists(labels_path):
            for filename in os.scandir(labels_path):
                uuid = os.path.splitext(os.path.basename(filename.path))[0]
                if self.is_cla:
                    classification_detections = self._parse_file_cla(model_information, filename)
                    detections.append({'image_id': uuid, 'classification_detections': classification_detections})
                else:
                    box_detections, point_detections = self._parse_file(model_information, images_folder, filename.path)
                    detections.append(Detections(box_detections=box_detections,
                                      point_detections=point_detections, image_id=uuid))
        return detections

    @staticmethod
    def _parse_file_cla(model_info, filename) -> List[ClassificationDetection]:
        with open(filename.path, 'r') as f:
            content = f.readlines()
        classification_detections = []

        for line in content:
            probability, c = line.split(' ', maxsplit=1)
            probability = float(probability) * 100
            c = c.strip()
            category = [category for category in model_info.categories if category.name == c]
            if category:
                category = category[0]
                classification_detection = ClassificationDetection(
                    category_name=category.name, model_name=model_info.version, confidence=probability,
                    category_id=category.id)

                classification_detections.append(classification_detection)
        return classification_detections

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
        if self.is_cla:
            keep_files = ['last_training.log', 'last.pt']
        else:
            keep_files = ['last_training.log', 'hyp.yaml', 'dataset.yaml', 'best.pt']
        keep_dirs = ['result', 'weights']
        for root, dirs, files in os.walk(training_folder, topdown=False):
            for file in files:
                if file not in keep_files:
                    os.remove(os.path.join(root, file))
            for dir_ in dirs:
                if dir_ not in keep_dirs:
                    shutil.rmtree(os.path.join(root, dir_))

    @property
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        if self.is_cla:
            pm = [
                PretrainedModel(name='s-cls', label='YOLO v5 small classification',
                                description='~5 fps on Jetson Nano'),
                PretrainedModel(name='x-cls', label='YOLO v5 small classification',
                                description='~5 fps on Jetson Nano'),
                # PretrainedModel(name='yolov5m', label='YOLO v5 medium', description='~2 fps on Jetson Nano'),
            ]
        else:
            pm = [
                PretrainedModel(name='s6', label='YOLO v5 small', description='~5 fps on Jetson Nano'),
                # PretrainedModel(name='yolov5m', label='YOLO v5 medium', description='~2 fps on Jetson Nano'),
            ]
        return pm

    @property
    def model_architecture(self):
        return 'yolov5_cls' if self.is_cla else 'yolov5'

    @property
    def progress(self) -> float:
        if self.is_cla:
            return self.get_progress_from_log_cla()
        return self.get_progress_from_log()

    def get_progress_from_log_cla(self) -> float:
        if self.epochs == 0 or not self.is_initialized or self._executor is None:
            return 0.0
        lines = list(reversed(self.executor.get_log_by_lines()))
        for line in lines:
            if re.search(f'/{self.epochs}', line):
                found_line = line.split('/')
                if found_line:
                    return float(found_line[0]) / float(self.epochs)
        return 0.0

    def get_progress_from_log(self) -> float:
        if self.epochs == 1:
            return 0.0  # NOTE: We would divide by 0 in this case
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
