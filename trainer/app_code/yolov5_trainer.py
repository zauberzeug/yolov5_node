import asyncio
import json
import logging
import os
import re
import shutil
from asyncio import sleep
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import yaml  # type: ignore
from fastapi.encoders import jsonable_encoder
from learning_loop_node.data_classes import BasicModel  # type: ignore
from learning_loop_node.data_classes import (BoxDetection, CategoryType,
                                             ClassificationDetection,
                                             Detections, Hyperparameter,
                                             ModelInformation, PointDetection,
                                             PretrainedModel)
from learning_loop_node.trainer import trainer_logic  # type: ignore
from learning_loop_node.trainer.executor import Executor  # type: ignore

from . import batch_size_calculation, model_files, yolov5_format
from .yolov5 import gen_wts


class Yolov5TrainerLogic(trainer_logic.TrainerLogic):

    def __init__(self) -> None:
        self.is_cla = os.getenv('YOLOV5_MODE') == 'CLASSIFICATION'
        if not self.is_cla:
            assert os.getenv('YOLOV5_MODE') == 'DETECTION', 'YOLOV5_MODE should be `DETECTION` or `CLASSIFICATION`'
        super().__init__(model_format='yolov5_pytorch' if not self.is_cla else 'yolov5_cla_pytorch')

        logging.info(f'------ STARTING YOLOV5 TRAINER LOGIC WITH MODE {os.getenv("YOLOV5_MODE")} ------')
        self.latest_epoch = 0
        self.epochs = 0  # will be overwritten by hyp.yaml
        self.patience = 300

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

    # ---------------------------------------- IMPLEMENTED ABSTRACT METHODS ----------------------------------------

    @property
    def training_progress(self) -> Optional[float]:
        if self._executor is None:
            return None
        if self.is_cla:
            return self.get_progress_from_log_cla()
        return self.get_progress_from_log()

    @property
    def provided_pretrained_models(self) -> List[PretrainedModel]:
        if self.is_cla:
            pm = [PretrainedModel(name='s-cls', label='YOLO v5 small', description='~5fps on Jetson Nano'),
                  PretrainedModel(name='x-cls', label='YOLO v5 large', description='~5fps on Jetson Nano'),]
        else:
            pm = [PretrainedModel(name='s6', label='YOLO v5 small', description='~5 fps on Jetson Nano'), ]
        return pm

    @property
    def model_architecture(self):
        return 'yolov5_cls' if self.is_cla else 'yolov5'

    async def start_training(self, model: str = 'model.pt') -> None:
        app_root = Path(__file__).resolve().parents[1]

        if self.is_cla:
            yolov5_format.create_file_structure_cla(self.training)
            if model == 'model.pt':
                model = f'{self.training.training_folder}/model.pt'
            parameter_destination_path = f'{self.training.training_folder}/hyp.yaml'
            parameter_source_path = app_root / 'hyp_cla.yaml'
            assert (parameter_source_path).exists(), 'Hyperparameter file not found at "{hyperparameter_source_path}"'
            shutil.copy(parameter_source_path, parameter_destination_path)
            yolov5_format.update_hyp(parameter_destination_path, self.hyperparameter)
            await self._start(model)
        else:
            yolov5_format.create_file_structure(self.training)
            parameter_destination_path = f'{self.training.training_folder}/hyp.yaml'
            parameter_source_path = app_root / 'hyp_det.yaml'
            assert (parameter_source_path).exists(), 'Hyperparameter file not found at "{hyperparameter_source_path}"'
            shutil.copy(parameter_source_path, parameter_destination_path)
            yolov5_format.update_hyp(parameter_destination_path, self.hyperparameter)
            await self._start(model, " --clear")

    async def start_training_from_scratch(self, base_model_id: str) -> None:
        await self.start_training(model=f'yolov5{base_model_id}.pt')

    def can_resume(self) -> bool:
        path = self.training_folder / 'result/weights/published/latest.pt'
        return path.exists()

    async def resume(self) -> None:
        logging.info('resume called')
        await self._start(str(self.training_folder / 'result/weights/published/latest.pt'))

    def get_executor_error_from_log(self) -> Optional[str]:
        if self._executor is None:
            return None
        for line in self._executor.get_log_by_lines(since_last_start=True):
            if 'CUDA out of memory' in line:
                return 'graphics card is out of memory'
            if 'CUDA error: invalid device ordinal' in line:
                return 'graphics card not found'
        return None

    def get_new_model(self) -> Optional[BasicModel]:
        if self.is_cla:
            weightfile = model_files.get_best(self.training_folder)
        else:
            weightfile = model_files.get_new(self.training_folder)
        if not weightfile:
            return None
        weightfile_str = str(weightfile.absolute())
        logging.info(f'found new model at {weightfile_str}')
        # NOTE /yolov5 is patched to create confusion matrix json files
        with open(str(weightfile_str)[:-3] + '.json') as f:
            matrix = json.load(f)
            categories = yolov5_format.category_lookup_from_training(self.training)
            for category_name in list(matrix.keys()):
                matrix[categories[category_name]] = matrix.pop(category_name)

        return BasicModel(confusion_matrix=matrix, meta_information={'weightfile': weightfile_str})

    def on_model_published(self, basic_model: BasicModel) -> None:
        path = (self.training_folder / 'result/weights/published').absolute()
        path.mkdir(parents=True, exist_ok=True)

        assert basic_model.meta_information is not None, 'meta_information should be set'
        weightfile = basic_model.meta_information['weightfile']

        target = path / 'latest.pt'
        shutil.move(weightfile, target)
        model_files.delete_json_for_weightfile(Path(weightfile))
        # TODO why are the older epochs not deleted for cla model? .. ignored atm
        model_files.delete_older_epochs(Path(self.training.training_folder), Path(weightfile))

    def get_latest_model_files(self) -> Optional[Union[List[str], Dict[str, List[str]]]]:
        path = (self.training_folder / 'result/weights/published').absolute()
        weightfile = f'{path}/latest.pt'
        if not os.path.isfile(weightfile):
            logging.error(f'No model found at {weightfile}')
            return None
        shutil.copy(weightfile, '/tmp/model.pt')
        training_path = '/'.join(weightfile.split('/')[:-4])

        if self.is_cla:
            return {self.model_format: ['/tmp/model.pt', f'{training_path}/result/opt.yaml']}
        else:
            gen_wts(pt_file_path=weightfile, wts_file_path='/tmp/model.wts')
            return {self.model_format: ['/tmp/model.pt', f'{training_path}/hyp.yaml'],
                    'yolov5_wts': ['/tmp/model.wts']}

    async def _detect(self, model_information: ModelInformation, images:  List[str], model_folder: str) -> List[Detections]:
        images_folder = '/tmp/imagelinks_for_detecting'
        shutil.rmtree(images_folder, ignore_errors=True)
        os.makedirs(images_folder)
        for img in images:
            image_name = os.path.basename(img)
            os.symlink(img, f'{images_folder}/{image_name}')

        logging.info('start detecting')
        shutil.rmtree('/app/app_code/yolov5/runs', ignore_errors=True)
        os.makedirs('/app/app_code/yolov5/runs')
        executor = Executor(images_folder)
        img_size = model_information.resolution

        if self.is_cla:
            cmd = f'python /app/pred_cla.py --weights {model_folder}/model.pt --source {images_folder} --img-size {img_size} --save-txt'
        else:
            cmd = f'python /app/pred_det.py --weights {model_folder}/model.pt --source {images_folder} --img-size {img_size} --conf-thres 0.2 --save-txt --save-conf --nosave'
        logging.info(f'running detection with command :\n {cmd}')

        executor.start(cmd)
        while executor.is_process_running():
            await sleep(1)

        if executor.return_code == 1:
            logging.error(f'Error during detecting: \n {executor.get_log()}')
            raise Exception('Error during detecting')

        detections = []
        logging.info('start parsing detections')
        labels_path = '/app/app_code/yolov5/runs/predict-cls/exp/labels' if self.is_cla else '/app/app_code/yolov5/runs/detect/exp/labels'
        detections = await asyncio.get_event_loop().run_in_executor(None, self._parse, labels_path, images_folder, model_information)

        return detections

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

    # ---------------------------------------- ADDITIONAL METHODS ----------------------------------------
    def load_hyps_set_epochs(self, hyp_path: str) -> None:
        with open(hyp_path, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
        hyp = {k: float(v) for k, v in hyp.items()}
        hyp_str = 'hyps: ' + ', '.join(f'{k}={v}' for k, v in hyp.items())
        logging.info(hyp_str)
        self.epochs = int(hyp.get('epochs', self.epochs))

    async def _start(self, model: str, additional_parameters: str = ''):
        resolution = self.hyperparameter.resolution
        hyperparameter_path = f'{self.training.training_folder}/hyp.yaml'
        self.load_hyps_set_epochs(hyperparameter_path)

        if self.is_cla:
            cmd = f'python /app/train_cla.py --exist-ok --img {resolution} \
                --data {self.training.training_folder} --model {model} \
                --project {self.training.training_folder} --name result \
                --hyp {hyperparameter_path} --optimizer SGD {additional_parameters}'
        else:
            p_ids, p_sizes = yolov5_format.get_ids_and_sizes_of_point_classes(self.training)
            self.try_replace_optimized_hyperparameter()
            batch_size = await batch_size_calculation.calc(self.training.training_folder, model, hyperparameter_path,
                                                           f'{self.training.training_folder}/dataset.yaml', resolution)
            cmd = f'WANDB_MODE=disabled python /app/train_det.py --exist-ok --patience {self.patience} \
                --batch-size {batch_size} --img {resolution} --data dataset.yaml --weights {model} \
                --project {self.training.training_folder} --name result --hyp {hyperparameter_path} \
                --epochs {self.epochs} {additional_parameters}'
            if p_ids:
                cmd += f' --point_ids {",".join(p_ids)} --point_sizes {",".join(p_sizes)}'

            with open(hyperparameter_path) as f:
                logging.info(f'running training with command :\n {cmd} \nand hyperparameter\n{f.read()}')

        logging.info(f'running training with command :\n {cmd}')
        self.executor.start(cmd)

    def try_replace_optimized_hyperparameter(self):
        optimized_hyp = f'{self.training.project_folder}/yolov5s6_evolved_hyperparameter.yaml'
        if os.path.exists(optimized_hyp):
            logging.info('Found optimized hyperparameter')
            shutil.copy(optimized_hyp, f'{self.training.training_folder}/hyp.yaml')
        else:
            logging.warning('No optimized hyperparameter found (!)')

    def _parse(self, labels_path: str, images_folder: str, model_information: ModelInformation) -> List[Detections]:
        detections = []
        if os.path.exists(labels_path):
            for filename in os.scandir(labels_path):
                uuid = os.path.splitext(os.path.basename(filename.path))[0]
                if self.is_cla:
                    classification_detections = self._parse_file_cla(model_information, filename.path)
                    detections.append(Detections(classification_detections=classification_detections, image_id=uuid))
                else:
                    box_detections, point_detections = self._parse_file(model_information, images_folder, filename.path)
                    detections.append(Detections(box_detections=box_detections,
                                      point_detections=point_detections, image_id=uuid))
        return detections

    def get_progress_from_log_cla(self) -> float:
        if self.epochs == 0:
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
                        epoch, total_epochs = item.split('/')[:2]
                        try:
                            progress = float(epoch) / float(total_epochs)
                        except ValueError:
                            progress = -1.0
                        return progress
        return progress

    # ---------------------------------------- HELPER METHODS ----------------------------------------

    @staticmethod
    def _parse_file_cla(model_info: ModelInformation, filepath: str) -> List[ClassificationDetection]:
        with open(filepath, 'r') as f:
            content = f.readlines()
        classification_detections = []

        for line in content:
            probability_str, c = line.split(' ', maxsplit=1)
            c = c.strip()
            probability = float(probability_str) * 100
            if probability < 20:
                continue
            categories = [category for category in model_info.categories if category.name == c]
            if categories:
                category = categories[0]
                classification_detection = ClassificationDetection(
                    category_name=category.name, model_name=model_info.version, confidence=probability,
                    category_id=category.id)

                classification_detections.append(classification_detection)
        return classification_detections

    @staticmethod
    def _parse_file(model_info: ModelInformation, images_folder: str, filename: str) -> Tuple[
            List[BoxDetection], List[PointDetection]]:
        uuid = os.path.splitext(os.path.basename(filename))[0]  # TODO use pathlib

        image_path = f'{images_folder}/{uuid}.jpg'  # TODO change to approach that does not require to read the image
        img_height, img_width, _ = cv2.imread(image_path).shape
        with open(filename, 'r') as f:
            content = f.readlines()
        box_detections = []
        point_detections = []

        for line in content:
            c, x, y, w, h, probability_str = line.split(' ')

            category = model_info.categories[int(c)]
            x = float(x) * img_width
            y = float(y) * img_height
            width = float(w) * img_width
            height = float(h) * img_height
            probability = float(probability_str) * 100

            if (category.type == CategoryType.Box):
                box_detections.append(
                    BoxDetection(
                        category_name=category.name, x=int(x - 0.5 * width),
                        y=int(y - 0.5 * height),
                        width=int(width),
                        height=int(height),
                        model_name=model_info.version, confidence=probability, category_id=category.id))
            elif (category.type == CategoryType.Point):
                point_detections.append(
                    PointDetection(category_name=category.name, x=x, y=y, model_name=model_info.version,
                                   confidence=probability, category_id=category.id))
        return box_detections, point_detections

    @staticmethod
    def infer_image(model_folder: str, image_path: str) -> None:
        '''Run this function from within the docker container. Example Usage:
            python -c 'from yolov5_trainer import Yolov5Trainer; Yolov5Trainer.infer_image("/data/some_folder_with_model.pt_and_model.json","/data/img.jpg")
        '''
        trainer_logic = Yolov5TrainerLogic()
        model_information = ModelInformation.load_from_disk(model_folder)
        assert model_information is not None, 'model_information should not be None'

        detections = asyncio.get_event_loop().run_until_complete(
            trainer_logic._detect(model_information, [image_path], model_folder))  # pylint: disable=protected-access

        for detection in detections:
            print(jsonable_encoder(asdict(detection)))
