import asyncio
import json
import logging
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path

import cv2
import yaml  # type: ignore
from fastapi.encoders import jsonable_encoder
from learning_loop_node.data_classes import (
    BoxDetection,
    ClassificationDetection,
    Detections,
    ModelInformation,
    PointDetection,
    PretrainedModel,
    TrainingStateData,
)
from learning_loop_node.enums import CategoryType
from learning_loop_node.trainer import trainer_logic
from learning_loop_node.trainer.exceptions import CriticalError, NodeNeedsRestartError
from learning_loop_node.trainer.executor import Executor

from . import batch_size_calculation, model_files, yolov5_format


class Yolov5TrainerLogic(trainer_logic.TrainerLogic):

    def __init__(self) -> None:
        self.is_cla = os.getenv('YOLOV5_MODE') == 'CLASSIFICATION'
        if not self.is_cla:
            assert os.getenv('YOLOV5_MODE') == 'DETECTION', 'YOLOV5_MODE should be `DETECTION` or `CLASSIFICATION`'
        super().__init__(model_format='yolov5_pytorch' if not self.is_cla else 'yolov5_cla_pytorch')

        logging.info('------ STARTING YOLOV5 TRAINER LOGIC WITH MODE %s ------', os.getenv('YOLOV5_MODE'))
        self.latest_epoch = 0
        self.patience = 300
        self.inference_batch_size = 100  # yolo processes images one by one

        # Following will be overwritten by hyp.yaml
        self.epochs = 0
        self.detect_nms_conf_thres = 0.2
        self.detect_nms_iou_thres = 0.45
        self.point_sizes_by_uuid: dict[str, float] = {}
        self.flip_label_uuid_pairs: list[tuple[str, str]] = []

        self.additional_hyperparameters_parsed = False

    # ---------------------------------------- IMPLEMENTED ABSTRACT PROPERTIES ----------------------------------------

    @property
    def training_progress(self) -> float | None:
        if self._executor is None:
            return None
        if self.is_cla:
            return self._get_progress_from_log_cla()
        return self._get_progress_from_log()

    @property
    def model_architecture(self) -> str:
        return 'yolov5_cls' if self.is_cla else 'yolov5'

    @property
    def provided_pretrained_models(self) -> list[PretrainedModel]:
        if self.is_cla:
            return [PretrainedModel(name='s-cls', label='YOLO v5 small', description='~5fps on Jetson Nano'),
                    PretrainedModel(name='x-cls', label='YOLO v5 large', description='~5fps on Jetson Nano'),]

        return [PretrainedModel(name='s6', label='YOLO v5 small', description='~5 fps on Jetson Nano'), ]

    @property
    def hyperparameter_path(self) -> str:
        return f'{self.training.training_folder}/hyp.yaml'

    # ---------------------------------------- IMPLEMENTED ABSTRACT METHODS ----------------------------------------

    async def _start_training_from_base_model(self) -> None:
        await self._start_training_from_model(f'{self.training.training_folder}/model.pt')

    async def _start_training_from_scratch(self) -> None:
        await self._start_training_from_model(f'yolov5{self.training.model_variant}.pt')

    def _can_resume(self) -> bool:
        path = self.training.training_folder_path / 'result/weights/published/latest.pt'
        return path.exists()

    async def _resume(self) -> None:
        await self._start(model=str(self.training.training_folder_path / 'result/weights/published/latest.pt'))

    def _get_executor_error_from_log(self) -> str | None:
        if self._executor is None:
            return None
        for line in self._executor.get_log_by_lines(tail=50):
            if 'CUDA out of memory' in line:
                return 'graphics card is out of memory'
            if 'CUDA error: invalid device ordinal' in line:
                return 'graphics card not found'
        return None

    def _get_new_best_training_state(self) -> TrainingStateData | None:
        if self.is_cla:
            weightfile = model_files.get_best(self.training.training_folder_path)
        else:
            weightfile = model_files.get_new(self.training.training_folder_path)
        if not weightfile:
            return None

        if self.is_cla:
            epoch = None
        else:
            epoch = model_files.epoch_from_weightfile(weightfile)

        weightfile_str = str(weightfile.absolute())
        logging.info('found new best model at %s', weightfile_str)

        with open(str(weightfile_str)[:-3] + '.json') as f:
            metrics = json.load(f)
            categories = yolov5_format.category_lookup_from_training(self.training)
            for category_name in list(metrics.keys()):
                metrics[categories[category_name]] = metrics.pop(category_name)

        return TrainingStateData(confusion_matrix=metrics, meta_information={'weightfile': weightfile_str}, epoch=epoch)

    def _on_metrics_published(self, training_state_data: TrainingStateData) -> None:
        pub_path = (self.training.training_folder_path / 'result/weights/published').absolute()
        pub_path.mkdir(parents=True, exist_ok=True)

        weightfile = training_state_data.meta_information['weightfile']
        shutil.move(weightfile, pub_path / 'latest.pt')

        model_files.delete_json_for_weightfile(Path(weightfile))
        model_files.delete_older_epochs(Path(self.training.training_folder), Path(weightfile))

    async def _get_latest_model_files(self) -> dict[str, list[str]]:
        weightfile = (self.training.training_folder_path / "result/weights/published/latest.pt").absolute()
        if not os.path.isfile(weightfile):
            logging.error('No model found at %s - Training failed!', weightfile)
            raise CriticalError(f'No model found at {weightfile}')

        shutil.copy(weightfile, '/tmp/model.pt')
        training_path = '/'.join(str(weightfile).split('/')[:-4])

        if self.is_cla:
            return {self.model_format: ['/tmp/model.pt', f'{training_path}/result/opt.yaml']}

        executor = Executor(self.training.training_folder, 'wts-converter.log')
        await executor.start('python /app/generate_wts.py -w /tmp/model.pt -o /tmp/model.wts')
        if await executor.wait() != 0:
            logging.error('Error during generating wts file: %s', executor.get_log())
            raise Exception('Error during generating wts file')

        return {self.model_format: ['/tmp/model.pt', f'{training_path}/hyp.yaml'], 'yolov5_wts': ['/tmp/model.wts']}

    async def _detect(self, model_information: ModelInformation, images: list[str],
                      model_folder: str) -> list[Detections]:

        self._save_additional_hyperparameters()

        images_folder = '/tmp/imagelinks_for_detecting'
        shutil.rmtree(images_folder, ignore_errors=True)
        os.makedirs(images_folder)

        for img in images:
            image_name = os.path.basename(img)
            os.symlink(img, f'{images_folder}/{image_name}')

        shutil.rmtree('/app/app_code/yolov5/runs', ignore_errors=True)
        os.makedirs('/app/app_code/yolov5/runs')

        executor = Executor(self.training.training_folder, 'detect.log')
        img_size = model_information.resolution

        if self.is_cla:
            cmd = f'python /app/pred_cla.py --weights {model_folder}/model.pt --source {images_folder} --img-size {img_size} --save-txt'
        else:
            cmd = f'python /app/pred_det.py --weights {model_folder}/model.pt --source {images_folder} --img-size {img_size} --conf-thres {self.detect_nms_conf_thres} --iou-thres {self.detect_nms_iou_thres}'

        await executor.start(cmd)
        if await executor.wait() != 0:
            executor_log = executor.get_log()
            logging.error('Error during detecting: %s', executor_log)
            if 'CUDA out of memory' in executor_log or 'No CUDA GPUs are available' in executor_log:
                raise NodeNeedsRestartError()

            raise Exception('Error during detecting')

        logging.info('Start parsing detections')
        labels_path = '/app/app_code/yolov5/runs/predict-cls/exp/labels' if self.is_cla else '/app/app_code/yolov5/runs/detect/exp/labels'

        # NOTE: parse function is blocking by IO operations, so we need to run it in a separate thread (default executor is ThreadPoolExecutor)
        return await asyncio.get_event_loop().run_in_executor(None, self._parse, labels_path, images_folder, model_information)

    async def _clear_training_data(self, training_folder: str) -> None:
        if self.is_cla:        # Note: Keep best.pt in case uploaded model was not best.
            keep_files = ['last.pt']
        else:
            keep_files = ['hyp.yaml', 'dataset.yaml', 'best.pt']
        keep_dirs = ['result', 'weights']
        for root, dirs, files in os.walk(training_folder, topdown=False):
            for file in files:
                if file not in keep_files and not file.endswith('.log'):
                    os.remove(os.path.join(root, file))
            for dir_ in dirs:
                if dir_ not in keep_dirs:
                    shutil.rmtree(os.path.join(root, dir_))

    # ---------------------------------------- ADDITIONAL METHODS ----------------------------------------

    async def _start_training_from_model(self, model: str) -> None:

        if self.is_cla:
            yolov5_format.create_file_structure_cla(self.training)
            if model == 'model.pt':
                model = f'{self.training.training_folder}/model.pt'
            additional_params = ''
        else:
            yolov5_format.create_file_structure(self.training)
            additional_params = ' --clear'

        base_hyp_path = Path(__file__).resolve().parents[1] / ('hyp_cla.yaml' if self.is_cla else 'hyp_det.yaml')
        assert (base_hyp_path).exists(), f'Hyperparameter file not found at "{base_hyp_path}"'
        shutil.copy(base_hyp_path, self.hyperparameter_path)
        yolov5_format.set_hyperparameters_in_file(self.hyperparameter_path, self.hyperparameters)

        await self._start(model, additional_params)

    async def _start(self, model: str, additional_parameters: str = ''):
        resolution = self.training.hyperparameters['resolution']

        self._save_additional_hyperparameters()

        if self.is_cla:
            cmd = f'python /app/train_cla.py --exist-ok --img {resolution} \
                --data {self.training.training_folder} --model {model} \
                --project {self.training.training_folder} --name result \
                --hyp {self.hyperparameter_path} --optimizer SGD {additional_parameters}'
        else:
            try:
                batch_size = await batch_size_calculation.calc(self.training.training_folder, model, self.hyperparameter_path,
                                                               f'{self.training.training_folder}/dataset.yaml', resolution)
            except Exception as e:
                logging.exception('Error during batch size calculation:')
                raise NodeNeedsRestartError() from e

            p_sizes_by_id = ""
            for i, category in enumerate(self.training.categories):
                if category.type == CategoryType.Point:
                    size = self.point_sizes_by_uuid.get(category.id, 0.03)
                    p_sizes_by_id += f"{i}:{size},"

            flip_label_pairs = ""
            for uuid_i, uuid_j in self.flip_label_uuid_pairs:
                id_i = None
                id_j = None
                for i, category in enumerate(self.training.categories):
                    if category.id == uuid_i:
                        id_i = i
                    if category.id == uuid_j:
                        id_j = i
                if id_i is not None and id_j is not None:
                    flip_label_pairs += f"{id_i}:{id_j},"

            cmd = f'python /app/train_det.py --exist-ok --patience {self.patience} \
                --batch-size {batch_size} --img {resolution} --data dataset.yaml --weights {model} \
                --project {self.training.training_folder} --name result --hyp {self.hyperparameter_path} \
                --epochs {self.epochs} {additional_parameters}'
            if p_sizes_by_id:
                cmd += f' --point_sizes_by_id {p_sizes_by_id[:-1]}'
            if flip_label_pairs:
                cmd += f' --flip_label_pairs {flip_label_pairs[:-1]}'

        await self.executor.start(cmd, env={'WANDB_MODE': 'disabled'})

    def _save_additional_hyperparameters(self) -> None:
        """Save additional hyperparameters to attributes of self.
        These parameters are not passed to the yolov5 trainer, but are used to modify the training (and inference) process.
        """
        if self.additional_hyperparameters_parsed:
            return
        self.additional_hyperparameters_parsed = True

        if not os.path.exists(self.hyperparameter_path):
            logging.warning('No hyperparameter file found at %s', self.hyperparameter_path)
            raise CriticalError(f'No hyperparameter file found at {self.hyperparameter_path}')

        with open(self.hyperparameter_path, errors='ignore') as f:
            hyp = dict(yaml.safe_load(f))  # load hyps dict

        self.epochs = int(hyp.get('epochs', self.epochs))
        self.detect_nms_conf_thres = float(hyp.get('detect_nms_conf_thres', self.detect_nms_conf_thres))
        self.detect_nms_iou_thres = float(hyp.get('detect_nms_iou_thres', self.detect_nms_iou_thres))

        if point_sizes_by_id_str := str(hyp.get('point_sizes_by_id', '')):
            for item in point_sizes_by_id_str.split(','):
                k, v = item.split(':')
                self.point_sizes_by_uuid[str(k)] = float(v)

        if flip_label_pairs_str := str(hyp.get('flip_label_pairs', '')):
            for item in flip_label_pairs_str.split(','):
                k, v = item.split(':')
                self.flip_label_uuid_pairs.append((str(k), str(v)))

        hyp_str = ', '.join(f'{k}={v}' for k, v in hyp.items())
        logging.info('parsed hyperparameters %s: epochs: %d, detect_nms_conf_thres: %f, detect_nms_iou_thres: %f',
                     hyp_str, self.epochs, self.detect_nms_conf_thres, self.detect_nms_iou_thres)
        logging.info('point_sizes_by_id: %s', self.point_sizes_by_uuid)
        logging.info('flip_label_pairs: %s', self.flip_label_uuid_pairs)

    def _parse(self, labels_path: str, images_folder: str, model_information: ModelInformation) -> list[Detections]:
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

    def _get_progress_from_log_cla(self) -> float:
        if self.epochs == 0:
            return 0.0
        lines = list(reversed(self.executor.get_log_by_lines()))
        for line in lines:
            if re.search(f'/{self.epochs}', line):
                found_line = line.split('/')
                if found_line:
                    return float(found_line[0]) / float(self.epochs)
        return 0.0

    def _get_progress_from_log(self) -> float:
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
    def _parse_file_cla(model_info: ModelInformation, filepath: str) -> list[ClassificationDetection]:
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
    def clip_box(x: float, y: float, width: float, height: float, img_width: int, img_height: int
                 ) -> tuple[float, float, float, float]:
        '''make sure the box is within the image
            x,y is the center of the box
        '''
        left = max(0, x - 0.5 * width)
        top = max(0, y - 0.5 * height)
        right = min(img_width, x + 0.5 * width)
        bottom = min(img_height, y + 0.5 * height)

        x = 0.5 * (left + right)
        y = 0.5 * (top + bottom)
        width = right - left
        height = bottom - top

        return x, y, width, height

    @staticmethod
    def clip_point(x: float, y: float, img_width: int, img_height: int) -> tuple[float, float]:
        x = min(max(0, x), img_width)
        y = min(max(0, y), img_height)
        return x, y

    @staticmethod
    def _parse_file(model_info: ModelInformation, images_folder: str, filename: str
                    ) -> tuple[list[BoxDetection], list[PointDetection]]:
        uuid = os.path.splitext(os.path.basename(filename))[0]

        # TODO change to approach that does not require to read the image
        image_path = f'{images_folder}/{uuid}.jpg'
        img_height, img_width, _ = cv2.imread(image_path).shape
        with open(filename, 'r') as f:
            content = f.readlines()
        box_detections = []
        point_detections = []

        for line in content:
            c, x_, y_, w_, h_, probability_str = line.split(' ')

            category = model_info.categories[int(c)]
            x = float(x_) * img_width
            y = float(y_) * img_height
            width = float(w_) * img_width
            height = float(h_) * img_height
            probability = float(probability_str) * 100

            if category.type == CategoryType.Box:
                x, y, width, height = Yolov5TrainerLogic.clip_box(x, y, width, height, img_width, img_height)
                box_detections.append(
                    BoxDetection(category_name=category.name, x=int(x - 0.5 * width),
                                 y=int(y - 0.5 * height),
                                 width=int(width),
                                 height=int(height),
                                 model_name=model_info.version, confidence=probability, category_id=category.id))
            elif category.type == CategoryType.Point:
                x, y = Yolov5TrainerLogic.clip_point(x, y, img_width, img_height)
                point_detections.append(
                    PointDetection(category_name=category.name, x=x, y=y, model_name=model_info.version,
                                   confidence=probability, category_id=category.id))
        return box_detections, point_detections

    @staticmethod
    def infer_image(model_folder: str, image_path: str) -> None:
        '''Run this function from within the docker container. Example Usage:
            python -c 'from yolov5_trainer import Yolov5Trainer; Yolov5Trainer.infer_image("/data/some_folder_with_model.pt_and_model.json","/data/img.jpg")
        '''
        trainer_logic_ = Yolov5TrainerLogic()
        model_information = ModelInformation.load_from_disk(model_folder)
        assert model_information is not None, 'model_information should not be None'

        detections = asyncio.get_event_loop().run_until_complete(
            trainer_logic_._detect(model_information, [image_path], model_folder))  # pylint: disable=protected-access

        for detection in detections:
            print(jsonable_encoder(asdict(detection)))
