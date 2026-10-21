import asyncio
import json
import logging
import math
import os
import re
import shlex
import shutil
from dataclasses import asdict
from pathlib import Path

import cv2  # type: ignore # pylint: disable=import-error
from fastapi.encoders import jsonable_encoder
from learning_loop_node.data_classes import (
    Detections,
    Hyperparameter,
    ModelInformation,
    PretrainedModel,
    TrainingStateData,
)
from learning_loop_node.detector.postprocess import Prediction, to_detections
from learning_loop_node.enums import CategoryType
from learning_loop_node.trainer import trainer_logic
from learning_loop_node.trainer.batch_size import requested_batch_size
from learning_loop_node.trainer.exceptions import CriticalError, NodeNeedsRestartError
from learning_loop_node.trainer.executor import Executor
from learning_loop_node.trainer.hyperparameters import parse_pairs

from . import batch_size_calculation, model_files, training_run, yolov5_format
from .hyperparameters import HYPERPARAMETERS, TEMPLATE_PATH


class Yolov5TrainerLogic(trainer_logic.TrainerLogic):

    def __init__(self, vram_limit_gb: float = 0) -> None:
        super().__init__(model_format='yolov5_pytorch')

        self._vram_limit_gb = vram_limit_gb

        logging.info('------ STARTING YOLOV5 TRAINER LOGIC ------')
        self.latest_epoch = 0
        self.inference_batch_size = 100  # yolo processes images one by one

    # ---------------------------------------- IMPLEMENTED ABSTRACT PROPERTIES ----------------------------------------

    @property
    def training_progress(self) -> float | None:
        if self._executor is None:
            return None
        return self._get_progress_from_log()

    @property
    def model_architecture(self) -> str:
        return 'yolov5'

    @property
    def provided_pretrained_models(self) -> list[PretrainedModel]:
        return [PretrainedModel(name='n', label='YOLO v5 640 nano', description=''),
                PretrainedModel(name='s', label='YOLO v5 640 small', description=''),
                PretrainedModel(name='m', label='YOLO v5 640 medium', description=''),
                PretrainedModel(name='l', label='YOLO v5 640 large', description=''),
                PretrainedModel(name='x', label='YOLO v5 640 extra large', description=''),
                PretrainedModel(name='n6', label='YOLO v5 1280 nano', description=''),
                PretrainedModel(name='s6', label='YOLO v5 1280 small', description=''),
                PretrainedModel(name='m6', label='YOLO v5 1280 medium', description=''),
                PretrainedModel(name='l6', label='YOLO v5 1280 large', description=''),
                PretrainedModel(name='x6', label='YOLO v5 1280 extra large', description='')]

    @property
    def provided_hyperparameters(self) -> list[Hyperparameter]:
        return HYPERPARAMETERS

    @property
    def hyperparameter_path(self) -> str:
        return f'{self.training.training_folder}/hyp.yaml'

    # ---------------------------------------- IMPLEMENTED ABSTRACT METHODS ----------------------------------------

    async def _start_training_from_base_model(self) -> None:
        await self._start_training_from_model(f'{self.training.training_folder}/model.pt')

    async def _start_training_from_scratch(self) -> None:
        await self._start_training_from_model(f'yolov5{self.training.model_variant}.pt')

    def _can_resume(self) -> bool:
        return model_files.resume_checkpoint(self.training.training_folder_path) is not None

    async def _resume(self) -> None:
        checkpoint = model_files.resume_checkpoint(self.training.training_folder_path)
        assert checkpoint is not None
        self.training.hyperparameters['trainer_version'] = os.environ.get('NODE_VERSION') or 'unknown'
        self._training_run.clear_failure()
        await self.executor.start(f'python /app/train_det.py --resume {shlex.quote(str(checkpoint))} \
            --project {self.training.training_folder} --name result --vram-limit-gb {self._vram_limit_gb}',
                                  env={'WANDB_MODE': 'disabled'})

    def _get_new_best_training_state(self) -> TrainingStateData | None:
        weightfile = model_files.get_new(self.training.training_folder_path)
        if not weightfile:
            return None

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

        executor = Executor(self.training.training_folder, 'wts-converter.log')
        await executor.start('python /app/generate_wts.py -w /tmp/model.pt -o /tmp/model.wts')
        if await executor.wait() != 0:
            logging.error('Error during generating wts file: %s', executor.get_log())
            raise Exception('Error during generating wts file')

        return {self.model_format: ['/tmp/model.pt', f'{training_path}/hyp.yaml'], 'yolov5_wts': ['/tmp/model.wts']}

    async def _detect(self, model_information: ModelInformation, images: list[str],
                      model_folder: str) -> list[Detections]:
        hyperparameters = self.training.hyperparameters

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

        cmd = f'python /app/pred_det.py --weights {model_folder}/model.pt --source {images_folder}'
        cmd += f' --img-size {img_size} --conf-thres {hyperparameters["detect_nms_conf_thres"]}'
        cmd += f' --iou-thres {hyperparameters["detect_nms_iou_thres"]}'
        cmd += f' --vram-limit-gb {self._vram_limit_gb}'

        await executor.start(cmd)
        if await executor.wait() != 0:
            executor_log = executor.get_log()
            logging.error('Error during detecting: %s', executor_log)
            if 'CUDA out of memory' in executor_log or 'No CUDA GPUs are available' in executor_log:
                raise NodeNeedsRestartError()

            raise Exception('Error during detecting')

        logging.info('Start parsing detections')
        labels_path = '/app/app_code/yolov5/runs/detect/exp/labels'

        # NOTE: parse function is blocking by IO operations, so we need to run it in a separate thread (default executor is ThreadPoolExecutor)
        return await asyncio.get_event_loop().run_in_executor(None, self._parse, labels_path, images_folder, model_information)

    async def _clear_training_data(self, training_folder: str) -> None:
        keep_files = ['hyp.yaml', 'dataset.yaml', 'best.pt']
        keep_dirs = ['result', 'weights']
        for root, dirs, files in os.walk(training_folder, topdown=False):
            for file in files:
                if file not in keep_files and not file.endswith('.log'):
                    os.remove(os.path.join(root, file))
            for dir_ in dirs:
                if dir_ not in keep_dirs:
                    shutil.rmtree(os.path.join(root, dir_))

    # ---------------------------------------- OVERWRITTEN PROPERTIES AND METHODS ----------------------------------------

    @property
    def hyperparameters_for_state_sync(self) -> dict | None:
        """The hyperparameters, with the `batch_size` `train_det.py` measured once it has."""
        hyperparameters = super().hyperparameters_for_state_sync
        if hyperparameters is None or (batch_size := self._training_run.batch_size) is None:
            return hyperparameters
        return {**hyperparameters, 'batch_size': batch_size}

    def _get_executor_error_from_log(self) -> str | None:
        """Also raise the failures `train_det.py` recorded, which no retry can fix.

        :raises InvalidResolutionError: If the resolution does not suit the model's largest stride.
        :raises CriticalError: If not even the smallest batch fits on the graphics card.
        :raises NoGpuError: If there is no graphics card to train on.
        """
        self._training_run.raise_failure()
        return super()._get_executor_error_from_log()

    # ---------------------------------------- ADDITIONAL METHODS ----------------------------------------

    async def _start_training_from_model(self, model: str) -> None:
        yolov5_format.create_file_structure(self.training)
        additional_params = ' --clear'

        assert TEMPLATE_PATH.exists(), f'Hyperparameter file not found at "{TEMPLATE_PATH}"'
        shutil.copy(TEMPLATE_PATH, self.hyperparameter_path)
        yolov5_format.set_hyperparameters_in_file(self.hyperparameter_path, self.hyperparameters)

        await self._start(model, additional_params)

    async def _start(self, model: str, additional_parameters: str = ''):
        hyperparameters = self.training.hyperparameters
        batch_size_calculation.check_resolution(hyperparameters['resolution'], batch_size_calculation.MIN_STRIDE)
        point_sizes_by_uuid = _parse_point_sizes(hyperparameters['point_sizes_by_id'])

        trainer_version = os.environ.get('NODE_VERSION') or 'unknown'
        logging.info('Training with trainer version %s', trainer_version)
        hyperparameters['trainer_version'] = trainer_version

        p_sizes_by_id = ""
        for i, category in enumerate(self.training.categories):
            if category.type == CategoryType.Point:
                p_sizes_by_id += f"{i}:{point_sizes_by_uuid.get(category.id, 0.03)},"

        flip_label_pairs = ""
        for uuid_i, uuid_j in parse_pairs(hyperparameters['flip_label_pairs']):
            id_i = None
            id_j = None
            for i, category in enumerate(self.training.categories):
                if category.id == uuid_i:
                    id_i = i
                if category.id == uuid_j:
                    id_j = i
            if id_i is not None and id_j is not None:
                flip_label_pairs += f"{id_i}:{id_j},"

        cmd = f'python /app/train_det.py --exist-ok --patience {hyperparameters["patience"]} \
            --batch-size -1 --max-batch-size {requested_batch_size(hyperparameters)} --img {hyperparameters["resolution"]} \
            --data dataset.yaml --weights {model} \
            --project {self.training.training_folder} --name result --hyp {self.hyperparameter_path} \
            --epochs {hyperparameters["epochs"]} --conf-thres {hyperparameters["detect_nms_conf_thres"]} \
            --iou-thres {hyperparameters["detect_nms_iou_thres"]} \
            --vram-limit-gb {self._vram_limit_gb} \
            {additional_parameters}'
        if p_sizes_by_id:
            cmd += f' --point_sizes_by_id {p_sizes_by_id[:-1]}'
        if flip_label_pairs:
            cmd += f' --flip_label_pairs {flip_label_pairs[:-1]}'

        self._training_run.clear_failure()
        await self.executor.start(cmd, env={'WANDB_MODE': 'disabled'})

    @property
    def _training_run(self) -> training_run.TrainingRun:
        return training_run.TrainingRun(self.training.training_folder_path / 'result')

    def _parse(self, labels_path: str, images_folder: str, model_information: ModelInformation) -> list[Detections]:
        detections = []
        if os.path.exists(labels_path):
            for filename in os.scandir(labels_path):
                uuid = os.path.splitext(os.path.basename(filename.path))[0]
                predictions, img_height, img_width = self._parse_file(images_folder, filename.path)
                detections.append(to_detections(predictions, model_information, img_height, img_width,
                                                image_id=uuid))
        return detections

    def _get_progress_from_log(self) -> float:
        epochs = self.training.hyperparameters['epochs']
        if epochs == 1:
            return 0.0  # NOTE: We would divide by 0 in this case
        lines = list(reversed(self.executor.get_log_by_lines()))
        progress = 0.0
        for line in lines:
            if re.search(f'/{epochs - 1}', line):
                found_line = line.split(' ')
                for item in found_line:
                    if f'/{epochs - 1}' in item:
                        epoch, total_epochs = item.split('/')[:2]
                        try:
                            progress = float(epoch) / float(total_epochs)
                        except ValueError:
                            progress = -1.0
                        return progress
        return progress

    # ---------------------------------------- HELPER METHODS ----------------------------------------

    @staticmethod
    def _parse_file(images_folder: str, filename: str) -> tuple[list[Prediction], int, int]:
        """Read one YOLOv5 label file, with the size of the image it belongs to.

        A label line is ``<class> <cx> <cy> <w> <h> <confidence>``, the four coordinates
        normalised to the image and anchored on the box centre. :class:`Prediction` is anchored
        on the top-left corner and measured in pixels, which is what :func:`to_detections` takes;
        resolving the category, clipping and building the loop's dataclasses happen there.

        :return: ``(predictions, img_height, img_width)``
        """
        uuid = os.path.splitext(os.path.basename(filename))[0]

        # TODO change to approach that does not require to read the image
        image_path = f'{images_folder}/{uuid}.jpg'
        img_height, img_width, _ = cv2.imread(image_path).shape
        with open(filename, 'r') as f:
            content = f.readlines()

        predictions = []
        for line in content:
            c, x_, y_, w_, h_, probability_str = line.split(' ')
            width = float(w_) * img_width
            height = float(h_) * img_height
            predictions.append(Prediction(
                x=float(x_) * img_width - 0.5 * width,
                y=float(y_) * img_height - 0.5 * height,
                width=width,
                height=height,
                category_index=int(c),
                confidence=float(probability_str) * 100,  # The loop interprets values <= 1 as fractions.
            ))
        return predictions, img_height, img_width

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


def _parse_point_sizes(text: str) -> dict[str, float]:
    """The point size per category uuid from `point_sizes_by_id`, whose format the node library has checked.

    :raises CriticalError: If a size is not a positive number.
    """
    sizes = {}
    for uuid, size in parse_pairs(text):
        try:
            sizes[uuid] = float(size)
        except ValueError:
            sizes[uuid] = math.nan
        if not math.isfinite(sizes[uuid]) or sizes[uuid] <= 0:
            raise CriticalError(f'Invalid hyperparameter point_sizes_by_id: "{size}" for {uuid} is not a positive number')
    return sizes
