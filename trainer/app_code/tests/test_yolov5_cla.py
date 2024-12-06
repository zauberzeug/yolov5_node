import asyncio
import glob
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import pytest
from learning_loop_node.data_classes import (Category, Context,
                                             ModelInformation, Training)
from learning_loop_node.data_exchanger import DataExchanger
from learning_loop_node.enums import CategoryType, TrainerState
from learning_loop_node.helpers.misc import create_image_folder
from learning_loop_node.loop_communication import LoopCommunicator
from learning_loop_node.trainer.downloader import TrainingsDownloader
from learning_loop_node.trainer.executor import Executor

from .. import yolov5_format
from ..yolov5_trainer import Yolov5TrainerLogic

# pylint: disable=protected-access,unused-argument

project_configuration = {
    'project_name': 'pytest_yolo5cla', 'box_categories': 2, 'segmentation_categories': 2, 'point_categories': 2,
    'classification_categories': 2, 'inbox': 4, 'annotate': 0, 'review': 0, 'complete': 14, 'image_style': 'beautiful',
    'thumbs': False, 'trainings': 1}


@pytest.mark.environment(organization='zauberzeug', project='pytest_yolo5cla', mode='CLASSIFICATION')
@pytest.mark.generate_project(configuration=project_configuration)
class TestWithLoop:
    """This test environment sets up the environment vars and
        a test project in the loop which is used for testing."""

    async def test_cla_training_creates_model(
            self, use_training_dir, data_exchanger: DataExchanger, glc: LoopCommunicator):
        """Training should create a model file (best.pt)"""

        project_folder = os.getcwd()
        categories, image_data = await download_training_data(create_image_folder(project_folder), data_exchanger, glc)

        training = Training(id=str(uuid4()),
                            project_folder=project_folder,
                            training_folder=os.getcwd() + '/training', images_folder=os.getcwd() + '/images',
                            model_uuid_for_detecting='model.pt',
                            context=Context(project=os.environ['LOOP_PROJECT'],
                                            organization=os.environ['LOOP_ORGANIZATION']),
                            categories=categories, hyperparameters={}, model_variant='',
                            image_data=image_data,
                            training_state=TrainerState.Initialized.value,
                            training_number=1)

        yolov5_format.create_file_structure_cla(training)
        logging.info(training.training_folder)  # /tmp/test_training/
        logging.info(list(Path(f'{training.training_folder}/train/green classification/').iterdir()))
        executor = Executor(os.getcwd())
        # from https://github.com/WongKinYiu/yolor#training
        ROOT = Path(__file__).resolve().parents[2]
        hyp_path = ROOT / 'app_code' / 'tests' / 'test_data' / 'hyp_cla.yaml'
        cmd = f'python {ROOT/"train_cla.py"} --project training --name result --hyp {hyp_path} --img 416 --data {training.training_folder} --model yolov5s-cls.pt --epochs 1'
        await executor.start(cmd, env={'WANDB_MODE': 'disabled'})
        assert (await executor.wait()) == 0

        logging.info(executor.get_log())
        assert 'Training complete' in executor.get_log()
        assert 'best.pt' in executor.get_log()
        best = training.training_folder + '/result/weights/best.pt'
        assert os.path.isfile(best)

    @pytest.mark.asyncio()
    async def test_cla_parse_progress_from_log(
            self, use_training_dir, data_exchanger: DataExchanger, glc: LoopCommunicator):
        """The progress should be parsed from the log file"""
        trainer = Yolov5TrainerLogic()
        trainer.epochs = 3  # NOTE: must correspond to the value set in test_data/hyp_cla.yaml
        if os.path.isfile('/tmp/model.pt'):
            os.remove('/tmp/model.pt')

        project_folder = os.getcwd()
        categories, image_data = await download_training_data(create_image_folder(project_folder), data_exchanger, glc)

        trainer._training = Training(id=str(uuid4()),
                                     project_folder=project_folder,
                                     training_folder=os.getcwd() + '/training',
                                     images_folder=os.getcwd() + '/images',
                                     model_uuid_for_detecting='model.pt',
                                     context=Context(project='demo', organization='zauberzeug'),
                                     categories=categories, hyperparameters={}, model_variant='',
                                     image_data=image_data,
                                     training_state=TrainerState.Initialized.value,
                                     training_number=1)
        yolov5_format.create_file_structure_cla(trainer.training)

        await asyncio.sleep(1)

        trainer._executor = Executor(os.getcwd())
        ROOT = Path(__file__).resolve().parents[2]
        hyp_path = ROOT / 'app_code' / 'tests' / 'test_data' / 'hyp_cla.yaml'
        cmd = f'python {ROOT/"train_cla.py"} --project training --name result --hyp {hyp_path} --img 416 --data {trainer.training.training_folder} --model yolov5s-cls.pt --epochs {trainer.epochs}'
        await trainer.executor.start(cmd, env={'WANDB_MODE': 'disabled'})
        assert (await trainer.executor.wait()) == 0

        logging.info(trainer.executor.get_log())
        assert f'{trainer.epochs}/{trainer.epochs}' in trainer.executor.get_log()
        assert trainer.training_progress == 1.0

    async def test_cla_new_model_discovery(self, use_training_dir):
        """The trainer should find new models"""
        trainer = Yolov5TrainerLogic()
        trainer._training = Training(id='someid', context=Context(organization='o', project='p'),
                                     project_folder='./', images_folder='./', training_folder='./',
                                     hyperparameters={}, model_variant='',
                                     training_number=1, training_state=TrainerState.Initialized.value,
                                     categories=[Category(name='class_a', id='uuid_of_class_a', type='classification')],
                                     image_data=[])
        assert trainer._get_new_best_training_state() is None, 'should not find any models'

        model_path = 'result/weights/published/latest.pt'
        mock_epoch({'class_a': {'fp': 0, 'tp': 1, 'fn': 0}})
        model = trainer._get_new_best_training_state()
        assert model is not None and model.confusion_matrix is not None and model.confusion_matrix[
            'uuid_of_class_a']['tp'] == 1
        trainer._on_metrics_published(model)
        assert os.path.isfile(model_path)
        modification_date = os.path.getmtime(model_path)

        mock_epoch({'class_a': {'fp': 1, 'tp': 2, 'fn': 1}})
        model = trainer._get_new_best_training_state()
        assert model is not None and model.confusion_matrix is not None and model.confusion_matrix[
            'uuid_of_class_a']['tp'] == 2
        trainer._on_metrics_published(model)

        assert trainer._get_new_best_training_state() is None, 'again we should not find any new models'

        await asyncio.sleep(0.1)

        mock_epoch({'class_a': {'fp': 0, 'tp': 3, 'fn': 1}})
        model = trainer._get_new_best_training_state()
        assert model is not None and model.confusion_matrix is not None and model.confusion_matrix[
            'uuid_of_class_a']['tp'] == 3
        trainer._on_metrics_published(model)
        assert os.path.isfile(model_path)
        new_modification_date = os.path.getmtime(model_path)
        assert new_modification_date > modification_date

        # get_latest_model_file
        files = await trainer._get_latest_model_files()
        assert files == {'yolov5_cla_pytorch': ['/tmp/model.pt', '/tmp/test_training/result/opt.yaml']}

    def test_cla_old_model_files_are_deleted_on_publish(self, use_training_dir):
        """When a model is published, the old model files should be deleted"""
        trainer = Yolov5TrainerLogic()
        trainer._training = Training(id='someid', context=Context(organization='o', project='p'),
                                     project_folder='./', images_folder='./', training_folder='./',
                                     categories=[Category(name='class_a', id='uuid_of_class_a', type='classification')],
                                     hyperparameters={}, model_variant='',
                                     image_data=[], training_number=1, training_state=TrainerState.Initialized.value)
        assert trainer._get_new_best_training_state() is None, 'should not find any models'

        mock_epoch({'class_a': {'fp': 0, 'tp': 1, 'fn': 0}})
        model = trainer._get_new_best_training_state()
        assert model is not None and model.confusion_matrix is not None and model.confusion_matrix[
            'uuid_of_class_a']['tp'] == 1
        mock_epoch({'class_a': {'fp': 1, 'tp': 2, 'fn': 1}})

        _, _, files = next(os.walk("result/weights"))
        assert len(files) == 2

        model = trainer._get_new_best_training_state()
        assert model is not None
        trainer._on_metrics_published(model)
        _, _, files = next(os.walk("result/weights/published"))
        assert len(files) == 1
        assert os.path.isfile('result/weights/published/latest.pt')

        _, _, files = next(os.walk("result/weights"))
        assert len(files) == 0

    async def test_cla_clear_training_data(self, use_training_dir):
        """The training data should be cleared except for the last_training.log and last.pt files"""
        trainer = Yolov5TrainerLogic()
        os.makedirs('./data/o/p/trainings/some_uuid', exist_ok=True)
        trainer._training = Training(id='someid', context=Context(organization='o', project='p'), project_folder='./',
                                     images_folder='./', training_folder='./data/o/p/trainings/some_uuid',
                                     categories=[], hyperparameters={}, model_variant='',
                                     image_data=[], training_number=1, training_state=TrainerState.Initialized.value)
        os.makedirs(f'{trainer.training.training_folder}/result/weights/', exist_ok=True)
        os.makedirs(f'{trainer.training.training_folder}/result/weights/published/', exist_ok=True)

        # Create needed .pt files and some dummy data
        with open(f'{trainer.training.training_folder}/result/weights/published/best.pt', 'wb') as f:
            f.write(b'0')
        with open(f'{trainer.training.training_folder}/result/weights/best.pt', 'wb') as f:
            f.write(b'0')
        with open(f'{trainer.training.training_folder}/result/weights/last.pt', 'wb') as f:
            f.write(b'0')
        with open(f'{trainer.training.training_folder}/last_training.log', 'wb') as f:
            f.write(b'0')

        data = glob.glob(trainer.training.training_folder + '/**', recursive=True)
        assert len(data) == 8
        files = [f for f in data if os.path.isfile(f)]
        assert len(files) == 4

        await trainer._clear_training_data(trainer.training.training_folder)
        data = glob.glob(trainer.training.training_folder + '/**', recursive=True)
        assert len(data) == 5
        files = [f for f in data if os.path.isfile(f)]
        assert len(files) == 2  # Note: Do not delete last_training.log und last.pt

    def test_cla_parse_detections_file(self):
        os.makedirs('/tmp/results', exist_ok=True)
        with open('/tmp/results/detection.txt', 'w') as f:
            f.writelines(['1.00 Thripse\n', '0.00 Kastanienminiermotte\n',
                          '0.00 Johannisbeerblasenlaus\n', '0.00 Blattlaeuse\n', '0.00 Wolllaeuse'])
        model_info = ModelInformation(id='someid', organization='o', project='p', host='h', version='1.1', categories=[
            Category(id='some_id', name='Thripse', type=CategoryType.Classification)], resolution=320)
        file = next(os.scandir('/tmp/results'))

        detections = Yolov5TrainerLogic._parse_file_cla(model_info=model_info, filepath=file.path)
        assert len(detections) > 0
        os.remove('/tmp/results/detection.txt')

    async def test_cla_create_file_structure(self, use_training_dir):
        """The file structure should be created correctly"""
        categories = [Category(name='classification_category_1', id='uuid_of_class_1'),
                      Category(name='classification_category_2', id='uuid_of_class_2')]
        image_data = [{'set': 'train',
                       'id': 'image_1',
                       'width': 100,
                       'height': 100,
                       'box_annotations': [],
                       'point_annotations': [],
                       'classification_annotation': {
                             'category_id': 'uuid_of_class_1', }},
                      {'set': 'test',
                       'id': 'image_2',
                       'width': 100,
                       'height': 100,
                       'box_annotations': [],
                       'point_annotations': [],
                       'classification_annotation': {
                           'category_id': 'uuid_of_class_2', }}]
        trainer = Yolov5TrainerLogic()
        trainer._training = Training(id='someid', context=Context(organization='o', project='p'),
                                     project_folder='./', images_folder='./', training_folder='./',
                                     image_data=image_data, categories=categories, hyperparameters={},
                                     model_variant='', training_number=1,
                                     training_state=TrainerState.Initialized.value)

        yolov5_format.create_file_structure_cla(trainer.training)

        assert Path('./train/classification_category_1/image_1.jpg').is_symlink()
        assert Path('./test/classification_category_2/image_2.jpg').is_symlink()


# =======================================================================================================================
# ---------------------------------------------- HELPERS ----------------------------------------------------------------
# =======================================================================================================================


async def download_training_data(images_folder: str, data_exchanger: DataExchanger, glc: LoopCommunicator
                                 ) -> tuple[list[Category], list[dict]]:
    image_data, _ = await TrainingsDownloader(data_exchanger).download_training_data(images_folder)

    response = await glc.get(f"/{os.environ['LOOP_ORGANIZATION']}/projects/{os.environ['LOOP_PROJECT']}/data")
    assert response.status_code != 401, 'Authentification error - did you set LOOP_USERNAME and LOOP_PASSWORD in your environment?'
    assert response.status_code == 200
    data = response.json()
    categories = Category.from_list(
        [category for category in data['categories'] if category['type'] == 'classification'])

    return categories, image_data


def mock_epoch(confusion_matrix: dict) -> None:
    os.makedirs('result/weights/', exist_ok=True)
    with open('result/weights/best.json', 'w') as f:
        json.dump(confusion_matrix, f)
    with open('result/weights/best.pt', 'wb') as f:
        f.write(b'0')
