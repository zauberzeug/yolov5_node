import asyncio
import glob
import json
import logging
import os
import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from learning_loop_node.data_classes import Category, Context, Training
from learning_loop_node.data_exchanger import DataExchanger
from learning_loop_node.enums import TrainerState
from learning_loop_node.helpers.misc import create_image_folder
from learning_loop_node.loop_communication import LoopCommunicator
from learning_loop_node.trainer.downloader import TrainingsDownloader
from learning_loop_node.trainer.executor import Executor
from ruamel.yaml import YAML

from .. import model_files, yolov5_format
from ..yolov5_format import set_hyperparameters_in_file
from ..yolov5_trainer import Yolov5TrainerLogic

# pylint: disable=protected-access,unused-argument

logging.basicConfig(level=logging.DEBUG)
yaml = YAML()

project_configuration = {
    'project_name': 'pytest_yolo5det', 'box_categories': 2, 'point_categories': 2, 'inbox': 4, 'annotate': 0, 'review': 0,
    'complete': 14, 'image_style': 'beautiful', 'thumbs': False, 'trainings': 1}


@pytest.mark.environment(organization='zauberzeug', project='pytest_yolo5det', mode='DETECTION')
@pytest.mark.generate_project(configuration=project_configuration)
class TestWithLoop:
    """This test environment sets up the environment vars and
    a test project in the loop which is used for testing."""

    @pytest.mark.usefixtures('use_training_dir')
    async def test_training_creates_model(self, data_exchanger: DataExchanger, glc: LoopCommunicator):
        """Test if training creates a model"""

        project_folder = os.getcwd()
        images_folder = create_image_folder(project_folder)
        categories, image_data = await download_training_data(images_folder, data_exchanger, glc)
        training = Training(id=str(uuid4()),
                            project_folder=project_folder,
                            training_folder=project_folder + '/training',
                            images_folder=images_folder,
                            model_variant='',
                            context=Context(project='pytest_yolo5det', organization='zauberzeug'),
                            categories=categories, hyperparameters={}, training_number=1,
                            training_state=TrainerState.Initialized.value,
                            image_data=image_data)
        yolov5_format.create_file_structure(training)
        executor = Executor(os.getcwd())
        # from https://github.com/WongKinYiu/yolor#training
        ROOT = Path(__file__).resolve().parents[2]
        cmd = f'python {ROOT/"train_det.py"} --project training --name result --batch 4 --img 416 --data training/dataset.yaml --weights model.pt --epochs 1'
        await executor.start(cmd, env={'WANDB_MODE': 'disabled'})
        while executor.is_running():
            await asyncio.sleep(1)
        assert '1 epochs completed' in executor.get_log()
        assert 'best.pt' in executor.get_log()
        best = training.training_folder + '/result/weights/best.pt'
        assert os.path.isfile(best)

    @pytest.mark.usefixtures('use_training_dir')
    async def test_parse_progress_from_log(self, data_exchanger: DataExchanger, glc: LoopCommunicator):
        """Test if progress is parsed correctly from log"""
        trainer = Yolov5TrainerLogic()
        trainer.epochs = 2
        project_folder = os.getcwd()
        images_folder = create_image_folder(project_folder)
        categories, image_data = await download_training_data(images_folder, data_exchanger, glc)
        trainer._training = Training(
            id=str(uuid4()),
            project_folder=project_folder,
            training_folder=project_folder + '/training',
            images_folder=images_folder,
            model_variant='',
            context=Context(project='pytest_yolo5det', organization='zauberzeug'),
            categories=categories, hyperparameters={}, training_number=1,
            training_state=TrainerState.Initialized.value,
            image_data=image_data,
        )
        yolov5_format.create_file_structure(trainer.training)

        trainer._executor = Executor(os.getcwd())
        ROOT = Path(__file__).resolve().parents[2]
        cmd = f'python {ROOT/"train_det.py"} --project training --name result --batch 4 --img 416 --data training/dataset.yaml --weights model.pt --epochs {trainer.epochs}'
        await trainer.executor.start(cmd, env={'WANDB_MODE': 'disabled'})
        while trainer.executor.is_running():
            await asyncio.sleep(1)

        logging.info(trainer.executor.get_log())
        assert f'{trainer.epochs} epochs completed' in trainer.executor.get_log()
        assert trainer.training_progress == 1.0

# =======================================================================================================================
# ----------------- The following tests do not need a loop project as they are not using the loop -----------------------
# =======================================================================================================================


@pytest.mark.environment(organization='', project='', mode='DETECTION')
class TestWithDetection:

    @pytest.mark.usefixtures('use_training_dir')
    async def test_create_file_structure_box_size(self):
        categories = [Category(name='point_category_1', id='uuid_of_class_1'),
                      Category(name='point_category_2', id='uuid_of_class_2', point_size=30)]
        image_data = [{'set': 'train',  'id': 'image_1', 'width': 100, 'height': 100, 'box_annotations': [],
                       'point_annotations': [{'category_id': 'uuid_of_class_1', 'x': 50, 'y': 60},
                                             {'category_id': 'uuid_of_class_2', 'x': 60, 'y': 70}]},
                      {'set': 'test', 'id': 'image_2', 'width': 100, 'height': 100, 'box_annotations': [],
                       'point_annotations': []}]
        trainer = Yolov5TrainerLogic()
        trainer._training = Training(
            id='someid', context=Context(organization='o', project='p'),
            project_folder='./', images_folder='./', training_folder='./',
            image_data=image_data, categories=categories, hyperparameters={},
            model_variant='', training_number=1,
            training_state=TrainerState.Initialized.value)

        yolov5_format.create_file_structure(trainer.training)

        with open('./train/image_1.txt', 'r') as f:
            lines = f.readlines()

        assert '0 0.500000 0.600000 0.200000 0.200000' in lines[0]
        assert '1 0.600000 0.700000 0.300000 0.300000' in lines[1]

    @pytest.mark.usefixtures('use_training_dir')
    async def test_new_model_discovery(self):
        """This test also triggers the creation of a wts file"""
        trainer = Yolov5TrainerLogic()
        trainer._training = Training(
            id='someid', context=Context(organization='o', project='p'),
            project_folder='./', images_folder='./', training_folder='./', image_data=[],
            categories=[Category(name='class_a', id='uuid_of_class_a', type='box')],
            hyperparameters={}, model_variant='', training_number=1,
            training_state=TrainerState.Initialized.value)

        assert trainer._get_new_best_training_state() is None, 'should not find any models'

        model_path = 'result/weights/published/latest.pt'

        mock_epoch(1, {'class_a': {'fp': 0, 'tp': 1, 'fn': 0}})
        model = trainer._get_new_best_training_state()
        assert model is not None and model.confusion_matrix is not None
        assert model.confusion_matrix['uuid_of_class_a']['tp'] == 1
        trainer._on_metrics_published(model)
        assert os.path.isfile(model_path)
        modification_date = os.path.getmtime(model_path)

        mock_epoch(2, {'class_a': {'fp': 1, 'tp': 2, 'fn': 1}})
        model = trainer._get_new_best_training_state()
        assert model is not None and model.confusion_matrix is not None
        assert model.confusion_matrix['uuid_of_class_a']['tp'] == 2
        trainer._on_metrics_published(model)
        assert trainer._get_new_best_training_state() is None, 'again we should not find any new models'

        await asyncio.sleep(0.1)  # To have a later modification date
        mock_epoch(3, {'class_a': {'fp': 0, 'tp': 3, 'fn': 1}})
        model = trainer._get_new_best_training_state()
        assert model is not None and model.confusion_matrix is not None
        assert model.confusion_matrix['uuid_of_class_a']['tp'] == 3
        trainer._on_metrics_published(model)
        assert os.path.getmtime(model_path) > modification_date

        # TODO: Generation of wts seems buggy in the test environment (training_folder is not set correctly?!)

        # files = await trainer._get_latest_model_files()  # get_latest_model_file
        # assert files == {
        #     'yolov5_pytorch': ['/tmp/model.pt', '/tmp/test_training/hyp.yaml'],
        #     'yolov5_wts': ['/tmp/model.wts']}

    @pytest.mark.usefixtures('use_training_dir')
    def test_newest_model_is_used(self):
        trainer = Yolov5TrainerLogic()
        trainer._training = Training(
            id='someid', context=Context(organization='o', project='p'),
            project_folder='./', images_folder='./', training_folder='./', image_data=[],
            categories=[Category(name='class_a', id='uuid_of_class_a', type='box')],
            hyperparameters={}, model_variant='', training_number=1,
            training_state=TrainerState.Initialized.value)

        # create some models.
        mock_epoch(10, {})
        mock_epoch(200, {})

        new_model = trainer._get_new_best_training_state()
        assert new_model is not None and new_model.meta_information is not None
        assert 'epoch10.pt' not in new_model.meta_information['weightfile']
        assert 'epoch200.pt' in new_model.meta_information['weightfile']

    @pytest.mark.usefixtures('use_training_dir')
    def test_old_model_files_are_deleted_on_publish(self):
        trainer = Yolov5TrainerLogic()
        trainer._training = Training(
            id='someid', context=Context(organization='o', project='p'),
            project_folder='./', images_folder='./', training_folder='./', image_data=[],
            categories=[Category(name='class_a', id='uuid_of_class_a', type='box')],
            hyperparameters={}, model_variant='', training_number=1,
            training_state=TrainerState.Initialized.value)

        assert trainer._get_new_best_training_state() is None, 'should not find any models'

        mock_epoch(1, {'class_a': {'fp': 0, 'tp': 1, 'fn': 0}})
        new_model = trainer._get_new_best_training_state()

        assert new_model is not None and new_model.confusion_matrix is not None
        assert new_model.confusion_matrix['uuid_of_class_a']['tp'] == 1
        mock_epoch(2, {'class_a': {'fp': 1, 'tp': 2, 'fn': 1}})

        _, _, files = next(os.walk("result/weights"))
        assert len(files) == 4

        new_model = trainer._get_new_best_training_state()
        assert new_model is not None
        trainer._on_metrics_published(new_model)
        _, _, files = next(os.walk("result/weights/published"))
        assert len(files) == 1
        assert os.path.isfile('result/weights/published/latest.pt')

        _, _, files = next(os.walk("result/weights"))
        assert len(files) == 0

    @pytest.mark.usefixtures('use_training_dir')
    def test_newer_model_files_are_kept_during_deleting(self):
        trainer = Yolov5TrainerLogic()
        trainer._training = Training(
            id='someid', context=Context(organization='o', project='p'),
            project_folder='./', images_folder='./', training_folder='./', image_data=[],
            categories=[Category(name='class_a', id='uuid_of_class_a', type='box')],
            hyperparameters={}, model_variant='', training_number=1,
            training_state=TrainerState.Initialized.value)

        # create some models.
        mock_epoch(10, {})
        mock_epoch(200, {})
        new_model = trainer._get_new_best_training_state()
        assert new_model is not None and new_model.meta_information is not None
        assert 'epoch200.pt' in new_model.meta_information['weightfile']
        mock_epoch(201, {})  # An epoch is finished after during communication with the LearningLoop

        trainer._on_metrics_published(new_model)

        all_model_files = model_files.get_all_weightfiles(Path(trainer.training.training_folder))
        assert len(all_model_files) == 1
        assert 'epoch201.pt' in str(all_model_files[0]), 'Epoch201 is not yed synced. It should not be deleted.'

    @pytest.mark.usefixtures('use_training_dir')
    async def test_clear_training_data(self):
        trainer = Yolov5TrainerLogic()
        trainer._training = Training(id='someid', context=Context(organization='o', project='p'),
                                     project_folder='./', images_folder='./', training_folder='./',
                                     categories=[], hyperparameters={}, model_variant='',
                                     image_data=[], training_number=1, training_state=TrainerState.Initialized.value)
        os.makedirs(f'{trainer.training.training_folder}/result/weights/', exist_ok=True)
        os.makedirs(f'{trainer.training.training_folder}/result/weights/published/', exist_ok=True)

        # Create needed .pt files and some dummy data
        with open(f'{trainer.training.training_folder}/result/weights/published/some_model_id.pt', 'wb') as f:
            f.write(b'0')
        with open(f'{trainer.training.training_folder}/result/weights/test.txt', 'wb') as f:
            f.write(b'0')
        with open(f'{trainer.training.training_folder}/result/weights/best.pt', 'wb') as f:
            f.write(b'0')
        with open(f'{trainer.training.training_folder}/last_training.log', 'wb') as f:
            f.write(b'0')

        data = glob.glob(trainer.training.training_folder + '/**', recursive=True)
        assert len(data) == 9
        files = [f for f in data if os.path.isfile(f)]
        assert len(files) == 5

        await trainer._clear_training_data(trainer.training.training_folder)
        data = glob.glob(trainer.training.training_folder + '/**', recursive=True)
        assert len(data) == 5
        files = [f for f in data if os.path.isfile(f)]
        assert len(files) == 2  # Note: Do not delete last_training.log and best.pt


def test_update_hyperparameter():
    """The hyperparameter file should be updated with the given hyperparameter"""
    def assert_yaml_content(yaml_path, **kwargs):
        with open(yaml_path) as f:
            content = yaml.load(f)
        for key, value in kwargs.items():
            assert content[key] == value

    shutil.copy('app_code/tests/test_data/hyp.yaml', '/tmp')
    hyperparameter = {'resolution': 600,
                      'fliplr': 0.5,
                      'flipud': 0.5}

    assert_yaml_content('/tmp/hyp.yaml', fliplr=0.0, flipud=0.0)
    set_hyperparameters_in_file('/tmp/hyp.yaml', hyperparameter)
    assert_yaml_content('/tmp/hyp.yaml', fliplr=0.5, flipud=0.5)

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
    categories = Category.from_list(data['categories'])
    return categories, image_data


def mock_epoch(number: int, confusion_matrix: dict) -> None:
    os.makedirs('result/weights/', exist_ok=True)
    with open(f'result/weights/epoch{number}.json', 'w') as f:
        json.dump(confusion_matrix, f)
    shutil.copy('model.pt', f'result/weights/epoch{number}.pt')
