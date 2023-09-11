import logging
import os
import shutil
import subprocess

import icecream
import pytest
from dotenv import load_dotenv
from learning_loop_node.data_classes import Context
from learning_loop_node.data_exchanger import DataExchanger
from learning_loop_node.loop_communication import LoopCommunicator

icecream.install()
logging.basicConfig(level=logging.INFO)

# load_dotenv()


@pytest.fixture()
async def glc():
    loop_communicator = LoopCommunicator()
    yield loop_communicator
    await loop_communicator.shutdown()


@pytest.fixture()
async def data_exchanger():
    loop_communicator = LoopCommunicator()
    context = Context(organization='zauberzeug', project='demo')
    dx = DataExchanger(context, loop_communicator)
    yield dx
    await loop_communicator.shutdown()


@pytest.fixture(scope="function")
def use_training_dir(request):
    shutil.rmtree('/tmp/test_training', ignore_errors=True)
    os.makedirs('/tmp/test_training', exist_ok=True)

    # TODO Download has to be done every time, otherwise the pt file may be faulty
    # if not os.path.isfile('/tmp/model.pt'):
    print('--------------Downloading model>')
    url = 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt'
    result = subprocess.run(f'curl  -L {url} -o /tmp/model.pt', shell=True, check=True)
    assert result.returncode == 0

    shutil.copyfile('/tmp/model.pt', '/tmp/test_training/model.pt')

    os.chdir('/tmp/test_training/')
    yield
    os.chdir(request.config.invocation_dir)


@pytest.fixture()
async def create_project():
    lc = LoopCommunicator()
    await lc.delete("/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {
        'project_name': 'pytest', 'box_categories': 2, 'point_categories': 1, 'inbox': 0, 'annotate': 0, 'review': 0,
        'complete': 0, 'image_style': 'plain', 'thumbs': False, 'trainings': 1}
    assert (await lc.post("/zauberzeug/projects/generator", json=project_configuration)).status_code == 200
    yield
    await lc.delete("/zauberzeug/projects/pytest?keep_images=true")
    await lc.shutdown()


@pytest.fixture()
async def create_cla_project():
    lc = LoopCommunicator()
    await lc.delete("/zauberzeug/projects/pytest?keep_images=true")
    project_configuration = {'project_name': 'pytest', 'classification_categories': 2, 'inbox': 0,
                             'annotate': 0, 'review': 0, 'complete': 0, 'image_style': 'plain', 'thumbs': False, 'trainings': 1}
    assert (await lc.post("/zauberzeug/projects/generator", json=project_configuration)).status_code == 200
    yield
    await lc.delete("/zauberzeug/projects/pytest?keep_images=true")
    await lc.shutdown()
