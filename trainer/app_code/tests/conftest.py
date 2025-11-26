import asyncio
import logging
import os
import shutil
import subprocess

import pytest
from _pytest.fixtures import SubRequest

# from dotenv import load_dotenv
from learning_loop_node.data_classes import Context
from learning_loop_node.data_exchanger import DataExchanger
from learning_loop_node.loop_communication import LoopCommunicator

# pylint: disable=unused-argument,redefined-outer-name

logging.basicConfig(level=logging.INFO)

# load_dotenv()

# -------------------- Session fixtures --------------------


@pytest.fixture(scope="session")
def prepare_model():
    """Download model for testing"""
    if not os.path.exists('app_code/tests/test_data/model.pt'):
        url = 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt'
        result = subprocess.run(f'curl  -L {url} -o app_code/tests/test_data/model.pt', shell=True, check=True)
        assert result.returncode == 0
    assert os.path.exists('app_code/tests/test_data/model.pt')
    yield

# -------------------- Class marks --------------------


@pytest.fixture(autouse=True, scope='class')
async def check_marks(request: SubRequest, glc: LoopCommunicator):  # pylint: disable=redefined-outer-name
    """Set environment variables for testing and generate project if requested"""

    markers = list(request.node.iter_markers('environment'))
    assert len(markers) <= 1, 'Only one environment marker allowed'
    if len(markers) == 1:
        marker = markers[0]
        os.environ['LOOP_ORGANIZATION'] = marker.kwargs['organization']
        os.environ['LOOP_PROJECT'] = marker.kwargs['project']

    markers = list(request.node.iter_markers('generate_project'))
    assert len(markers) <= 1, 'Only one generate_project marker allowed'
    if len(markers) == 1:
        marker = markers[0]
        configuration: dict = marker.kwargs['configuration']
        project = configuration['project_name']
        # May not return 200 if project does not exist
        await glc.delete(f"/zauberzeug/projects/{project}?keep_images=true")
        await asyncio.sleep(5)
        assert (await glc.post("/zauberzeug/projects/generator", json=configuration)).status_code == 200
        await asyncio.sleep(5)
        yield
        await asyncio.sleep(5)
        await glc.delete(f"/zauberzeug/projects/{project}?keep_images=true")
        # assert (await lc.delete(f"/zauberzeug/projects/{project}?keep_images=true")).status_code == 200
    else:
        yield


# -------------------- Optional fixtures --------------------

@pytest.fixture(scope="session")
async def glc():
    """The same LoopCommunicator is used for all tests
    Credentials are read from environment variables"""

    lc = LoopCommunicator()
    await lc.ensure_login()
    yield lc
    await lc.shutdown()


@pytest.fixture()
def data_exchanger(glc: LoopCommunicator):  # pylint: disable=redefined-outer-name
    context = Context(organization=os.environ['LOOP_ORGANIZATION'], project=os.environ['LOOP_PROJECT'])
    dx = DataExchanger(context, glc)
    yield dx


@pytest.fixture()
def use_training_dir(prepare_model, request: SubRequest):
    """Step into a temporary directory for training tests and back out again"""

    shutil.rmtree('/tmp/test_training', ignore_errors=True)
    os.makedirs('/tmp/test_training', exist_ok=True)
    shutil.copyfile('app_code/tests/test_data/model.pt', '/tmp/test_training/model.pt')
    os.chdir('/tmp/test_training/')
    yield
    shutil.rmtree('/tmp/test_training', ignore_errors=True)
    os.chdir(request.config.invocation_dir)  # type: ignore
