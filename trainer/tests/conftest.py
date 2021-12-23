import pytest
import os
import shutil
import icecream
import subprocess
import logging

icecream.install()
logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="function", autouse=True)
def clear_test_dir():
    shutil.rmtree('/tmp/test_training', ignore_errors=True)
    os.mkdir('/tmp/test_training')

    if not os.path.isfile('/tmp/model.pt'):
        url = 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt'
        subprocess.run(f'curl  -L {url} -o /tmp/model.pt', shell=True)
    shutil.copyfile('/tmp/model.pt', '/tmp/test_training/model.pt')


@pytest.fixture(scope="function")
def use_training_dir(request):
    os.chdir('/tmp/test_training/')
    yield
    os.chdir(request.config.invocation_dir)
