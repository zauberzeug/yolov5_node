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


@pytest.fixture(scope="function")
def use_training_dir(request):
    os.chdir('/tmp/test_training/')
    yield
    os.chdir(request.config.invocation_dir)
