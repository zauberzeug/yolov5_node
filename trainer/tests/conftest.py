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
        subprocess.run('''curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76" > /dev/null
    curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1Tdn3yqpZ79X7R1Ql0zNlNScB1Dv9Fp76" -o /tmp/model.pt
    rm ./cookie
    ''', shell=True)
    shutil.copyfile('/tmp/model.pt', '/tmp/test_training/model.pt')
    shutil.copyfile('/yolor/cfg/yolor_p6.cfg', '/tmp/test_training/model.cfg')


@pytest.fixture(scope="function")
def use_test_dir(request):
    os.chdir(request.fspath.dirname)
    yield
    os.chdir(request.config.invocation_dir)
