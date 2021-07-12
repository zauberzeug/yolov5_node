import pytest
import os
import icecream

icecream.install()


@pytest.fixture(scope="function")
def use_test_dir(request):
    os.chdir(request.fspath.dirname)
    yield
    os.chdir(request.config.invocation_dir)
