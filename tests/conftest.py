import pytest

def pytest_addoption(parser):
    parser.addoption("--tensorflow", action="store_true", default=False, help="optional tensorflow dependency")
    parser.addoption("--torch", action="store_true", default=False, help="optional tensorflow dependency")


@pytest.fixture(scope="session")
def skip_tensorflow(request):
    return request.config.getoption("--tensorflow")

@pytest.fixture(scope="session")
def skip_torch(request):
    return request.config.getoption("--torch")


def pytest_configure(config):
    config.addinivalue_line("markers", "tensorflow: run optional tensorflow tests")
    config.addinivalue_line("markers", "torch: run optional torch tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--tensorflow"):
        # --runslow given in cli: do not skip slow tests
        return
    if config.getoption("--torch"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_tensorflow = pytest.mark.skip(reason="need --tensorflow option to run")
    skip_torch = pytest.mark.skip(reason="need --torch option to run")
    for item in items:
        if "tensorflow" in item.keywords:
            item.add_marker(skip_tensorflow)
        if "torch" in item.keywords:
            item.add_marker(skip_torch)