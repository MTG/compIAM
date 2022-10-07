import pytest
import warnings

def pytest_addoption(parser):
    parser.addoption("--tensorflow", action="store_true", default=False, help="optional tensorflow dependency")
    parser.addoption("--torch", action="store_true", default=False, help="optional torch dependency")
    parser.addoption("--essentia", action="store_true", default=False, help="optional essentia dependency")

@pytest.fixture(scope="session")
def skip_tensorflow(request):
    return request.config.getoption("--tensorflow")

@pytest.fixture(scope="session")
def skip_torch(request):
    return request.config.getoption("--torch")

@pytest.fixture(scope="session")
def skip_essentia(request):
    return request.config.getoption("--essentia")

def pytest_configure(config):
    config.addinivalue_line("markers", "essentia: run optional essentia tests")
    config.addinivalue_line("markers", "tensorflow: run optional tensorflow tests")
    config.addinivalue_line("markers", "torch: run optional torch tests")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--tensorflow"):
        skip_tensorflow = pytest.mark.skip(reason="need --tensorflow option to run")
        for item in items:
            if "tensorflow" in item.keywords:
                item.add_marker(skip_tensorflow)
    if not config.getoption("--torch"):
        skip_torch = pytest.mark.skip(reason="need --torch option to run")
        for item in items:
            if "torch" in item.keywords:
                item.add_marker(skip_torch)
    if not config.getoption("--essentia"):
        skip_essentia = pytest.mark.skip(reason="need --essentia option to run")
        for item in items:
            if "essentia" in item.keywords:
                item.add_marker(skip_essentia)
    else:
        if config.getoption("--tensorflow"):
            warnings.warn("""
                Installing essentia and tensorflow independently can be source of unexpected errors.
                Please test them separatedly.
                We have already implemented an automatic test to check how essentia and tensorflow work together.
                Omitting tests...
            """)
            skip_tensorflow = pytest.mark.skip(reason="need --tensorflow option to run")
            for item in items:
                if "tensorflow" in item.keywords:
                    item.add_marker(skip_tensorflow)
            skip_essentia = pytest.mark.skip(reason="need --essentia option to run")
            for item in items:
                if "essentia" in item.keywords:
                    item.add_marker(skip_essentia)
    return