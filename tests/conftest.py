import pytest

OPTIONS = [
    "tensorflow",
    "torch",
    "essentia",
    "essentia_tensorflow",
    "essentia_torch",
    "full_ml",
    "all",
]


def pytest_addoption(parser):
    for option in OPTIONS:
        parser.addoption(
            "--" + option,
            action="store_true",
            default=False,
            help="test setting for " + option.replace("_", " and "),
        )


@pytest.fixture(scope="session")
def skip_tensorflow(request):
    return request.config.getoption("--tensorflow")


@pytest.fixture(scope="session")
def skip_torch(request):
    return request.config.getoption("--torch")


@pytest.fixture(scope="session")
def skip_essentia(request):
    return request.config.getoption("--essentia")


@pytest.fixture(scope="session")
def skip_essentia_tensorflow(request):
    return request.config.getoption("--essentia_tensorflow")


@pytest.fixture(scope="session")
def skip_essentia_torch(request):
    return request.config.getoption("--essentia_torch")


@pytest.fixture(scope="session")
def skip_full_ml(request):
    return request.config.getoption("--full_ml")


@pytest.fixture(scope="session")
def skip_all(request):
    return request.config.getoption("--all")


def pytest_configure(config):
    for option in OPTIONS:
        config.addinivalue_line(
            "markers",
            option + ": run optional " + option.replace("_", "and") + " tests",
        )


def _skip_tests_or_no(config, items, option, option_flag, running_test):
    # No option has been added
    if option_flag == -1:
        if not config.getoption("--" + option):
            skip_option = pytest.mark.skip(reason="need --" + option + " option to run")
            for item in items:
                if option in item.keywords:
                    item.add_marker(skip_option)
        else:
            running_test = option
            option_flag = 0
        return config, items, option_flag, running_test

    # One option already added. Warning user...
    elif option_flag == 0:
        if not config.getoption("--" + option):
            skip_option = pytest.mark.skip(reason="need --" + option + " option to run")
            for item in items:
                if option in item.keywords:
                    item.add_marker(skip_option)
        else:
            print(
                "\n\nIMPORTANT: You have entered two testing markers. "
                + "Please do run test options one by one. You will find the "
                + "available options in the docs or by running pytest --markers "
                + "in the terminal. Running tests only for: "
                + running_test
                + "\n"
            )
            skip_option = pytest.mark.skip(reason="need --" + option + " option to run")
            for item in items:
                if option in item.keywords:
                    item.add_marker(skip_option)
            option_flag = 1
        return config, items, option_flag, running_test

    # User already warned. No need for multiple warnings.
    elif option_flag == 1:
        if not config.getoption("--" + option):
            skip_option = pytest.mark.skip(reason="need --" + option + " option to run")
            for item in items:
                if option in item.keywords:
                    item.add_marker(skip_option)
        else:
            skip_option = pytest.mark.skip(reason="need --" + option + " option to run")
            for item in items:
                if option in item.keywords:
                    item.add_marker(skip_option)
        return config, items, option_flag, running_test


def pytest_collection_modifyitems(config, items):
    running_test = None
    option_flag = -1
    for option in OPTIONS:
        config, items, option_flag, running_test = _skip_tests_or_no(
            config, items, option, option_flag, running_test
        )
    return
