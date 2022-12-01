import os
import yaml
import inspect
import logging


def get_logger(name):
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger(__name__)


def create_if_not_exists(path):
    """If the directory at <path> does not exist, create it empty

    :param path: path to folder
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ""):
        os.makedirs(directory)


def get_tool_list(modules):
    """Given sys.modules[__name__], prints out the imported classes

    :param modules: basically the sys.modules[__name__] of a file
    """
    list_of_tools = []
    for _, obj in inspect.getmembers(modules):
        if inspect.isclass(obj):
            list_of_tools.append(obj.__name__)
    return list_of_tools
