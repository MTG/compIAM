import os
import yaml
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
    """
    If the directory at <path> does not exist, create it empty
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ""):
        os.makedirs(directory)


def load_yaml(path):
    """
    Load yaml at <path> to dictionary, d

    Returns
    =======
    Wrapper dictionary, D where
    D = {filename: d}
    """
    import zope.dottedname.resolve

    def constructor_dottedname(loader, node):
        value = loader.construct_scalar(node)
        return zope.dottedname.resolve.resolve(value)

    yaml.add_constructor("!dottedname", constructor_dottedname)

    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d