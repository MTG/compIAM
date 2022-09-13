import logging

def get_logger(name):
    logging.basicConfig(format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger(__name__)


def split_data(data, train_perc=0.7):
	return 

