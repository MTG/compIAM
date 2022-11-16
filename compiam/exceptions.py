class ModelNotFoundError(Exception):
    pass


class ModelNotDefinedError(Exception):
    pass


class ModelNotTrainedError(Exception):
    pass


class DatasetNotLoadedError(Exception):
    pass


class HTTPError(Exception):
    pass


class ConnectionError(Exception):
    pass
