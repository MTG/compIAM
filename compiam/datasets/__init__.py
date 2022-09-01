import mirdata

DATASETS = ['saraga_carnatic', 'saraga_hindustani', 'mridangam_stroke']

def list_datasets():
    """Get a list of all mirdata dataset names
    Returns:
        list: list of dataset names as strings
    """
    return DATASETS

def initialize(dataset_name, data_home=None, version="default"):
    """Alias function to load a mirdata Dataset class
    Args:
        dataset_name (str): the dataset's name
            see mirdata.DATASETS for a complete list of possibilities
        data_home (str or None): path where the data lives. If None
            uses the default location.
        version (str or None): which version of the dataset to load.
            If None, the default version is loaded.
    Returns:
        Dataset: a mirdata.core.Dataset object
    """
    if dataset_name not in DATASETS:
        raise ValueError("Invalid dataset {}".format(dataset_name)) 
    return mirdata.initialize(dataset_name=dataset_name, data_home=data_home, version=version)