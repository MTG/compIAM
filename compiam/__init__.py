import mirdata
from compiam.data import models_dict, datasets_list
from compiam.exceptions import ModelNotDefinedError

def load_model(model_name, models_dict=models_dict):
    if not model_name in models_dict:
        raise ModelNotDefinedError(
            (f'Model, {model_name} does not exist in compiam.model_store.models_dict, please follow ' 
                'instructions for adding new model to the model_store in model_store documentation'))

    m_dict = models_dict[model_name]

    return m_dict['wrapper'](**m_dict['kwargs'])

def load_dataset(dataset_name, data_home=None, version="default"):
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
    if dataset_name not in datasets_list:
        raise ValueError("Invalid dataset {}".format(dataset_name)) 
    return mirdata.initialize(dataset_name=dataset_name, data_home=data_home, version=version)

def list_datasets():
    """Get a list of all mirdata dataset names
    Returns:
        list: list of dataset names as strings
    """
    return datasets_list