import os
import mirdata

from importlib import import_module

from compiam import melody, rhythm, structure, timbre
from compiam.dunya import Corpora
from compiam.data import models_dict, datasets_list, WORKDIR
from compiam.exceptions import ModelNotDefinedError


def load_model(
    model_name, data_home=None, models_dict=models_dict, version=None, **kwargs
):
    """Wrapper for loading pre-trained models.

    :param model_name: name of the model, extractors, or algorithm to load.
    :param data_home: path where the data lives. If None uses the default.
    :param models_dict: dict object including the available models.
    :param version: which version of the model to load.
    :returns: specific Class of the selected model.
    """
    if not model_name in models_dict:
        raise ModelNotDefinedError(
            (
                f"Model, {model_name} does not exist in compiam.data.models_dict, please follow "
                "instructions for adding new model to in ``data.py`` documentation"
            )
        )
    m_dict = models_dict[model_name]
    version_ = m_dict["default_version"] if not version else version
    if version_ not in m_dict["kwargs"]:
        raise ValueError(
            f"""
            Model {model_name} does not have a version {version_}.
            Available versions are: {list(m_dict['kwargs'].keys())}.
        """
        )
    model_kwargs = m_dict["kwargs"][version_]
    kwarg_paths = [x for x in list(model_kwargs.keys()) if "_path" in x]
    for kp in kwarg_paths:
        if isinstance(model_kwargs[kp], dict):
            for k in list(model_kwargs[kp].keys()):
                model_kwargs[kp][k] = (
                    os.path.join(data_home, model_kwargs[kp][k])
                    if data_home is not None
                    else os.path.join(WORKDIR, model_kwargs[kp][k])
                )
        else:
            model_kwargs[kp] = (
                os.path.join(data_home, model_kwargs[kp])
                if data_home is not None
                else os.path.join(WORKDIR, model_kwargs[kp])
            )

    module = getattr(import_module(m_dict["module_name"]), m_dict["class_name"])
    model_kwargs.update(kwargs)
    return module(**model_kwargs)


def load_dataset(dataset_name, data_home=None, version="default"):
    """Alias function to load a mirdata Dataset class.

    :param dataset_name: the dataset's name, see mirdata.DATASETS for a
        complete list of possibilities.
    :param data_home: path where the data lives. If None uses the default
        home location.
    :param version: which version of the dataset to load. If None, the
        default version is loaded.
    :returns: a mirdata.core.Dataset object.
    """
    if dataset_name not in datasets_list:
        raise ValueError("Invalid dataset {}".format(dataset_name))
    return mirdata.initialize(
        dataset_name=dataset_name, data_home=data_home, version=version
    )


def load_corpora(tradition, token=None):
    """Wrapper to load access to the Dunya corpora.

    :param tradition: carnatic or hindustani.
    :param token: Dunya personal access token.
    :returns: a compiam.Corpora object.
    """
    if token is None:
        raise ImportError(
            """Please initialize the Corpora introducing your Dunya API token as parameter. 
            To get your token, first register to https://dunya.compmusic.upf.edu/ and then go to your user 
            page by clicking at your username at te right top of the webpage. You will find the API token 
            in the "API Access" section. Request restricted access if needed. Thanks.
        """
        )
    return Corpora(tradition=tradition, token=token)


def list_models():
    """Just listing the available models.

    :returns: a list of available models.
    """
    return list(models_dict.keys())


def get_model_info(model_key):
    """Get complete info in data/models_dict for a particular pre-trained model

    :param model_key: model key from models_dict
    :returns: information about a particular model.
    """
    if model_key not in list(models_dict.keys()):
        raise ValueError(
            "Please enter a valid model key from {}".format(list(models_dict.keys()))
        )
    return models_dict[model_key]


def list_datasets():
    """Just listing the available datasets.

    :returns: a list of available datasets.
    """
    return datasets_list
