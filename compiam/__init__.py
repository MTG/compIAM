from compiam.model_store.data import models_dict
from compiam.exceptions import ModelNotDefinedError

def load_model(model_name, models_dict=models_dict):
    if not model_name in models_dict:
        raise ModelNotDefinedError(
            (f'Model, {model_name} does not exist in compiam.model_store.models_dict, please follow ' 
                'instructions for adding new model to the model_store in model_store documentation'))

    m_dict = models_dict[model_name]

    return m_dict['wrapper'](m_dict['filepath'], **m_dict['kwargs'])


def load_dataset():
    pass