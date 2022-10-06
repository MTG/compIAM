"""Timbre analysis tools

    You will find in this module the tools and models for the timbre analysis of Indian Art Music. 
"""


import os

import numpy as np
import pandas as pd

from .stroke_classification.model import StrokeClassification
from .stroke_classification.feature_extraction import \
    features_for_pred, process_strokes, features_for_pred, normalise_features

from compiam.exceptions import ModelNotTrainedError

class MridangamStrokeClassification:
    """Mridangam stroke classification.
    """
    def __init__(self):
        """Mridangam stroke classification init method.
        """
        self.dataset = None
        self.generic_model = StrokeClassification()
        self.model = None
        self.feature_list = None

    def load_mridangam_dataset(self, data_home=None, version="default", download=True):
        """Load mirdata dataloader for mirdangam stroke.

        :param data_home: folder where the dataset is found.
        :param version: version of the dataset to use.
        :param download: if True the dataset is downloaded.
        :returns: None, but initializes the dataset of the class and the file dict of strokes.
        """
        from compiam import load_dataset  # Importing load function here to avoid circular imports
        self.dataset = load_dataset('mridangam_stroke', data_home=data_home, version=version)
        self.data_home = self.dataset.data_home
        if download:
            self.dataset.download()
            self.dataset.validate()
        else:
            if not os.path.exists(os.path.join(self.data_home, 'mridangam_stroke_1.5')):
                raise ValueError("Dataset not found, please re-run load_dataset with download=True") 
        self.mridangam_ids = self.dataset.track_ids  # Load Mridangam IDs
        self.mridangam_data = self.dataset.load_tracks()  # Load Mridangam data

        self.stroke_names = self.list_strokes()
        self.stroke_dict = {item: [] for item in self.stroke_names}
        for i in self.mridangam_ids:
            self.stroke_dict[self.mridangam_data[i].stroke_name].append(self.mridangam_data[i].audio_path)

    def list_strokes(self):
        """List available mridangam strokes in the dataset.

        :returns: list of strokes in the datasets.
        """
        stroke_names = []
        for i in self.mridangam_ids:
            stroke_names.append(self.mridangam_data[i].stroke_name)
        return np.unique(stroke_names)

    def dict_strokes(self):
        """List and convert to indexed dict the available mridangam strokes in the dataset.

        :returns: dict with strokes as values and unique integer as keys.
        """
        stroke_names = []
        for i in self.mridangam_ids:
            stroke_names.append(self.mridangam_data[i].stroke_name)
        stroke_names = np.unique(stroke_names)
        return {idx:x for idx, x in enumerate(stroke_names)}

    def train_model(self, model_type="svm", load_computed=True):
        """ List and convert to indexed dict the available mridangam strokes in the dataset
        :returns: dict with strokes as values and unique integer as keys
        """
        if self.dataset is None:
            raise ValueError("Dataset not found, please run load_mridangam_dataset") 
        df_features, self.feature_list = process_strokes(self.dict_strokes(), load_computed=load_computed)
        self.model = self.generic_model.train(df_features, self.feature_list, model_type=model_type)

    def predict(self, file_list):
        """Predict stroke type from list of files.

        :param file_list: list of files for prediction.
        :returns: dict containing filenames as keys and estimated strokes as values.
        """
        if self.model is None:
            raise ModelNotTrainedError("The model is not trained. Please run train_model().")

        if not isinstance(file_list, list):
            file_list = [file_list]

        list_of_feats = []
        for input_file in file_list:
            list_of_feats.append(features_for_pred(input_file=input_file))
        
        list_of_feats = pd.DataFrame(list_of_feats, columns=self.feature_list[:-1])
        data_modif = normalise_features(list_of_feats)
        input_feat = data_modif.iloc[:,:].values 
        pred_strokes = self.model.predict(input_feat)
        return {x:y for x,y in zip(file_list, [self.dict_strokes()[x] for x in pred_strokes])}
