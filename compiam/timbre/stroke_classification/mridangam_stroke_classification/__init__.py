import os
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from compiam.exceptions import ModelNotTrainedError, DatasetNotLoadedError
from compiam.data import WORKDIR
from compiam.utils import get_logger

logger = get_logger(__name__)


class MridangamStrokeClassification:
    """Mridangam stroke classification."""

    def __init__(self):
        """Mridangam stroke classification init method."""
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global features_for_pred, process_strokes, features_for_pred, normalise_features
            from compiam.timbre.stroke_classification.mridangam_stroke_classification.stroke_features import (
                features_for_pred,
                process_strokes,
                features_for_pred,
                normalise_features,
            )
        except:
            raise ImportError(
                "In order to use this tool you need to have essentia installed. "
                "Install compIAM with essentia support: pip install 'compiam[essentia]'"
            )
        ###

        self.dataset = None
        self.model = None
        self.feature_list = None
        self.computed_features_path = os.path.join(
            WORKDIR,
            "models",
            "timbre",
            "mridangam_stroke_classification",
            "pre-computed_features.csv",
        )

    def load_mridangam_dataset(self, data_home=None, version="default", download=True):
        """Load mirdata dataloader for mirdangam stroke.

        :param data_home: folder where the dataset is found.
        :param version: version of the dataset to use.
        :param download: if True the dataset is downloaded.
        :returns: None, but initializes the dataset of the class and the file dict of strokes.
        """
        from compiam import (
            load_dataset,
        )  # Importing load function here to avoid circular imports

        self.dataset = load_dataset(
            "mridangam_stroke", data_home=data_home, version=version
        )
        self.data_home = self.dataset.data_home
        if download:
            self.dataset.download()
            self.dataset.validate()
        else:
            if not os.path.exists(os.path.join(self.data_home, "mridangam_stroke_1.5")):
                raise ValueError(
                    "Dataset not found, please re-run load_dataset with download=True"
                )
        self.mridangam_ids = self.dataset.track_ids  # Load Mridangam IDs
        self.mridangam_tracks = self.dataset.load_tracks()  # Load Mridangam data

        self.stroke_names = self.list_strokes()
        self.stroke_dict = {item: [] for item in self.stroke_names}
        for i in self.mridangam_ids:
            self.stroke_dict[self.mridangam_tracks[i].stroke_name].append(
                self.mridangam_tracks[i].audio_path
            )

    def list_strokes(self):
        """List available mridangam strokes in the dataset.

        :returns: list of strokes in the datasets.
        """
        if self.dataset is None:
            raise DatasetNotLoadedError(
                """
                Please load the dataset using the .load_mridangam_dataset() method or the strokes 
                cannot be listed.
            """
            )

        stroke_names = []
        for i in self.mridangam_ids:
            stroke_names.append(self.mridangam_tracks[i].stroke_name)
        return list(np.unique(stroke_names))

    def dict_strokes(self):
        """List and convert to indexed dict the available mridangam strokes in the dataset.

        :returns: dict with strokes as values and unique integer as keys.
        """
        if self.dataset is None:
            raise DatasetNotLoadedError(
                """
                Please load the dataset using the .load_mridangam_dataset() method or the strokes 
                cannot be listed.
            """
            )

        stroke_names = []
        for i in self.mridangam_ids:
            stroke_names.append(self.mridangam_tracks[i].stroke_name)
        stroke_names = np.unique(stroke_names)
        return {idx: x for idx, x in enumerate(stroke_names)}

    def train_model(
        self, model_type="svm", load_computed=False, balance=False, balance_ref="random"
    ):
        """Train a support vector machine for stroke classification.

        :param model_type: type of model to train.
        :param model_type: bool to indicate if the features are computed or loaded from file.
        :param balance: balance the number of instances per class to prevent biases.
        :param balance_ref: reference class for data balancement.
        :returns: accuracy in percentage and rounded to two decimals
        """
        if (self.dataset is None) and (load_computed is False):
            raise DatasetNotLoadedError(
                "Dataset not found, please run load_mridangam_dataset"
            )
        if (load_computed is True) and not os.path.exists(self.computed_features_path):
            raise ValueError(
                """
                Training data not found. Please check you set the path correctly otherwise run .train_model()
                function with load_computed=False"""
            )
        file_dict = {item: [] for item in self.list_strokes()}
        for i in self.mridangam_ids:
            file_dict[self.mridangam_tracks[i].stroke_name].append(
                self.mridangam_tracks[i].audio_path
            )
        training_data, self.feature_list = process_strokes(
            file_dict, load_computed=load_computed
        )

        # Let"s use sklearn"s preprocessing tools for applying normalisation to features
        data_modif = normalise_features(training_data, self.feature_list)
        if balance == True:
            strokes = training_data.stroke.unique()
            count_dict = training_data["stroke"].value_counts().to_dict()
            min_stroke = min(count_dict, key=count_dict.get)
            min_number = (
                data_modif.stroke.value_counts()[min_stroke]
                if balance_ref == "lower"
                else data_modif.stroke.value_counts()[random.choice(strokes)]
            )
            reshaped_stroke_list = []
            for strk in strokes:
                if count_dict[strk] > min_number:
                    reshaped_stroke_list.append(
                        data_modif[data_modif.stroke == strk].sample(n=min_number)
                    )
                else:
                    reshaped_stroke_list.append(data_modif[data_modif.stroke == strk])
            # Merging after downsampling
            data_modif = pd.concat(reshaped_stroke_list)

        X = data_modif.iloc[:, : len(self.feature_list) - 1].values
        # Creating output values
        data_modif.stroke = pd.Categorical(
            data_modif.stroke
        )  # convert to categorical data
        y = np.array(data_modif.stroke.cat.codes)  # create label encoded outputs

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        if model_type == "svm":
            self.model = svm.SVC(gamma=1 / (X_train.shape[-1] * X_train.var()))
        elif model_type == "mlp":
            self.model = MLPClassifier(alpha=1, max_iter=1000)
        else:
            raise ValueError(
                "Model not available. Please check the available options in the documentation."
            )

        # Fit model with training data
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        logger.info(
            "{} model successfully trained with accuracy {}% in the testing set".format(
                model_type.upper(),
                round(np.sum(y_test == y_pred) / len(y_pred) * 100),
                2,
            )
        )
        return round((np.sum(y_test == y_pred) / len(y_pred) * 100), 2)

    def predict(self, file_list):
        """Predict stroke type from list of files.

        :param file_list: list of files for prediction.
        :returns: dict containing filenames as keys and estimated strokes as values.
        """
        if self.model is None:
            raise ModelNotTrainedError(
                "The model is not trained. Please run train_model()."
            )

        if not isinstance(file_list, list) and (isinstance(file_list, str)):
            file_list = [file_list]

        list_of_feats = []
        for input_file in file_list:
            list_of_feats.append(features_for_pred(input_file=input_file))

        list_of_feats = pd.DataFrame(list_of_feats, columns=self.feature_list[:-1])
        data_modif = normalise_features(list_of_feats)
        input_feat = data_modif.iloc[:, :].values
        pred_strokes = self.model.predict(input_feat)
        return {
            x: y
            for x, y in zip(file_list, [self.dict_strokes()[x] for x in pred_strokes])
        }
