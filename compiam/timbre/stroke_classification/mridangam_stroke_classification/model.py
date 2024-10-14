import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from compiam.timbre.stroke_classification.mridangam_stroke_classification import (
    normalise_features,
)
from compiam.utils import get_logger

logger = get_logger(__name__)


class StrokeClassification:
    """Mridangam stroke classification."""

    def __init__(self):
        """Mridangam stroke classification init method."""

    def train(
        self,
        training_data,
        feature_list,
        model_type="svm",
        balance=False,
        balance_ref="random",
    ):
        """Train a support vector machine for stroke classification.

        :param training_data: DataFrame including features to train.
        :param feature_list: list of features considered for training.
        :param model_type: type of model to train.
        :param balance: balance the number of instances per class to prevent biases.
        :param balance_ref: reference class for data balancement.
        :returns: a trained scikit learn classificator object.
        """

        if training_data is None:
            raise ValueError(
                "Prior to train the model please load the dataset using .process_strokes()"
            )

        # Let's use sklearn's preprocessing tools for applying normalisation to features
        data_modif = normalise_features(training_data, feature_list)

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

        X = data_modif.iloc[:, : len(feature_list) - 1].values
        # Creating output values
        data_modif.stroke = pd.Categorical(
            data_modif.stroke
        )  # convert to categorical data
        y = np.array(data_modif.stroke.cat.codes)  # create label encoded outputs

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        if model_type == "svm":
            clf = svm.SVC(gamma=1 / (X_train.shape[-1] * X_train.var()))
        elif model_type == "mlp":
            clf = MLPClassifier(alpha=1, max_iter=1000)
        else:
            raise ValueError(
                "Model not available. Please check the available options in the documentation."
            )

        # Fit model with training data
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        logger.info(
            "{} model successfully trained with accuracy {}% in the testing set".format(
                model_type.upper(),
                round(np.sum(y_test == y_pred) / len(y_pred) * 100),
                2,
            )
        )
        return clf
