import random

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neural_network import MLPClassifier

from .feature_extraction import normalize_features


class StrokeClassification:
    """Mridangam stroke classification 
    Attributes:
        load_dataset (True)
        data_home (str)
        version (str)
    """
    def __init__(self):
        """Mridangam stroke classification init method
        Args:
            data_home (str): folder where the dataset is found
            version (str): version of the dataset to use
        """

    def train(self, trainig_data, feature_list, model_type="svm", balance=False, balance_ref='random'):
        """Train a support vector machine for stroke classification
        Args:
            balance (bool): balance the number of instances per class to prevent biases
        """
        if trainig_data is None:
            raise ValueError("Prior to train the model please load the dataset using .process_strokes()") 

        #Let's use sklearn's preprocessing tools for applying normalisation to features
        data_modif = normalize_features(trainig_data, feature_list)
        
        if balance == True:
            strokes = trainig_data.stroke.unique()
            count_dict = trainig_data['stroke'].value_counts().to_dict()
            min_stroke = min(count_dict, key=count_dict.get)
            min_number = data_modif.stroke.value_counts()[min_stroke] if balance_ref == 'lower' \
                else data_modif.stroke.value_counts()[random.choice(strokes)]
            reshaped_stroke_list = []
            for strk in strokes:
                if count_dict[strk] > min_number:
                    reshaped_stroke_list.append(data_modif[data_modif.stroke == strk].sample(n = min_number))
                else:
                    reshaped_stroke_list.append(data_modif[data_modif.stroke == strk])
            #Merging after downsampling
            data_modif = pd.concat(reshaped_stroke_list)

        X = data_modif.iloc[:,:len(feature_list)-1].values 
        # Creating output values
        data_modif.stroke = pd.Categorical(data_modif.stroke)  # convert to categorical data
        y = np.array(data_modif.stroke.cat.codes)  # create label encoded outputs

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

        if model_type == "svm":
            clf = svm.SVC(gamma = 1 / (X_train.shape[-1] * X_train.var()))
        elif model_type == "mlp":
            clf = MLPClassifier(alpha=1, max_iter=1000)
        else:
            raise ValueError("Model not available. Please check the available options in the documentation.") 

        # Fit model with training data
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        print("{} model successfully trained with accuracy {}% in the testing set".format(
            model_type.upper(),
            round(np.sum(y_test == y_pred) / len(y_pred)*100), 2))

        return clf

    '''
    def train_nn_model(self, balance=False):
        """Train a support vector machine for stroke classification
        Args:
            balance (bool): balance the number of instances per class to prevent biases
        """
        if self.processed_stroke_dict is None:
            raise ValueError("Prior to train the model please load the dataset using .process_strokes()") 
        data_modif = self.processed_stroke_dict.copy()
        #Let's use sklearn's preprocessing tools for applying normalisation to features
        data_modif.iloc[:,:len(self.descriptors_to_train)-1] = MIX_MAX_SCALER.fit_transform(self.processed_stroke_dict.iloc[:,:len(self.descriptors_to_train)-1].values)
        if balance == True:
            min_number = data_modif.stroke.value_counts()['cha']
            thi_data = data_modif[data_modif.stroke == 'thi'].sample(n = min_number)
            tha_data = data_modif[data_modif.stroke == 'tha'].sample(n = min_number)
            ta_data = data_modif[data_modif.stroke == 'ta'].sample(n = min_number)
            thom_data = data_modif[data_modif.stroke == 'thom'].sample(n = min_number)
            num_data = data_modif[data_modif.stroke == 'num'].sample(n = min_number)
            dhin_data = data_modif[data_modif.stroke == 'dhin'].sample(n = min_number)
            dheem_data = data_modif[data_modif.stroke == 'dheem'].sample(n = min_number)
            tham_data = data_modif[data_modif.stroke == 'tham'].sample(n = min_number)
            cha_data = data_modif[data_modif.stroke == 'cha'].sample(n = min_number)
            bheem_data = data_modif[data_modif.stroke == 'bheem']
            #Merging after downsampling
            data_modif = pd.concat(
                [thi_data, tha_data, ta_data, thom_data, num_data, dhin_data, dheem_data, tham_data, cha_data, bheem_data]
            )

        X = data_modif.iloc[:,:len(self.descriptors_to_train)-1].values 
        # Creating output values
        data_modif.stroke = pd.Categorical(data_modif.stroke)  # convert to categorical data
        y = np.array(data_modif.stroke.cat.codes)  # create label encoded outputs

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

        self.clf = svm.SVC(gamma = 1 / (X_train.shape[-1] * X_train.var()))

        # Fit model with training data
        self.clf.fit(X_train, y_train)

        # Evaluate
        y_pred = self.clf.predict(X_test)
        print("Model successfully trained with accuracy {}% in the testing set".format(
            round(np.sum(y_test == y_pred) / len(y_pred)*100), 2))
        '''