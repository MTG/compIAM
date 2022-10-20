


import os
import math
import librosa

import numpy as np

from compiam.utils.pitch import pitch_normalisation
from compiam.melody.ftanet_carnatic.pitch_processing import batchize_test, get_est_arr
from compiam.melody.ftanet_carnatic.cfp import cfp_process

try:
    from tensorflow.keras import backend as K
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Lambda, \
    GlobalAveragePooling2D, MaxPooling2D, Concatenate, Add, Multiply, \
        Softmax, Reshape, UpSampling2D, Conv1D
except:
    raise ImportError(
        "In order to use this tool you need to have tensorflow installed. "
        "Please reinstall compiam using `pip install 'compiam[tensorflow]'"
    )


class DEEPSRGM(object):
    """DEEPSRGM model for raga classification
    """
    def __init__(self, model_path):
        """DEEPSRGM init method.

        :param model_path: path to file to the model weights.
        """
        if not os.path.exists(filepath + '.data-00000-of-00001'):
            raise ValueError("""
                Given path to model weights not found. Make sure you enter the path correctly.
                A training process for the FTA-Net tuned to Carnatic is under development right
                now and will be added to the library soon. Meanwhile, we provide the weights in the
                latest repository version (https://github.com/MTG/compIAM) so make sure you have these
                available before loading the Carnatic FTA-Net.
            """)
        self.model_path = model_path
        self.model = self.load_model()
        self.model.load_weights(filepath).expect_partial()


    def get_features(file, k=5):
        pitch = open(file+".pitch").read().strip().split("\n")
        pitch = np.array(list(map(lambda x: eval(x.split("\t")[-1]), pitch)))
        tonic = eval(open(file+".tonicFine").read().strip())
        feature = np.round(1200*np.log2(pitch/tonic)*(k/100)).clip(0)
        N = 200
        a=[]
        for i in range(N):
            c = np.random.randint(0, len(feature)-5000)
            a.append(feature[c:c+5000])
        return np.array(a)

    def get_class_data(cls):
        path = os.path.join(PATH, cls)
        files = [str(file)[:-6] for file in Path(path).glob('**/*.pitch')]
        X = np.empty((0, 5000))
        for file in files:
            X_new = get_feature(file)
            X = np.concatenate((X,X_new))
        return X

