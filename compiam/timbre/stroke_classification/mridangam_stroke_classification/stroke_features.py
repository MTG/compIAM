import os
import tqdm

import pandas as pd
import numpy as np

try:
    import essentia.standard as estd
except:
    raise ImportError(
        "In order to use this tool you need to have essentia installed. "
        "Install compIAM with essentia support: pip install 'compiam[essentia]'"
    )


SPLIT_PARAMS = {
    "fs": 44100,
    "windowSize": 1024,
    "hopSize": 512,
    "NRG_threshold_ratio": 0.005,
}
DESCRIPTORS_TO_DISREGARD = ["sfx", "tristimulus", "sccoeffs"]

from sklearn import preprocessing

MIN_MAX_SCALER = preprocessing.MinMaxScaler()


def split_file(input_file):
    """Define split boundaries based on a fixed energy threshold.

    :param input_file: path to file to process.
    :returns: a tuple with input file, energy threshold, split function, and
        start and end indexes of the detected splits.
    """
    x = estd.MonoLoader(filename=input_file, sampleRate=SPLIT_PARAMS.get("fs"))()
    NRG = []
    # Main windowing and feature extraction loop
    for frame in estd.FrameGenerator(
        x,
        frameSize=SPLIT_PARAMS.get("windowSize"),
        hopSize=SPLIT_PARAMS.get("hopSize"),
        startFromZero=True,
    ):
        NRG.append(estd.Energy()(frame))
    NRG = np.array(NRG)
    NRG = NRG / np.max(NRG)

    # Applying energy threshold to decide wave split boundaries
    split_decision_func = np.zeros_like(NRG)
    split_decision_func[NRG > SPLIT_PARAMS.get("NRG_threshold_ratio")] = 1
    # Setting segment boundaries
    # Inserting a zero at the beginning since we will decide the transitions using a diff function
    split_decision_func = np.insert(split_decision_func, 0, 0)
    diff_split_decision = np.diff(split_decision_func)
    # Start indexes: transition from 0 to 1
    start_indexes = np.nonzero(diff_split_decision > 0)[0] * SPLIT_PARAMS.get("hopSize")
    # Stop indexes: transition from 1 to 0
    stop_indexes = np.nonzero(diff_split_decision < 0)[0] * SPLIT_PARAMS.get("hopSize")
    return (x, NRG, split_decision_func, start_indexes, stop_indexes)


def process_strokes(file_dict, load_computed=False, computed_path=None):
    """Process and extract features from stroke files.

    :param stroke_dict: dict of files per stroke class (preferably generated through a mirdata loader).
    :param load_computed: if True the pre-computed file is loaded.
    :returns: DataFrame with features per split, and list of computed features.
    """
    if not isinstance(load_computed, bool):
        raise ValueError("load_computed must be whether True or False")
    first_one = True
    columns = []
    list_of_feat = []
    if load_computed == False:
        for stroke, files in tqdm.tqdm(file_dict.items()):
            for sample_file in files:
                # Get file id
                (x, _, _, start_indexes, stop_indexes) = split_file(sample_file)
                for start, stop in zip(start_indexes, stop_indexes):
                    x_seg = x[start:stop]
                    # Final check for amplitude (to avoid silent segments selection due to noise in split function)
                    if np.max(np.abs(x_seg)) > 0.05:
                        # Amplitude normalisation
                        x_seg = x_seg / np.max(np.abs(x_seg))
                        # Compute and write features for file
                        features = estd.Extractor(
                            dynamics=False,
                            rhythm=False,
                            midLevel=False,
                            highLevel=False,
                        )(x_seg)
                        feat = []
                        # Get descriptor names
                        descriptors = features.descriptorNames()
                        # Remove unneeded descriptors
                        for desc in DESCRIPTORS_TO_DISREGARD:
                            descriptors = [x for x in descriptors if desc not in x]
                        # Process MFCC
                        for i in np.arange(np.shape(features["lowLevel.mfcc"])[1]):
                            if first_one:
                                columns.append("mfcc" + str(i) + ".mean")
                                columns.append("mfcc" + str(i) + ".dev")
                            feat.append(np.mean(features["lowLevel.mfcc"][:, i]))
                            feat.append(np.std(features["lowLevel.mfcc"][:, i]))
                        # Now remove already computed mfcc
                        descriptors = [x for x in descriptors if "mfcc" not in x]
                        for desc in descriptors:
                            if first_one:
                                columns.append(desc + ".mean")
                                columns.append(desc + ".dev")
                            feat.append(np.mean(features[desc]))
                            feat.append(np.std(features[desc]))
                        feat.append(stroke)
                        list_of_feat.append(feat)
                        if first_one:
                            columns = columns + ["stroke"]
                            feature_list = columns
                            first_one = False
        # Convert list of features to dict and write to file
        df_features = pd.DataFrame(list_of_feat, columns=columns)
        df_features.to_csv(computed_path, index=False)
    else:
        if not os.path.exists(computed_path):
            raise FileNotFoundError(
                "Please enter a valid path for the computed features .csv path"
            )
        # Load the pre-computed dict
        df_features = pd.read_csv(computed_path)
        feature_list = list(df_features.columns)
    return df_features, feature_list


def normalise_features(training_data, feature_list=None):
    """Normalise feature DataFrames.

    :param training_data: DataFrame with no-normalised features.
    :param feature_list: list of features to prevent including the stroke label if included in the list.
    :returns: DataFrame with normalised features per split.
    """
    data_modif = training_data.copy()
    if feature_list is None:
        data_modif.iloc[:, :] = MIN_MAX_SCALER.fit_transform(
            training_data.iloc[:, :].values
        )
    else:
        data_modif.iloc[:, : len(feature_list) - 1] = MIN_MAX_SCALER.fit_transform(
            training_data.iloc[:, : len(feature_list) - 1].values
        )
    return data_modif


def features_for_pred(input_file):
    """Compute and format features for prediction.

    :param input_file: path to file to extract the features from.
    :returns: DataFrame with normalised features per split.
    """
    (audio, _, _, start_indexes, stop_indexes) = split_file(input_file)
    if len(start_indexes) > 1:
        max_len = np.argmax(
            [np.abs(y - x) for x, y in zip(start_indexes, stop_indexes)]
        )
    else:
        max_len = 0
    features = estd.Extractor(
        dynamics=False, rhythm=False, midLevel=False, highLevel=False
    )(audio[start_indexes[max_len] : stop_indexes[max_len]])
    feat = []
    descriptors = features.descriptorNames()
    # Remove unneeded descriptors
    for desc in DESCRIPTORS_TO_DISREGARD:
        descriptors = [x for x in descriptors if desc not in x]
    # Process MFCC
    for i in np.arange(np.shape(features["lowLevel.mfcc"])[1]):
        feat.append(np.mean(features["lowLevel.mfcc"][:, i]))
        feat.append(np.std(features["lowLevel.mfcc"][:, i]))
    # Now remove already computed mfcc
    descriptors = [x for x in descriptors if "mfcc" not in x]
    for desc in descriptors:
        feat.append(np.mean(features[desc]))
        feat.append(np.std(features[desc]))
    return feat
