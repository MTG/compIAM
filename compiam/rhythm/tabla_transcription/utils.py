import os
import librosa

import h5py as h5
import numpy as np

try:
    import torch
    from torch.utils import data
except:
    raise ImportError(
        "In order to use this tool you need to have torch installed. "
        "Please reinstall compiam using `pip install compiam[torch]`"
    )

# Dataloader class
class TablaDataset(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, filepath, seq_length=15, n_channels=3, mel_data=None):
        "Initialization"
        self.filepath = filepath
        self.seq_length = seq_length
        self.n_channels = n_channels
        self.hf = h5.File(self.filepath, "r")
        self.labels_weights = self.hf["data"][:]
        self.X_all = mel_data

    def __len__(self):
        "Denotes the total number of samples"
        return int(np.floor(self.labels_weights.shape[0]))

    def __getitem__(self, index):
        "Generates one sample of data"
        # Load data and get label
        row = self.labels_weights[index]
        start = int(row[1])
        song = row[0].decode() + ".hdf5"

        X = torch.tensor(
            self.X_all[song][:, :, start : start + self.seq_length], dtype=torch.float32
        )
        y = torch.tensor(float(row[2]), dtype=torch.float32)
        w = torch.tensor(float(row[3]), dtype=torch.float32)

        return X, y, w


# separate frame-wise labels and weights into train and val splits based on the CV fold
def make_train_val_split(
    folds,
    labels_weights_orig_filepath,
    labels_weights_aug_filepath,
    train_val_data_filepaths,
):
    for key in train_val_data_filepaths:
        if os.path.exists(train_val_data_filepaths[key]):
            os.system(f"rm {train_val_data_filepaths[key]}")

    with h5.File(labels_weights_orig_filepath, "r") as hf_data:
        with h5.File(train_val_data_filepaths["validation"], "w") as hf_fold:
            hf_fold.create_dataset("data", data=hf_data[f"fold{folds['val']}/data"])

    with h5.File(labels_weights_aug_filepath, "r") as hf_data:
        with h5.File(train_val_data_filepaths["train"], "a") as hf_fold:
            for fold in folds["train"]:
                if "data" not in hf_fold:
                    hf_fold.create_dataset(
                        "data", data=hf_data[f"fold{fold}/data"], maxshape=(None, None)
                    )
                else:
                    hf_fold["data"].resize(
                        (
                            hf_fold["data"].shape[0]
                            + hf_data[f"fold{fold}/data"].shape[0]
                        ),
                        axis=0,
                    )
                    hf_fold["data"][-hf_data[f"fold{fold}/data"].shape[0] :] = hf_data[
                        f"fold{fold}/data"
                    ]
    return


# pick peaks in activation signal
def peakPicker(data, peakThresh):
    peaks = np.array([], dtype="int")
    for ind in range(1, len(data) - 1):
        if (data[ind + 1] < data[ind] > data[ind - 1]) & (data[ind] > peakThresh):
            peaks = np.append(peaks, ind)
    return peaks


# generate log-mel-spectrograms given path to audio
def gen_melgrams(path_to_audio, stats):
    # analysis parameters
    fs = 16000
    hopDur = 10e-3
    hopSize = int(np.ceil(hopDur * fs))
    winDur_list = [23.2e-3, 46.4e-3, 92.8e-3]
    winSize_list = [int(np.ceil(winDur * fs)) for winDur in winDur_list]
    nFFT_list = [2 ** (int(np.ceil(np.log2(winSize)))) for winSize in winSize_list]
    fMin = 27.5
    fMax = 8000
    nMels = 80

    # context parameters
    contextlen = 7  # +- frames
    duration = 2 * contextlen + 1

    # data stats for normalization
    means = stats[0]
    stds = stats[1]

    x, fs = librosa.load(path_to_audio, sr=fs)

    # get mel spectrograms
    melgram1 = librosa.feature.melspectrogram(
        x,
        sr=fs,
        n_fft=nFFT_list[0],
        win_length=winSize_list[0],
        hop_length=hopSize,
        n_mels=nMels,
        fmin=fMin,
        fmax=fMax,
    )
    melgram2 = librosa.feature.melspectrogram(
        x,
        sr=fs,
        n_fft=nFFT_list[1],
        win_length=winSize_list[1],
        hop_length=hopSize,
        n_mels=nMels,
        fmin=fMin,
        fmax=fMax,
    )
    melgram3 = librosa.feature.melspectrogram(
        x,
        sr=fs,
        n_fft=nFFT_list[2],
        win_length=winSize_list[2],
        hop_length=hopSize,
        n_mels=nMels,
        fmin=fMin,
        fmax=fMax,
    )

    melgrams = np.array([melgram1, melgram2, melgram3])

    # log scaling
    melgrams = 10 * np.log10(1e-10 + melgrams)

    # normalize
    melgrams = (
        melgrams - np.repeat(np.atleast_3d(means), melgrams.shape[2], axis=-1)
    ) / np.repeat(np.atleast_3d(stds), melgrams.shape[2], axis=-1)

    # zero pad ends
    melgrams = np.concatenate(
        (
            np.zeros([melgrams.shape[0], melgrams.shape[1], contextlen]),
            melgrams,
            np.zeros([melgrams.shape[0], melgrams.shape[1], contextlen]),
        ),
        -1,
    )

    return melgrams


# separate filenames from songlist into train and val splits based on the CV fold
def get_train_val_split_filenames(
    train_val_folds, cv_split_filenames, songlist_orig, songlist_aug
):
    train_val_filenames = []

    # val files
    train_val_filenames += list(cv_split_filenames[train_val_folds["val"]])

    # train files
    for fold in train_val_folds["train"]:
        for song in cv_split_filenames[fold]:
            train_val_filenames += [item for item in songlist_aug if song in item]

    return train_val_filenames


# load all melgram data into memory in the form of filepath-melgram (key-value) pairs of a dict
def load_mel_data(datapath, folds, splits, songlist_orig, songlist_aug):
    train_val_filenames = get_train_val_split_filenames(
        folds, splits, songlist_orig, songlist_aug
    )
    mel_data_keys = []
    mel_data_vals = []
    for item in train_val_filenames:
        item_path = os.path.join(datapath, item + ".hdf5")
        mel_data_keys.append(item_path)
        with h5.File(item_path, "r") as hf:
            mel_data_vals.append(hf["data"][:])

    mel_data = dict(zip(mel_data_keys, mel_data_vals))
    return mel_data
