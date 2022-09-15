import os
import math

import torch
import numpy as np
import essentia.standard as estd

from compiam.rhythm.tabla_transcription.models import onsetCNN_D, onsetCNN_RT, onsetCNN, gen_melgrams, peakPicker
from compiam.melody.ftanet_carnatic.model import FTANetCarnatic

from compiam.melody.ftanet_carnatic.pitch_processing import batchize_test, get_est_arr
from compiam.melody.ftanet_carnatic.cfp import cfp_process

class fourWayTabla:
    """ TODO
    """
    def __init__(self, filepath, n_folds=3, seq_length=15, hop_dur=10e-3):
        self.filepath = filepath
        self.categories = ['D', 'RT', 'RB', 'B']
        self.model_names = {'D': onsetCNN_D(), 'RT': onsetCNN_RT(), 'RB': onsetCNN(), 'B': onsetCNN()}
        self.n_folds = n_folds
        self.seq_length = seq_length
        self.hop_dur = hop_dur

    def predict(self, path_to_audio, predict_thresh=0.3, device=None):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        #get log-mel-spectrogram of audio
        stats_path = os.path.join(self.filepath, 'means_stds.npy')
        stats = np.load(stats_path)
        melgrams = gen_melgrams(path_to_audio, stats=stats)

        #get frame-wise onset predictions
        n_frames = melgrams.shape[-1]-self.seq_length
        odf = dict(zip(self.categories, [np.zeros(n_frames)]*4))

        for i_frame in np.arange(0, n_frames):
            x = torch.tensor(melgrams[:,:,i_frame:i_frame + self.seq_length]).double().to(device)
            x = x.unsqueeze(0)

            for cat in self.categories:
                y=0
                for fold in range(self.n_folds):
                    saved_model_path = os.path.join(self.filepath, cat, 'saved_model_%d.pt'%fold)
                    model = self.model_names[cat].double().to(device)
                    model.load_state_dict(torch.load(saved_model_path, map_location=device))
                    model.eval()

                    y += model(x).squeeze().cpu().detach().numpy()
                odf[cat][i_frame] = y/self.n_folds

        #pick peaks in predicted activations
        odf_peaks = dict(zip(self.categories, []*4))
        for cat in self.categories:
            odf_peaks[cat] = peakPicker(odf[cat], predict_thresh)

        onsets = np.concatenate([odf_peaks[cat] for cat in odf_peaks])
        onsets = np.array(onsets*self.hop_dur, dtype=float)
        labels = np.concatenate([[cat]*len(odf_peaks[cat]) for cat in odf_peaks])

        sorted_order = onsets.argsort()
        onsets = onsets[sorted_order]
        labels = labels[sorted_order]

        return onsets, labels


class ftanetCarnatic:
    """ TODO
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.model = FTANetCarnatic().create_ftanet()
        self.model.load_weights(self.filepath)

    def predict(self, path_to_audio, sample_rate=8000, hop_size=80, batch_size=5):
        pitch_track = self.model.extract(path_to_audio, sample_rate=sample_rate, hop_size=hop_size, batch_size=batch_size)
        return pitch_track

    def load_weights(self, model_path):
        """TODO
        Args:
            TODO
        Returns:
            TODO
        """
        self.ftanet.load_weights(filepath=model_path).expect_partial()

    def predict(self, path_to_audio, sample_rate=8000, hop_size=80, batch_size=5):
        """Extract melody from filename
        Args:
            filename (str): path to file to extract
        """
        xlist = []
        timestamps = []
        print('CFP process in {}'.format(path_to_audio))
        y, _ = estd.MonoLoader(path_to_audio, sampleRate=8000)()
        audio_len = len(y)
        batch_min = 8000*60*batch_size
        freqs = []
        if len(y) > batch_min:
            iters = math.ceil(len(y)/batch_min)
            for i in np.arange(iters):
                if i < iters-1:
                    audio_in = y[batch_min*i:batch_min*(i+1)]
                if i == iters-1:
                    audio_in = y[batch_min*i:]
                feature, _, time_arr = cfp_process(audio_in, sr=sample_rate, hop=hop_size)
                data = batchize_test(feature, size=128)
                xlist.append(data)
                timestamps.append(time_arr)

                estimation = get_est_arr(self.ftanet, xlist, timestamps, batch_size=16)
                if i == 0:
                    freqs = estimation[:, 1]
                else:
                    freqs = np.concatenate((freqs, estimation[:, 1]))
        else:
            feature, _, time_arr = cfp_process(y, sr=sample_rate, hop=hop_size)
            data = batchize_test(feature, size=128)
            xlist.append(data)
            timestamps.append(time_arr)
            # Getting estimatted pitch
            estimation = get_est_arr(self.ftanet, xlist, timestamps, batch_size=16)
            freqs = estimation[:, 1]
        TStamps = np.linspace(0, audio_len/sample_rate, len(freqs))

        ### TODO: Write code to re-sample in case sampling frequency is initialized different than 8k
        return np.array([TStamps, freqs]).transpose().toList()

    def normalise_pitch(self, pitch, tonic, bins_per_octave=120, max_value=4):
        """Normalize pitch given a tonic
        Args:
            pitch (list): list of pitch values and time-stamps
            tonic (float): TODO
            bins_per_octave (int): cents per bin
            max_value (int): TODO
        """
        pitch_values = pitch[:, 1]
        eps = np.finfo(np.float).eps
        normalised_pitch = bins_per_octave * np.log2(2.0 * (pitch_values + eps) / tonic)
        indexes = np.where(normalised_pitch <= 0)
        normalised_pitch[indexes] = 0
        indexes = np.where(normalised_pitch > max_value)
        normalised_pitch[indexes] = max_value
        return np.array([pitch[:, 0], normalised_pitch]).transpose().toList()
