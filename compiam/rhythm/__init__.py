import os

import numpy as np
import torch

from compiam.rhythm.tabla_transcription.models import onsetCNN_D, onsetCNN_RT, onsetCNN, gen_melgrams, peakPicker
from compiam.exceptions import ModelNotTrainedError
from compiam.utils import get_logger

logger = get_logger(__name__)


class fourWayTabla:

    def __init__(self, filepath=None, n_folds=3, seq_length=15, hop_dur=10e-3, device=None):
        
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.filepath = filepath
        if self.filepath:
            self.load_model(filepath, self.device)

        self.categories = ['D', 'RT', 'RB', 'B']
        self.model_names = {'D': onsetCNN_D(), 'RT': onsetCNN_RT(), 'RB': onsetCNN(), 'B': onsetCNN()}
        self.models = {}
        self.stats = None
        self.n_folds = n_folds
        self.seq_length = seq_length
        self.hop_dur = hop_dur
        
        # Load model if passed
        self.filepath = filepath
        if self.filepath:
            self.load_model(filepath)

    def load_model(self, filepath):
        stats_path = os.path.join(filepath, 'means_stds.npy')
        self.stats = np.load(stats_path)
        for cat in self.categories:
            self.models[cat] = {}
            y=0
            for fold in range(self.n_folds):
                saved_model_path = os.path.join(filepath, cat, 'saved_model_%d.pt'%fold)
                model = self.model_names[cat].double().to(self.device)
                model.load_state_dict(torch.load(saved_model_path, map_location=self.device))
                model.eval()

                self.models[cat][fold] = model

    def train(self):
        if self.models:
            logger.warning("Model is already initalised, overwriting with new data")
        pass

    def predict(self, path_to_audio, predict_thresh=0.3):
        if not self.models:
            raise ModelNotTrainedError('Please load or train model before predicting')

        #get log-mel-spectrogram of audio
        melgrams = gen_melgrams(path_to_audio, stats=self.stats)

        #get frame-wise onset predictions
        n_frames = melgrams.shape[-1]-self.seq_length
        odf = dict(zip(self.categories, [np.zeros(n_frames)]*4))

        for i_frame in np.arange(0, n_frames):
            x = torch.tensor(melgrams[:,:,i_frame:i_frame + self.seq_length]).double().to(self.device)
            x = x.unsqueeze(0)

            for cat in self.categories:
                y=0
                for fold in range(self.n_folds):
                    model = self.models[cat][fold]

                    y += model(x).squeeze().cpu().detach().numpy()
                odf[cat][i_frame] = y/self.n_folds

        # pick peaks in predicted activations
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
  