import numpy as np

from compiam.exceptions import ModelNotTrainedError
from compiam.utils import get_logger

logger = get_logger(__name__)


class fourWayTabla:
    """TODO
    """
    def __init__(self, filepath=None, n_folds=3, seq_length=15, hop_dur=10e-3, device=None):
        """TODO
        """
        from compiam.rhythm.tabla_transcription.models import onsetCNN_D, onsetCNN_RT, onsetCNN
        from compiam.rhythm.tabla_transcription.models import load_models, check_cuda
        if not device:
            self.device = check_cuda(device)

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
            self.models, self.stats = load_models(
                filepath, self.model_names, self.categories, self.n_folds, self.device)

    def train(self):
        """TODO
        """
        if self.models:
            logger.warning("Model is already initalised, overwriting with new data")
        pass

    def predict(self, path_to_audio, predict_thresh=0.3):
        """TODO
        """
        from compiam.rhythm.tabla_transcription.models import gen_melgrams, peakPicker, get_odf
        if not self.models:
            raise ModelNotTrainedError('Please load or train model before predicting')

        #get log-mel-spectrogram of audio
        melgrams = gen_melgrams(path_to_audio, stats=self.stats)

        #get frame-wise onset predictions
        odf = get_odf(self.models, melgrams, self.seq_length, self.categories, self.n_folds, self.device)

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
  