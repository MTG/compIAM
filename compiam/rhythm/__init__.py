import os

try:
    import torch
except:
    raise ImportError(
        "In order to use this tool you need to have torch installed. "
        "Please reinstall compiam using `pip install compiam[torch]`"
    )

import numpy as np

from compiam.exceptions import ModelNotTrainedError
from compiam.utils import get_logger

logger = get_logger(__name__)

from compiam.rhythm.tabla_transcription.models import onsetCNN_D, onsetCNN_RT, onsetCNN
from compiam.rhythm.tabla_transcription.models import gen_melgrams, peakPicker

class FourWayTabla:
    """TODO
    """
    def __init__(self, filepath=None, n_folds=3, seq_length=15, hop_dur=10e-3, device=None):
        """TODO
        """
        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
            self.load_models(filepath)

    def load_models(self, filepath):
        """TODO
        """
        stats_path = os.path.join(filepath, 'means_stds.npy')
        self.stats = np.load(stats_path)
        for cat in self.categories:
            self.models[cat] = {}
            for fold in range(self.n_folds):
                saved_model_path = os.path.join(filepath, cat, 'saved_model_%d.pt'%fold)
                model = self.model_names[cat].double().to(self.device)
                model.load_state_dict(torch.load(saved_model_path, map_location=self.device))
                model.eval()

                self.models[cat][fold] = model

#    def train(self, model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-4), criterion=torch.nn.BCELoss(reduction='none'), training_generator):
#        """TODO
#        """
#        if self.models:
#            logger.warning("Model is already initalised, overwriting with new data")

#        model.train()
#        n_batch=0
#        loss_epoch=0
#        for local_batch, local_labels, local_weights in training_generator:
#            n_batch+=1

#            #transfer to GPU
#            local_batch, local_labels, local_weights = local_batch.double().to(device), local_labels.double().to(device), local_weights.double().to(device)

#            #model forward pass
#            optimizer.zero_grad()
#            outs = model(local_batch).squeeze()
#            outs = outs.double()

#            #compute loss
#            loss = criterion(outs, local_labels)
#            loss = loss.double()
#            loss = torch.dot(loss,local_weights)
#            loss /= local_batch.size()[0]
#            loss_epoch+=loss.item()

#            #update weights
#            loss.backward()
#            optimizer.step()
#        return model, loss_epoch/n_batch

    def predict(self, path_to_audio, predict_thresh=0.3):
        """TODO
        """
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
  