import os

import torch
import numpy as np

from compiam.rhythm.tabla_transcription.models import onsetCNN_D, onsetCNN_RT, onsetCNN, gen_melgrams, peakPicker

class fourWayTabla:

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