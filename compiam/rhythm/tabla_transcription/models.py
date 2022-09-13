import os

import librosa
import torch
import numpy as np
import torch.nn as nn
import torch

from compiam.exceptions import ModelNotFoundError
from compiam.utils.core import get_logger
from compiam.rhythm.tabla_transcription.__paths import models_paths_dict as paths_dict

logger = get_logger(__name__)

def load_model(name):
	if name not in path_dict:
		raise ModelNotFoundError(f"Model name not recognised, available models: {path_dict.values()}")

	model_dict = path_dict[name]
	path = model_dict['path']
	model = model_dict['model']

	_, file_extension = os.path.splitext(path)

	if file_extension == '.pt':\
		logger.info(f'Loading model, {name} at {path}...')
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = model.double().to(device)
		model.load_state_dict(torch.load(path, map_location=device))
		model.eval()

		return model


#model definition for resonant bass and resonant both categories
class onsetCNN(nn.Module):
	def __init__(self):
		super(onsetCNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, (3,7))
		self.bn1 = nn.BatchNorm2d(16)
		self.pool1 = nn.MaxPool2d((3,1))
		self.conv2 = nn.Conv2d(16, 32, 3)
		self.bn2 = nn.BatchNorm2d(32)
		self.pool2 = nn.MaxPool2d((3,1))
		self.bn3 = nn.BatchNorm1d(128)
		self.fc1 = nn.Linear(32 * 7 * 8, 128)
		self.bn4 = nn.BatchNorm1d(1)
		self.fc2 = nn.Linear(128,1)
		self.dout2 = nn.Dropout(p=0.25)

	def forward(self,x):
		y=torch.relu(self.bn1(self.conv1(x)))
		y=self.pool1(y)
		y=torch.relu(self.bn2(self.conv2(y)))
		y=self.pool2(y)
		y=self.dout2(y.view(-1,32*7*8))
		y=self.dout2(torch.relu(self.bn3(self.fc1(y))))
		y=torch.sigmoid(self.bn4(self.fc2(y)))
		return y


#model definition for damped category
class onsetCNN_D(nn.Module):
	def __init__(self):
		super(onsetCNN_D, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, (3,7))
		self.bn1 = nn.BatchNorm2d(16)
		self.pool1 = nn.MaxPool2d((3,1))
		self.conv2 = nn.Conv2d(16, 32, 3)
		self.bn2 = nn.BatchNorm2d(32)
		self.pool2 = nn.MaxPool2d((3,1))
		self.bn3 = nn.BatchNorm1d(256)
		self.fc1 = nn.Linear(32 * 7 * 8, 256)
		self.bn4 = nn.BatchNorm1d(1)
		self.fc2 = nn.Linear(256,1)
		self.dout2 = nn.Dropout(p=0.25)

	def forward(self,x):
		y=torch.relu(self.bn1(self.conv1(x)))
		y=self.pool1(y)
		y=torch.relu(self.bn2(self.conv2(y)))
		y=self.pool2(y)
		y=self.dout2(y.view(-1,32*7*8))
		y=self.dout2(torch.relu(self.bn3(self.fc1(y))))
		y=torch.sigmoid(self.bn4(self.fc2(y)))
		return y


#model definition for resonant treble category
class onsetCNN_RT(nn.Module):
	def __init__(self):
		super(onsetCNN_RT, self).__init__()
		self.conv1 = nn.Conv2d(3, 32, (3,7))
		self.bn1 = nn.BatchNorm2d(32)
		self.pool1 = nn.MaxPool2d((3,1))
		self.conv2 = nn.Conv2d(32, 64, 3)
		self.bn2 = nn.BatchNorm2d(64)
		self.pool2 = nn.MaxPool2d((3,1))
		self.bn3 = nn.BatchNorm1d(128)
		self.fc1 = nn.Linear(64 * 7 * 8, 128)
		self.bn4 = nn.BatchNorm1d(1)
		self.fc2 = nn.Linear(128,1)
		self.dout2 = nn.Dropout(p=0.25)

	def forward(self,x):
		y=torch.relu(self.bn1(self.conv1(x)))
		y=self.pool1(y)
		y=torch.relu(self.bn2(self.conv2(y)))
		y=self.pool2(y)
		y=self.dout2(y.view(-1,64*7*8))
		y=self.dout2(torch.relu(self.bn3(self.fc1(y))))
		y=torch.sigmoid(self.bn4(self.fc2(y)))
		return y


#pick peaks in activation signal
def peakPicker(data, peakThresh):
	peaks=np.array([],dtype='int')
	for ind in range(1,len(data)-1):
		if ((data[ind+1] < data[ind] > data[ind-1]) & (data[ind]>peakThresh)):
			peaks=np.append(peaks,ind)
	return peaks


#generate log-mel-spectrograms given path to audio
def gen_melgrams(path_to_audio):
	#analysis parameters
	fs=16000
	hopDur=10e-3
	hopSize = int(np.ceil(hopDur*fs))
	winDur_list = [23.2e-3, 46.4e-3, 92.8e-3]
	winSize_list = [int(np.ceil(winDur*fs)) for winDur in winDur_list]
	nFFT_list = [2**(int(np.ceil(np.log2(winSize)))) for winSize in winSize_list]
	fMin=27.5
	fMax=8000
	nMels=80

	#context parameters
	contextlen=7 #+- frames
	duration=2*contextlen+1

	#data stats for normalization
	stats=np.load('./means_stds.npy')
	means=stats[0]
	stds=stats[1]

	x,fs = librosa.load(path_to_audio, sr=fs)

	#get mel spectrograms
	melgram1=librosa.feature.melspectrogram(x,sr=fs,n_fft=nFFT_list[0], win_length=winSize_list[0], hop_length=hopSize, n_mels=nMels, fmin=fMin, fmax=fMax)
	melgram2=librosa.feature.melspectrogram(x,sr=fs,n_fft=nFFT_list[1], win_length=winSize_list[1], hop_length=hopSize, n_mels=nMels, fmin=fMin, fmax=fMax)
	melgram3=librosa.feature.melspectrogram(x,sr=fs,n_fft=nFFT_list[2], win_length=winSize_list[2], hop_length=hopSize, n_mels=nMels, fmin=fMin, fmax=fMax)

	melgrams = np.array([melgram1, melgram2, melgram3])

	#log scaling
	melgrams=10*np.log10(1e-10+melgrams)

	#normalize
	melgrams = (melgrams - np.repeat(np.atleast_3d(means), melgrams.shape[2], axis=-1))/np.repeat(np.atleast_3d(stds), melgrams.shape[2], axis=-1)

	#zero pad ends
	melgrams = np.concatenate((np.zeros([melgrams.shape[0], melgrams.shape[1], contextlen]), melgrams, np.zeros([melgrams.shape[0], melgrams.shape[1], contextlen])), -1)

	return melgrams
