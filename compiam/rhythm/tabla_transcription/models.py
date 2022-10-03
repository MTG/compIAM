import os
import librosa

import numpy as np

try:
    import torch
    import torch.nn as nn
except:
    raise ImportError(
        "In order to use this tool you need to have torch installed. "
        "Please reinstall compiam using `pip install compiam[torch]`"
    )

#from compiam.utils.core import get_logger
#logger = get_logger(__name__)

# model definition for resonant bass and resonant both categories
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
def gen_melgrams(path_to_audio, stats):
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


def check_cuda(device):
    if device:
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return "cpu"
     

def get_odf(models, melgrams, seq_length, categories, n_folds, device="cuda"):
    n_frames = melgrams.shape[-1]-seq_length
    odf = dict(zip(categories, [np.zeros(n_frames)]*4))

    for i_frame in np.arange(0, n_frames):
        x = torch.tensor(melgrams[:,:,i_frame:i_frame + seq_length]).double().to(device)
        x = x.unsqueeze(0)

        for cat in categories:
            y=0
            for fold in range(n_folds):
                model = models[cat][fold]

                y += model(x).squeeze().cpu().detach().numpy()
            odf[cat][i_frame] = y/n_folds


def load_models(filepath, model_names, categories, n_folds, device="cuda"):
    """TODO
    """
    models = {}
    stats_path = os.path.join(filepath, 'means_stds.npy')
    stats = np.load(stats_path)
    for cat in categories:
        models[cat] = {}
        y=0
        for fold in range(n_folds):
            saved_model_path = os.path.join(filepath, cat, 'saved_model_%d.pt'%fold)
            model = model_names[cat].double().to(device)
            model.load_state_dict(torch.load(saved_model_path, map_location=device))
            model.eval()

            models[cat][fold] = model
    return models, stats
