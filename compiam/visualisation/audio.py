import librosa
import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(path_to_audio, t1, t2, labels=None, sr=44100):
	# Get data
	y,_ = librosa.load(path_to_audio, sr=sr)
	y1 = t1*sr
	y2 = t2*sr
	y = y[y1:y2]
	max_y = max(y)
	t = np.linspace(y1, y2, len(y))

	# Plot
	plt.figure(figsize=(15, 5))
	plt.plot(y, t)
	plt.ylabel('Signal Value')
	plt.xlabel('Time (s)')

	if labels:
		for k,v in labels.items():
			if t1<k<t2:
				k_ = k*sr
				plt.axvline(k_)
				plt.text(k_, max_y+10, v)
	plt.show()