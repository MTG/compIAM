import librosa
import matplotlib.pyplot as plt
import numpy as np

def plot_waveform(path_to_audio, t1, t2, labels=None, sr=44100, filepath=None):
    # Get data
    y,_ = librosa.load(path_to_audio, sr=sr)
    y1 = t1*sr
    y2 = t2*sr
    y = y[y1:y2]
    max_y = max(y)
    min_y = min(y)
    t = np.linspace(t1, t2, len(y))

    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(t, y)
    plt.ylabel('Signal Value')
    plt.xlabel('Time (s)')
    plt.ylim((max_y+max_y*0.02, min_y+min_y*0.02))

    if labels is not None:
        for o,l in labels.items():
            if t1<=o<=t2:
                plt.axvline(o)
                plt.text(o, max_y+10, l, color='red')

    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()
