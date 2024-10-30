import os
import librosa

import numpy as np
import matplotlib.pyplot as plt

from compiam.utils import get_logger

logger = get_logger(__name__)


def plot_waveform(
    input_data,
    t1,
    t2,
    labels=None,
    input_sr=44100,
    sr=44100,
    output_path=None,
    verbose=False,
):
    """Plotting waveform between two given points with optional labels

    :param input_data: path to audio file or numpy array like audio signal
    :param input_sr: sampling rate of the input array of data (if any). This variable is only
        relevant if the input is an array of data instead of a filepath.
    :param t1: starting point for plotting
    :param t2: ending point for plotting
    :param labels: dictionary {time_stamp:label} to plot on top of waveform
    :param sr: sampling rate
    :param output_path: optional path (finished with .png) where the plot is saved
    """
    if verbose is False:
        logger.setLevel("ERROR")
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise FileNotFoundError("Target audio not found.")
        audio, _ = librosa.load(input_data, sr=sr)
    elif isinstance(input_data, np.ndarray):
        logger.warning(
            f"Resampling... (input sampling rate is {input_sr}Hz, make sure this is correct)"
        )
        audio = librosa.resample(input_data, orig_sr=input_sr, target_sr=sr)
    else:
        raise ValueError("Input must be path to audio signal or an audio array")

    y1 = t1 * sr
    y2 = t2 * sr
    audio = audio[y1:y2]
    max_y = max(audio)
    min_y = min(audio)
    t = np.linspace(t1, t2, len(audio))

    # Plot
    plt.figure(figsize=(20, 5))
    fig, ax = plt.subplots()

    ax.set_facecolor("#dbdbdb")
    plt.plot(t, audio, color="darkgreen")
    plt.ylabel("Signal Value")
    plt.xlabel("Time (s)")
    plt.ylim((min_y - min_y * 0.1, max_y + max_y * 0.1))

    if labels is not None:
        for o, l in labels.items():
            if t1 <= o <= t2:
                logger.info(f"{o}:{l}")
                plt.axvline(o, color="firebrick", linestyle="--")
                plt.text(o, max_y + max_y * 0.11, l, color="firebrick")

    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
