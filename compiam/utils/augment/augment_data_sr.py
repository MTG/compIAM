import os
import sys
import numpy as np
import librosa
import soundfile as sf
from joblib import Parallel, delayed
from pathlib import PurePath
import argparse

from compiam.utils.NMFtoolbox.forwardSTFT import forwardSTFT
from compiam.utils.NMFtoolbox.inverseSTFT import inverseSTFT
from compiam.utils.NMFtoolbox.initTemplates import initTemplates
from compiam.utils.NMFtoolbox.initActivations import initActivations
from compiam.utils.NMFtoolbox.NMF import NMF
from compiam.utils.NMFtoolbox.alphaWienerFilter import alphaWienerFilter
from compiam.utils.NMFtoolbox.utils import make_monaural, pcmInt16ToFloat32Numpy, EPS


def tuple_list(s):
    try:
        x, y, z = map(float, s.split(","))
        return x, y, z
    except:
        raise argparse.ArgumentTypeError("params must be x, y, z")


def apply_nmf_separation(x, fs, hopSize, nFFT, W):
    # spectral parameters
    paramSTFT = dict()
    paramSTFT["blockSize"] = nFFT
    paramSTFT["hopSize"] = hopSize
    paramSTFT["winFunc"] = np.hanning(paramSTFT["blockSize"])
    paramSTFT["reconstMirror"] = True
    paramSTFT["appendFrame"] = True
    paramSTFT["numSamples"] = len(x)

    # STFT computation
    # X, A, P = forwardSTFT(x, paramSTFT)
    X = librosa.stft(
        x,
        n_fft=nFFT,
        hop_length=hopSize,
        win_length=paramSTFT["blockSize"],
        window="hann",
        center=True,
        pad_mode="reflect",
    )

    # compute derived matrices
    # get magnitude
    A = np.abs(X)

    # get phase
    P = np.angle(X)

    # get dimensions and time and freq resolutions
    numBins, numFrames = X.shape
    deltaT = paramSTFT["hopSize"] / fs
    deltaF = fs / paramSTFT["blockSize"]

    # set common parameters
    numComp = W.shape[1]
    numIter = 30
    numTemplateFrames = 1

    # generate initial guess for templates
    paramTemplates = dict()
    paramTemplates["deltaF"] = deltaF
    paramTemplates["numComp"] = numComp
    paramTemplates["numBins"] = numBins
    paramTemplates["numTemplateFrames"] = numTemplateFrames
    initW = W

    for k in range(paramTemplates["numComp"]):
        initW[:, k] /= EPS + initW[:, k].max()

    # generate initial activations
    paramActivations = dict()
    paramActivations["numComp"] = numComp
    paramActivations["numFrames"] = numFrames
    initH = initActivations(paramActivations, "uniform")

    # NMF parameters
    paramNMF = dict()
    paramNMF["numComp"] = numComp
    paramNMF["numFrames"] = numFrames
    paramNMF["numIter"] = numIter
    paramNMF["numTemplateFrames"] = numTemplateFrames
    paramNMF["initW"] = initW
    paramNMF["initH"] = initH
    paramNMF["fixW"] = True

    # NMF core method
    NMFW, NMFH, NMFV = NMF(A, paramNMF)

    # alpha-Wiener filtering
    NMFA, _ = alphaWienerFilter(A, NMFV, 1.0)

    # resynthesize results of NMF with soft constraints and score information
    audios = []
    for k in range(numComp):
        Y = NMFA[k] * np.exp(1j * P)
        y, _ = inverseSTFT(Y, paramSTFT)
        audios.append(y)

    return audios


def augment_data_stroke_remix(
    audio_path, out_dir, fs, gain_factors, nmf_templates_path, winDur, hopDur
):
    # analysis parameters
    winSize = int(np.ceil(winDur * fs))
    hopSize = int(np.ceil(hopDur * fs))
    nFFT = 2 ** (int(np.ceil(np.log2(winSize))) + 1)

    # load audio
    x, fs = librosa.load(audio_path, sr=fs)
    x /= np.max(np.abs(x))

    # apply nmf
    nmf_templates = np.load(nmf_templates_path, allow_pickle=True).item()
    nmf_templates = np.array(
        [item for key in ["rb", "rt", "d"] for item in nmf_templates[key].T]
    ).T

    components = apply_nmf_separation(x, fs, hopSize, nFFT, nmf_templates)

    comp_rb = components[0] + components[1]
    comp_rt = components[2] + components[3]
    comp_d = components[4] + components[5]

    x_resyn = (
        gain_factors[0] * comp_rb + gain_factors[1] * comp_rt + gain_factors[2] * comp_d
    )
    x_resyn /= np.max(np.abs(x_resyn))

    audio_save_name = PurePath(audio_path).name.replace(
        ".wav", "_sr_bass_%2.2f_treble_%2.2f_damp_%2.2f.wav" % tuple(gain_factors)
    )

    sf.write(os.path.join(out_dir, audio_save_name), x_resyn, fs)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="attack-remix")
    parser.add_argument("--input", type=str, default="", help="path to input audio")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("..", "outputs"),
        help="folder to save modified audios",
    )
    parser.add_argument("--fs", type=int, default=16000, help="sampling rate")
    parser.add_argument(
        "--win_dur", type=float, default=46.4, help="window size in milliseconds"
    )
    parser.add_argument(
        "--hop_dur", type=str, default=5e-3, help="hop size in milliseconds"
    )
    parser.add_argument(
        "--params",
        nargs="+",
        type=tuple_list,
        default=[(0.6, 2, 0.2), (0.6, 2, 3), (2, 0.5, 0.2), (2, 0.5, 3)],
        help="list of 3-tuples with gain factors for remixing. Tuple entries correspond to each of bass, treble, & damped components.",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default=os.path.join(".", "nmf_templates.npy"),
        help="path to saved nmf templates",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=4, help="number of cores to run program on"
    )

    args, _ = parser.parse_known_args()

    winDur = args.win_dur * 1e-3
    hopDur = args.hop_dur * 1e-3

    _ = Parallel(n_jobs=args.n_jobs)(
        delayed(augment_data_stroke_remix)(
            args.input, args.output, args.fs, gain_set, args.templates, winDur, hopDur
        )
        for gain_set in args.params
    )
