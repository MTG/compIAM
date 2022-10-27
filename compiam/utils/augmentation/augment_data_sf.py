import os
import numpy as np
import librosa
import soundfile as sf
import scipy
from joblib import Parallel, delayed
from pathlib import PurePath
import argparse


def tuple_list(s):
    try:
        x, y, z = map(float, s.split(","))
        return x, y, z
    except:
        raise argparse.ArgumentTypeError("Gains must be x,y,z")


def make_gain_curve(
    gain_curve_center,
    gain_curve_width,
    gain_factor,
    fs,
    nFFT,
    curve_func=scipy.signal.windows.hann,
):
    gain_curve = np.ones(nFFT // 2 + 1)
    if gain_factor == 1.0:
        return gain_curve

    gain_curve_center = int(np.floor(gain_curve_center * (nFFT // 2 + 1) / (fs / 2)))
    gain_curve_width = int(np.floor(gain_curve_width * (nFFT // 2 + 1) / (fs / 2)))
    low_end = np.max([0, gain_curve_center - gain_curve_width // 2])
    high_end = np.min([nFFT // 2 + 1, gain_curve_center + gain_curve_width // 2])

    try:
        gain_curve[low_end:high_end] += (gain_factor - 1) * curve_func(gain_curve_width)
    except ValueError:
        gain_curve[low_end : high_end + 1] += (gain_factor - 1) * curve_func(
            gain_curve_width
        )

    return gain_curve


def augment_data_spectral_shape(
    audio_path, out_dir, fs, gain_curve_params, winDur=46.4e-3, hopDur=5e-3
):
    # analysis parameters
    winSize = int(np.ceil(winDur * fs))
    hopSize = int(np.ceil(hopDur * fs))
    nFFT = 2 ** (int(np.ceil(np.log2(winSize))) + 1)

    # load audio
    x, fs = librosa.load(audio_path, sr=fs)
    x /= np.max(np.abs(x))

    # apply hps
    components = ["harm", "perc"]
    S = librosa.stft(
        x,
        n_fft=nFFT,
        hop_length=hopSize,
        win_length=winSize,
        window="hann",
        center=True,
        pad_mode="reflect",
    )
    S = dict(
        zip(
            components,
            librosa.decompose.hpss(
                S, kernel_size=31, power=1.0, mask=False, margin=1.0
            ),
        )
    )

    x = {}
    x_stft = {}
    x_resyn = []
    for comp in components:
        x[comp] = librosa.istft(
            S[comp], hop_length=hopSize, win_length=winSize, window="hann", center=True
        )
        x_stft[comp] = librosa.stft(
            x[comp],
            n_fft=nFFT,
            hop_length=hopSize,
            win_length=winSize,
            window="hann",
            center=True,
            pad_mode="reflect",
        )
    S = []

    gain_curves = {}
    for comp in components:
        # make gain_curve
        if comp == "perc":
            gain_curves[comp] = make_gain_curve(
                fs / 4, fs / 2, gain_curve_params[2], fs, nFFT
            )
        else:
            # bass
            gain_curves[comp] = make_gain_curve(
                100, 100, gain_curve_params[0], fs, nFFT
            )
            # treble
            gain_curves[comp] *= make_gain_curve(
                1100, 1800, gain_curve_params[1], fs, nFFT
            )

        # modify shape
        x_stft[comp] *= np.atleast_2d(gain_curves[comp]).T

        x[comp] = librosa.istft(
            x_stft[comp],
            hop_length=hopSize,
            win_length=None,
            window="hann",
            center=True,
        )

        if len(x_resyn) == 0:
            x_resyn = x[comp]
        else:
            x_resyn += x[comp]

    x_resyn /= np.max(x_resyn)

    audio_save_name = PurePath(audio_path).name.replace(
        ".wav", "_sf_bass_%2.2f_treble_%2.2f_tilt_%2.2f.wav" % tuple(gain_curve_params)
    )

    sf.write(os.path.join(out_dir, audio_save_name), x_resyn, fs)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="attack-remix")
    parser.add_argument("--input", type=str, default="", help="path to input audio")
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/",
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
        help="list of 3-tuples with gain factors for filtering. Tuple entries correspond to each of bass, treble, & tilt filters.",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=4, help="number of cores to run program on"
    )

    args, _ = parser.parse_known_args()

    winDur = args.win_dur * 1e-3
    hopDur = args.hop_dur * 1e-3

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    _ = Parallel(n_jobs=args.n_jobs)(
        delayed(augment_data_spectral_shape)(
            args.input, args.output, args.fs, gain_set, winDur=winDur, hopDur=hopDur
        )
        for gain_set in args.params
    )
