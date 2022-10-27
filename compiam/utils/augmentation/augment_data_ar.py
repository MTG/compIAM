import os
import numpy as np
import librosa
import soundfile as sf
from joblib import Parallel, delayed
import argparse
from pathlib import PurePath


def attack_remix_hps(x, fs, winSize, hopSize, window_type, nFFT, attack_gain_factor):
    S = librosa.stft(
        x,
        n_fft=nFFT,
        hop_length=hopSize,
        win_length=winSize,
        window=window_type,
        center=True,
        pad_mode="reflect",
    )
    H, P = librosa.decompose.hpss(S, kernel_size=31, power=1.0, mask=False, margin=1.0)
    x_harm = librosa.istft(
        H, hop_length=hopSize, win_length=winSize, window="hann", center=True
    )
    x_perc = librosa.istft(
        P, hop_length=hopSize, win_length=winSize, window="hann", center=True
    )
    x_remixed = x_harm + attack_gain_factor * x_perc
    return x_remixed


def augment_data_attack_remix(
    audio_path, out_dir, fs, attack_gain_factor, winDur, hopDur
):
    winSize = int(np.ceil(winDur * fs))
    hopSize = int(np.ceil(hopDur * fs))
    nFFT = 2 ** (int(np.ceil(np.log2(winSize))) + 1)

    x, fs = librosa.load(audio_path, sr=fs)
    x /= np.max(np.abs(x))

    x_remixed = attack_remix_hps(
        x, fs, winSize, hopSize, "hann", nFFT, attack_gain_factor
    )
    x_remixed /= np.max(np.abs(x_remixed))
    sf.write(
        os.path.join(
            out_dir,
            PurePath(audio_path).name.replace(
                ".wav", "_ar_%2.2f.wav" % attack_gain_factor
            ),
        ),
        x_remixed,
        fs,
    )
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
        "--hop_dur", type=str, default=5, help="hop size in milliseconds"
    )
    parser.add_argument(
        "--params",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 2, 3],
        help="list of gain factors to scale attack portion with",
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
        delayed(augment_data_attack_remix)(
            args.input, args.output, args.fs, G, winDur, hopDur
        )
        for G in args.params
    )
