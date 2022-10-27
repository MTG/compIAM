import os
import numpy as np
import librosa
import soundfile as sf
from joblib import Parallel, delayed
import pytsmod as tsm
import argparse
from pathlib import PurePath


def augment_data_pitch_shift(audio_path, out_dir, fs, pitch_shift_factor):
    x, fs = librosa.load(audio_path, sr=fs)
    x /= np.max(np.abs(x))

    pitch_shift_factor_hz = 2 ** (pitch_shift_factor / 12)

    y = tsm.hptsm(x, pitch_shift_factor_hz)
    y = librosa.resample(y, fs, fs / pitch_shift_factor_hz)

    sf.write(
        os.path.join(
            out_dir,
            PurePath(audio_path).name.replace(
                ".wav", "_ps_%2.2f.wav" % pitch_shift_factor
            ),
        ),
        y,
        fs,
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pitch-shift")
    parser.add_argument("--input", type=str, default="", help="path to input audio")
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/",
        help="folder to save modified audios",
    )
    parser.add_argument("--fs", type=int, default=16000, help="sampling rate")
    parser.add_argument(
        "--params",
        nargs="+",
        type=float,
        default=[-1.0, -0.5, 0.5, 1.0],
        help="list of pitch shift values in semitones",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=4, help="number of cores to run program on"
    )

    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    _ = Parallel(n_jobs=args.n_jobs)(
        delayed(augment_data_pitch_shift)(args.input, args.output, args.fs, ps)
        for ps in args.shifts
    )
