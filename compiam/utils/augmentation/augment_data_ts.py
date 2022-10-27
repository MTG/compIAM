import os
import numpy as np
import librosa
import soundfile as sf
import pytsmod as tsm
from joblib import Parallel, delayed
import argparse
from pathlib import PurePath


def augment_data_time_scale(audio_path, out_dir, fs, time_scale_factor):
    x, fs = librosa.load(audio_path, sr=fs)
    x /= np.max(np.abs(x))

    y = tsm.hptsm(x, time_scale_factor)

    sf.write(
        os.path.join(
            out_dir,
            PurePath(audio_path).name.replace(
                ".wav", "_ts_%2.2f.wav" % time_scale_factor
            ),
        ),
        y,
        fs,
    )
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="time-scale")
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
        default=[0.8, 0.9, 1.2, 1.3],
        help="list of time-scale values",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=4, help="number of cores to run program on"
    )

    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    _ = Parallel(n_jobs=args.n_jobs)(
        delayed(augment_data_time_scale)(args.input, args.output, args.fs, ts)
        for ts in args.params
    )
