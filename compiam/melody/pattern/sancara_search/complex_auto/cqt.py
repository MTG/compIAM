"""
Created on July 05, 2019

@author: Stefan Lattner

Sony CSL Paris, France

"""

import os
import librosa
import numpy as np

from functools import partial
from multiprocessing.pool import Pool

from compiam.melody.pattern.sancara_search.complex_auto.util import (
    normalize,
    load_pyc_bz,
    save_pyc_bz,
)
from compiam.utils import get_logger

logger = get_logger(__name__)


def standardize(x, axis=-1):
    """
    Performs contrast normalization (zero mean, unit variance)
    along the given axis.

    :param x: array to normalize
    :param axis: normalize along that axis
    :return: contrast-normalized array
    """
    stds_avg = np.std(x, axis=axis, keepdims=True)
    x -= np.mean(x, axis=axis, keepdims=True)
    x /= stds_avg + 1e-8
    return x


def load_audio(fn, sr=-1):
    file = fn
    logger.info(f"loading file {file}")
    audio, fs = librosa.load(fn, sr=sr)
    return audio, fs


def to_mono(signal):
    if len(signal.shape) == 1:
        return signal
    return signal[:, 0] / 2 + signal[:, 1] / 2


def get_signal(fn, use_nr_samples=-1, rand_midpoint=False, sr=-1):
    audio, fs = load_audio(fn, sr)
    audio = to_mono(audio)
    if use_nr_samples > 0 and use_nr_samples < len(audio):
        if rand_midpoint:
            mid_point = np.random.randint(
                use_nr_samples // 2, len(audio) - 1 - use_nr_samples // 2
            )
        else:
            mid_point = len(audio) // 2
        audio_snippet = audio[
            mid_point - use_nr_samples // 2 : mid_point + use_nr_samples // 2
        ]
    else:
        audio_snippet = audio

    return audio_snippet, fs


def to_cqt_repr(
    fn,
    n_bins,
    bins_per_oct,
    fmin,
    hop_length,
    use_nr_samples,
    rand_midpoint=False,
    standard=False,
    normal=False,
    mult=1.0,
    sr=-1,
):
    audio, sr = get_signal(fn, use_nr_samples, rand_midpoint, sr=sr)

    cqt = librosa.cqt(
        audio,
        sr=sr,
        n_bins=n_bins,
        bins_per_octave=bins_per_oct,
        fmin=fmin,
        hop_length=hop_length,
    )
    mag = librosa.magphase(cqt)[0]

    if standard:
        mag = standardize(mag, axis=0)

    if normal:
        mag = normalize(mag)

    return mag * mult


def get_cqts(
    files,
    cache_key="train",
    rebuild=False,
    use_nr_samples=-1,
    processes=10,
    sr=-1,
    args=None,
):
    assert args is not None, "args are needed."
    cache_fn = os.path.join(args.cache_dir, "hist_cache_" + cache_key + ".pyc.bz")
    cqts = []
    if not os.path.isfile(cache_fn) or rebuild:
        if processes > 1:
            calc_cqt_f = partial(
                to_cqt_repr,
                n_bins=args.n_bins,
                bins_per_oct=args.bins_per_oct,
                fmin=args.fmin,
                hop_length=args.hop_length,
                use_nr_samples=use_nr_samples,
                rand_midpoint=False,
                standard=True,
                sr=sr,
                mult=10,
            )

            with Pool(processes=processes) as pool:
                cqts = pool.map(calc_cqt_f, files)

        save_pyc_bz(cqts, cache_fn)
    else:
        cqts = load_pyc_bz(cache_fn)

    return cqts
