import numpy as np
import torch
import torch.nn.functional as F

from compiam.utils.dbn import dbn_downbeat_tracking


def deduplicate_peaks(peaks, width=1):
    """Replace groups of adjacent peak frame indices, each not more than `width` frames apart, by their average."""
    result = []
    peaks = map(int, peaks)
    try:
        p = next(peaks)
    except StopIteration:
        return np.array(result)
    c = 1
    for p2 in peaks:
        if p2 - p <= width:
            c += 1
            p += (p2 - p) / c  # update mean
        else:
            result.append(p)
            p = p2
            c = 1
    result.append(p)
    return np.array(result)


def beat_positions(beats, downbeats):
    """Position of each beat within its bar.

    Beats are numbered from 1 (downbeat) to the last beat before the next downbeat. Beats before
    the first downbeat are numbered backwards from it, using the length of the first bar. If the
    length of the bars is unknown (less than two downbeats) these are assigned position 0.

    :param beats: 1-D array with the beat times.
    :param downbeats: 1-D array with the downbeat times, that are a subset of the beats.

    :returns: 1-D array of integers with the position of each beat.
    """
    positions = np.zeros(len(beats), dtype=int)
    if len(downbeats) == 0:
        return positions
    downbeat_idx = np.flatnonzero(np.isin(beats, downbeats))
    for start, stop in zip(downbeat_idx, np.r_[downbeat_idx[1:], len(beats)]):
        positions[start:stop] = np.arange(1, stop - start + 1)
    if downbeat_idx[0] > 0 and len(downbeat_idx) > 1:
        bar_length = downbeat_idx[1] - downbeat_idx[0]
        pickup = np.arange(downbeat_idx[0])
        positions[: downbeat_idx[0]] = bar_length - (downbeat_idx[0] - pickup) + 1
    return positions


def minimal_tracker(beats_act, downbeats_act, fps=50):
    """Pick the peaks of the beat and downbeat activations, without any further modelling.

    :param beats_act: 1-D tensor with the beat logits.
    :param downbeats_act: 1-D tensor with the downbeat logits.
    :param fps: frames per second of the activations.

    :returns: 2-D array with the beat times (in seconds) and their position in the bar.
    """
    peaks = []
    for logits in (beats_act, downbeats_act):
        # maxima within +/- 70ms, that have over 0.5 probability (logit > 0)
        pooled = F.max_pool1d(logits[None, None], 7, 1, 3)[0, 0]
        peaks.append(
            torch.nonzero((logits == pooled) & (logits > 0))[:, 0].cpu().numpy()
        )
    beat_time = deduplicate_peaks(peaks[0]) / fps
    downbeat_time = deduplicate_peaks(peaks[1]) / fps

    if len(beat_time) == 0:
        return np.empty((0, 2))
    # move the downbeats to the nearest beat, and remove duplicates
    downbeat_time = np.unique(
        beat_time[np.argmin(np.abs(beat_time[:, None] - downbeat_time[None]), axis=0)]
        if len(downbeat_time) > 0
        else downbeat_time
    )
    return np.vstack((beat_time, beat_positions(beat_time, downbeat_time))).T


def dbn_tracker(
    beats_act,
    downbeats_act,
    fps=50,
    min_bpm=55,
    max_bpm=230,
    beats_per_bar=(3, 5, 7, 8),
):
    """Track beats and downbeats with a Dynamic Bayesian Network.

    :param beats_act: 1-D tensor with the beat logits.
    :param downbeats_act: 1-D tensor with the downbeat logits.
    :param fps: frames per second of the activations.
    :param min_bpm: minimum BPM for beat tracking.
    :param max_bpm: maximum BPM for beat tracking.
    :param beats_per_bar: list of possible beats per bar.

    :returns: 2-D array with the beat times (in seconds) and their position in the bar.
    """
    # limit lower and upper bound, since 0 and 1 create problems in the DBN
    epsilon = 1e-5
    beat_prob = beats_act.double().sigmoid().cpu().numpy() * (1 - epsilon) + epsilon / 2
    downbeat_prob = (
        downbeats_act.double().sigmoid().cpu().numpy() * (1 - epsilon) + epsilon / 2
    )
    # artificial multiclass prediction, as suggested by Böck et al.
    combined_act = np.vstack(
        (np.maximum(beat_prob - downbeat_prob, epsilon / 2), downbeat_prob)
    ).T
    return dbn_downbeat_tracking(
        combined_act,
        fps=fps,
        beats_per_bar=beats_per_bar,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        transition_lambda=100,
    )
