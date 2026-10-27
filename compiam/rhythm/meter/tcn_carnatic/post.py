import numpy as np
from scipy.ndimage import maximum_filter1d

from compiam.utils.dbn import (
    dbn_beat_tracking,
    dbn_downbeat_tracking,
    dbn_bar_tracking,
)


def clip_probabilities(probs, epsilon=1e-5):
    """Clip probabilities to avoid exact 0 and 1 values that cause DBN issues."""
    probs = np.maximum(probs, 0)
    probs = np.minimum(probs, 1)
    return probs * (1 - epsilon) + epsilon / 2


def beat_tracker(beats_act, min_bpm=55, max_bpm=230, fps=100, transition_lambda=100):
    """Track beats in a beat activation function.

    :param beats_act: 1-D array with the beat activations.
    :param min_bpm: minimum BPM for beat tracking.
    :param max_bpm: maximum BPM for beat tracking.
    :param fps: frames per second of the activations.
    :param transition_lambda: higher values favour a constant tempo over a tempo change.

    :returns: 1-D array with the beat times (in seconds).
    """
    beats_act = clip_probabilities(beats_act)
    if beats_act.size <= 1:
        return np.array([])
    return dbn_beat_tracking(
        beats_act,
        fps=fps,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        transition_lambda=transition_lambda,
    )


def joint_tracker(
    beats_act,
    downbeats_act,
    min_bpm=55,
    max_bpm=230,
    fps=100,
    beats_per_bar=(3, 5, 7, 8),
):
    """Jointly track beats and downbeats.

    :param beats_act: 1-D array with the beat activations.
    :param downbeats_act: 1-D array with the downbeat activations.
    :param min_bpm: minimum BPM for beat tracking.
    :param max_bpm: maximum BPM for beat tracking.
    :param fps: frames per second of the activations.
    :param beats_per_bar: list of possible beats per bar.

    :returns: 2-D array with the beat times (in seconds) and their position in the bar.
    """
    beats_act = clip_probabilities(beats_act)
    downbeats_act = clip_probabilities(downbeats_act)

    combined_act = np.vstack(
        (np.maximum(beats_act - downbeats_act, 0), downbeats_act)
    ).T
    return dbn_downbeat_tracking(
        combined_act,
        fps=fps,
        beats_per_bar=beats_per_bar,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )


def sequential_tracker(
    beats_act,
    downbeats_act,
    min_bpm=55,
    max_bpm=230,
    fps=100,
    beats_per_bar=(3, 5, 7, 8),
):
    """Track beats first, and then their position in the bar.

    :param beats_act: 1-D array with the beat activations.
    :param downbeats_act: 1-D array with the downbeat activations.
    :param min_bpm: minimum BPM for beat tracking.
    :param max_bpm: maximum BPM for beat tracking.
    :param fps: frames per second of the activations.
    :param beats_per_bar: list of possible beats per bar.

    :returns: 2-D array with the beat times (in seconds) and their position in the bar.
    """
    downbeats_act = clip_probabilities(downbeats_act)

    beats = beat_tracker(beats_act, min_bpm=min_bpm, max_bpm=max_bpm, fps=fps)
    if len(beats) < 2:
        return np.empty((0, 2))

    beat_idx = np.round(beats * fps).astype(int)
    bar_act = maximum_filter1d(downbeats_act, size=3)[beat_idx]
    return dbn_bar_tracking(
        beats,
        bar_act,
        beats_per_bar=beats_per_bar,
        meter_change_prob=1e-3,
        observation_weight=4,
    )
