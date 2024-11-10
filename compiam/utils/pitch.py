import math
import scipy
import warnings

import numpy as np
import pandas as pd


###############
# Melody utils
###############


def normalisation(pitch, tonic, bins_per_octave=120, max_value=4):
    """Normalize pitch given a tonic.

    :param pitch: a 2-D list with time-stamps and pitch values per timestamp.
    :param tonic: recording tonic to normalize the pitch to.
    :param bins_per_octave: number of frequency bins per octave.
    :param max_value: maximum value to clip the normalized pitch to.
    :returns: a 2-D list with time-stamps and normalized to a given tonic
        pitch values per timestamp.
    """
    pitch_values = pitch[:, 1]
    eps = np.finfo(float).eps
    normalised_pitch = bins_per_octave * np.log2(2.0 * (pitch_values + eps) / tonic)
    indexes = np.where(normalised_pitch <= 0)
    normalised_pitch[indexes] = 0
    indexes = np.where(normalised_pitch > max_value)
    normalised_pitch[indexes] = max_value
    return np.array([pitch[:, 0], normalised_pitch]).transpose()


def resampling(pitch, new_len):
    """Resample pitch to a given new length in samples

    :param pitch: a 2-D list with time-stamps and pitch values per timestamp.
    :param new_len: new length of the output pitch
    """
    times = pitch[:, 0]
    frequencies = pitch[:, 1]

    voicing = []
    for freq in frequencies:
        voicing.append(1) if freq > 0 else voicing.append(0)

    times_new = np.linspace(times[0], times[-1], num=new_len)

    frequencies_resampled, _ = resample_melody_series(
        times=times,
        frequencies=frequencies,
        voicing=np.array(voicing),
        times_new=times_new,
        kind="linear",
    )

    return np.array([times_new, frequencies_resampled]).transpose()


def resample_melody_series(times, frequencies, voicing, times_new, kind="linear"):
    """
    NOTE: This function is DIRECTLY PORTED FROM mir_eval.melody

    Resamples frequency and voicing time series to a new timescale. Maintains
    any zero ("unvoiced") values in frequencies.

    If ``times`` and ``times_new`` are equivalent, no resampling will be
    performed.

    :param times: Times of each frequency value
    :param frequencies: Array of frequency values, >= 0
    :param voicing: Array which indicates voiced or unvoiced. This array may be binary
        or have continuous values between 0 and 1.
    :param times_new: Times to resample frequency and voicing sequences to
    :param kind:kind parameter to pass to scipy.interpolate.interp1d.
        (Default value = 'linear')

    :returns frequencies_resampled: Frequency array resampled to new timebase
    :returns voicing_resampled: Voicing array resampled to new timebase
    """
    # If the timebases are already the same, no need to interpolate
    if times.shape == times_new.shape and np.allclose(times, times_new):
        return frequencies, voicing

    # Warn when the delta between the original times is not constant,
    # unless times[0] == 0. and frequencies[0] == frequencies[1] (see logic at
    # the beginning of to_cent_voicing)
    if not (
        np.allclose(np.diff(times), np.diff(times).mean())
        or (
            np.allclose(np.diff(times[1:]), np.diff(times[1:]).mean())
            and frequencies[0] == frequencies[1]
        )
    ):
        warnings.warn(
            "Non-uniform timescale passed to resample_melody_series.  Pitch "
            "will be linearly interpolated, which will result in undesirable "
            "behavior if silences are indicated by missing values.  Silences "
            "should be indicated by nonpositive frequency values."
        )
    # Round to avoid floating point problems
    times = np.round(times, 10)
    times_new = np.round(times_new, 10)
    # Add in an additional sample if we'll be asking for a time too large
    if times_new.max() > times.max():
        times = np.append(times, times_new.max())
        frequencies = np.append(frequencies, 0)
        voicing = np.append(voicing, 0)
    # We need to fix zero transitions if interpolation is not zero or nearest
    if kind != "zero" and kind != "nearest":
        # Fill in zero values with the last reported frequency
        # to avoid erroneous values when resampling
        frequencies_held = np.array(frequencies)
        for n, frequency in enumerate(frequencies[1:]):
            if frequency == 0:
                frequencies_held[n + 1] = frequencies_held[n]
        # Linearly interpolate frequencies
        frequencies_resampled = scipy.interpolate.interp1d(
            times, frequencies_held, kind
        )(times_new)
        # Retain zeros
        frequency_mask = scipy.interpolate.interp1d(times, frequencies, "zero")(
            times_new
        )
        frequencies_resampled *= frequency_mask != 0
    else:
        frequencies_resampled = scipy.interpolate.interp1d(times, frequencies, kind)(
            times_new
        )

    # Use nearest-neighbor for voicing if it was used for frequencies
    # if voicing is not binary, use linear interpolation
    is_binary_voicing = np.all(
        np.logical_or(np.equal(voicing, 0), np.equal(voicing, 1))
    )
    if kind == "nearest" or (kind == "linear" and not is_binary_voicing):
        voicing_resampled = scipy.interpolate.interp1d(times, voicing, kind)(times_new)
    # otherwise, always use zeroth order
    else:
        voicing_resampled = scipy.interpolate.interp1d(times, voicing, "zero")(
            times_new
        )

    return frequencies_resampled, voicing_resampled


#####################
## Pitch Stability ##
#####################


def extract_stability_mask(pitch, min_stab_sec, hop_sec, var, timestep):
    """
    Extract boolean array corresponding to <pitch> - yes/no does point correspond
    to a region of "stable" pitch.

    A window is passed along the pitch track, <pitch> and the minimum and maximum values
    compared to the average for that window. Regions corresponding to windows
    whose extremes deviate significantly from their means are marked as stable.
    Consecutive stable regions summing to more than <min_stab_sec> seconds in
    length are annotated with 1 indicating stable. Regions which are not stable
    or that are stable but do not make up at least <min_stab_sec> in length are
    annotated with 0 - not stable.

    :param pitch: Pitch values in Hz or cents
    :type pitch: np.ndarray
    :param min_stab_sec: Stable regions of at least <min_stab_sec> seconds
        in length are annotated as stable. Shorter regions are not annotated.
    :type min_stab_sec: float
    :param hop_sec: Hop length in seconds of window
    :type hop_sec: float
    :param var: If the maximum/minimum pitch in a window deviates from its mean
        by more than this value, the window is considered unstable. Important
        to consider if the input is in cents or hertz!
    :type var: float
    :param timestep: Time difference, in seconds, between each element in pitch
    :type timestep: float

    :return: Boolean array equal in length to <pitch>: is stable region or not?
    :rtype: np.ndarray
    """
    stab_hop = int(hop_sec / timestep)
    reverse_pitch = np.flip(pitch)

    # apply in both directions to array to account for hop_size errors
    stable_mask_1 = [is_stable(pitch[s : s + stab_hop], var) for s in range(len(pitch))]
    stable_mask_2 = [
        is_stable(reverse_pitch[s : s + stab_hop], var)
        for s in range(len(reverse_pitch))
    ]

    silence_mask = pitch == 0

    zipped = zip(stable_mask_1, np.flip(stable_mask_2), silence_mask)

    stable_mask = np.array([int((any([s1, s2]) and not sil)) for s1, s2, sil in zipped])

    stable_mask = reduce_stability_mask(stable_mask, min_stab_sec, timestep)

    # stable_mask = add_center_to_mask(stable_mask)

    return stable_mask


def reduce_stability_mask(stable_mask, min_stab_sec, timestep):
    min_stability_length = int(min_stab_sec / timestep)
    num_one = 0
    indices = []
    for i, s in enumerate(stable_mask):
        if s == 1:
            num_one += 1
            indices.append(i)
        else:
            if num_one < min_stability_length:
                for ix in indices:
                    stable_mask[ix] = 0
            num_one = 0
            indices = []
    return stable_mask


def is_stable(seq, max_var):
    """Compute is sequence of value has stability given an input tolerance

    :param seq: sequence of values to study
    :param max_var: Maximum tolerance to consider stable/not stable

    :returns: 1 (stable) or 0 (not stable)
    """
    if None in seq:
        return 0
    seq = seq.astype(float)
    mu = np.nanmean(seq)

    maximum = np.nanmax(seq)
    minimum = np.nanmin(seq)
    if (maximum < mu + max_var) and (minimum > mu - max_var):
        return 1
    else:
        return 0


def add_center_to_mask(mask):
    num_one = 0
    indices = []
    for i, s in enumerate(mask):
        if s == 1:
            num_one += 1
            indices.append(i)
        else:
            li = len(indices)
            if li:
                middle = indices[int(li / 2)]
                mask[middle] = 2
                num_one = 0
                indices = []
    return mask


def pitch_to_cents(p, tonic):
    """
    Convert pitch value, <p> to cents above <tonic>.

    :param p: Pitch value in Hz
    :type p: float
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Pitch value, <p> in cents above <tonic>
    :rtype: float
    """
    return 1200 * math.log(p / tonic, 2) if p else None


def cents_to_pitch(c, tonic):
    """
    Convert cents value, <c> to pitch in Hz

    :param c: Pitch value in cents above <tonic>
    :type c: float/int
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Pitch value, <c> in Hz
    :rtype: float
    """
    return (2 ** (c / 1200)) * tonic


def pitch_seq_to_cents(pseq, tonic):
    """
    Convert sequence of pitch values to sequence of
    cents above <tonic> values

    :param pseq: Array of pitch values in Hz
    :type pseq: np.array
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Sequence of original pitch value in cents above <tonic>
    :rtype: np.array
    """
    return np.vectorize(lambda y: pitch_to_cents(y, tonic))(pseq)


def interpolate_below_length(arr, val, gap):
    """
    Interpolate gaps of value, <val> of
    length equal to or shorter than <gap> in <arr>

    :param arr: Array to interpolate
    :type arr: np.array
    :param val: Value expected in gaps to interpolate
    :type val: number
    :param gap: Maximum gap length to interpolate, gaps of <val> longer than <g> will not be interpolated
    :type gap: number

    :return: interpolated array
    :rtype: np.array
    """
    s = np.copy(arr)
    is_zero = s == val
    cumsum = np.cumsum(is_zero).astype("float")
    diff = np.zeros_like(s)
    diff[~is_zero] = np.diff(cumsum[~is_zero], prepend=0)
    for i, d in enumerate(diff):
        if d <= gap:
            s[int(i - d) : i] = np.nan
    interp = pd.Series(s).interpolate(method="linear", axis=0).ffill().bfill().values
    return interp
