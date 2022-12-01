import mir_eval

import numpy as np

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

    frequencies_resampled, _ = mir_eval.melody.resample_melody_series(
        times=times,
        frequencies=frequencies,
        voicing=np.array(voicing),
        times_new=times_new,
        kind="linear",
    )

    return np.array([times_new, frequencies_resampled]).transpose()


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
    stab_hop = int(hop_sec/timestep)
    reverse_pitch = np.flip(pitch)

    # apply in both directions to array to account for hop_size errors
    stable_mask_1 = [is_stable(pitch[s:s+stab_hop], var) for s in range(len(pitch))]
    stable_mask_2 = [is_stable(reverse_pitch[s:s+stab_hop], var) for s in range(len(reverse_pitch))]
    
    silence_mask = pitch == 0

    zipped = zip(stable_mask_1, np.flip(stable_mask_2), silence_mask)
    
    stable_mask = np.array([int((any([s1,s2]) and not sil)) for s1,s2,sil in zipped])

    stable_mask = reduce_stability_mask(stable_mask, min_stab_sec, timestep)

    # stable_mask = add_center_to_mask(stable_mask)

    return stable_mask


def reduce_stability_mask(stable_mask, min_stab_sec, timestep):
    min_stability_length = int(min_stab_sec/timestep)
    num_one = 0
    indices = []
    for i,s in enumerate(stable_mask):
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
    for i,s in enumerate(mask):
        if s == 1:
            num_one += 1
            indices.append(i)
        else:
            li = len(indices)
            if li:
                middle = indices[int(li/2)]
                mask[middle] = 2
                num_one = 0
                indices = []
    return mask