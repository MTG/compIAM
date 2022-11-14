import numpy as np

###############
# Melody utils
###############
def pitch_normalisation(pitch, tonic, bins_per_octave=120, max_value=4):
    """Normalize pitch given a tonic.

    :param pitch: a 2-D list with time-stamps and pitch values per timestamp.
    :param tonic: recording tonic to normalize the pitch to.
    :param bins_per_octave: number of frequency bins per octave.
    :param max_value: maximum value to clip the normalized pitch to.
    :returns: a 2-D list with time-stamps and normalized to a given tonic
        pitch values per timestamp.
    """
    pitch_values = pitch[:, 1]
    eps = np.finfo(np.float).eps
    normalised_pitch = bins_per_octave * np.log2(2.0 * (pitch_values + eps) / tonic)
    indexes = np.where(normalised_pitch <= 0)
    normalised_pitch[indexes] = 0
    indexes = np.where(normalised_pitch > max_value)
    normalised_pitch[indexes] = max_value
    return np.array([pitch[:, 0], normalised_pitch]).transpose()
