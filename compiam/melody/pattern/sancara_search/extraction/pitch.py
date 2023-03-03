import math
import numpy as np

from compiam.melody.pattern.sancara_search.extraction.sequence import (
    get_stability_mask,
    add_center_to_mask,
)
from compiam.melody.pattern.sancara_search.extraction.io import (
    get_timeseries,
    write_timeseries,
)
from compiam.utils.pitch import interpolate_below_length
from compiam.utils import get_logger

logger = get_logger(__name__)


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


def silence_stability_from_file(
    inpath,
    outpath,
    tonic=None,
    min_stability_length_secs=1,
    stab_hop_secs=0.2,
    freq_var_thresh_stab=8,
    gap_interp=0.250,
):
    pitch, time, timestep = get_timeseries(inpath)
    pitch_interp = interpolate_below_length(pitch, 0, (gap_interp / timestep))

    logger.info("Computing stability/silence mask")
    if tonic:
        pi = pitch_seq_to_cents(pitch_interp, tonic)
    else:
        pi = pitch_interp
    stable_mask = get_stability_mask(
        pi, min_stability_length_secs, stab_hop_secs, freq_var_thresh_stab, timestep
    )
    silence_mask = (pitch_interp == 0).astype(int)
    silence_mask = add_center_to_mask(silence_mask)
    silence_and_stable_mask = np.array(
        [int(any([i, j])) for i, j in zip(silence_mask, stable_mask)]
    )
    write_timeseries([time, silence_and_stable_mask], outpath)
