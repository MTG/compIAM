import numpy as np
import pandas as pd

from compiam.utils import get_logger

logger = get_logger(__name__)


def find_nearest(array, value, index=True):
    """
    Find the closest element of <array> to <value>

    :param array: array of values
    :type array: numpy.array
    :param value: value to check
    :type value: float
    :param index: True or False, return index or value in <array> of closest element?
    :type index: bool

    :return: index/value of element in <array> closest to <value>
    :rtype: number
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx if index else array[idx]


def myround(x, base=5):
    return base * round(x / base)


def check_stability(this_seq, thresh=130):
    return True if np.var(this_seq) < thresh else False


def get_timestamp(secs, divider="-"):
    """
    Convert seconds into timestamp

    :param secs: seconds
    :type secs: int
    :param divider: divider between minute and second, default "-"
    :type divider: str

    :return: timestamp
    :rtype: str
    """
    minutes = int(secs / 60)
    seconds = round(secs % 60, 2)
    return f"{minutes}min{divider}{seconds}sec"


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
