"""
    Name: NEMA
    Date: Jun 2019
    Programmer: Christian Dittmar, Yiğitcan Özer

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    If you use the 'NMF toolbox' please refer to:
    [1] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
        Müller
        NMF Toolbox: Music Processing Applications of Nonnegative Matrix
        Factorization
        In Proceedings of the International Conference on Digital Audio Effects
        (DAFx), 2019.

    License:
    This file is part of 'NMF toolbox'.
    https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
    'NMF toolbox' is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    the Free Software Foundation, either version 3 of the License, or (at
    your option) any later version.

    'NMF toolbox' is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
    Public License for more details.

    You should have received a copy of the GNU General Public License along
    with 'NMF toolbox'. If not, see http://www.gnu.org/licenses/.
"""

from copy import deepcopy
import numpy as np


def NEMA(A, lamb=0.9):
    """This function takes a matrix of row-wise time series and applies a
    non-linear exponential moving average (NEMA) to each row. This filter
    introduces exponentially decaying slopes and is defined in eq. (3) from [2].

    The difference equation of that filter would be:
    y(n) = max( x(n), y(n-1)*(decay) + x(n)*(1-decay) )

    References
    ----------
    [2] Christian Dittmar, Patricio López-Serrano, Meinard Müller: "Unifying
    Local and Global Methods for Harmonic-Percussive Source Separation"
    In Proceedings of the IEEE International Conference on Acoustics,
    Speech, and Signal Processing (ICASSP), 2018.

    Parameters
    ----------
    A: array-like
        The matrix with time series in its rows

    lamb: array-like / float
        The decay parameter in the range [0 ... 1], this can be
        given as a column-vector with individual decays per row
        or as a scalar

    Results
    -------
    filtered: array-like
        The result after application of the NEMA filter
    """

    # Prevent instable filter
    lamb = max(0.0, min(0.9999999, lamb))

    numRows, numCols = A.shape
    filtered = deepcopy(A)

    for k in range(1, numCols):
        storeRow = deepcopy(filtered[:, k])
        filtered[:, k] = lamb * filtered[:, k-1] + filtered[:, k] * (1 - lamb)
        filtered[:, k] = np.maximum(filtered[:, k], storeRow)

    return filtered
