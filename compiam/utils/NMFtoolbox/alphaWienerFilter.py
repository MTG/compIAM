"""
    Name: alphaWienerFilter
    Date: Jun 2019
    Programmer: Christian Dittmar, Yiğitcan Özer

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    If you use the 'NMF toolbox' please refer to:
    [1]  Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
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

import numpy as np

from NMFtoolbox.utils import EPS


def alphaWienerFilter(mixtureX, sourceA, alpha=1.2, binarize=False):
    """Given a cell-array of spectrogram estimates as input, this function
    computes the alpha-related soft masks for extracting the sources. Details
    about this procedure are given in [2], further experimental studies in [3].

    References
    ----------
    [2] Antoine Liutkus and Roland Badeau: Generalized Wiener filtering with
    fractional power spectrograms, ICASPP 2015

    [3] Christian Dittmar et al.: An Experimental Approach to Generalized
    Wiener Filtering in Music Source Separation, EUSIPCO 2016

    Parameters
    ----------
    mixtureX: array_like
        The mixture spectrogram (numBins x numFrames) (may be real-or complex-valued)

    sourceA: array_like
        A list holding the equally sized spectrogram estimates of single sound sources (aka components)

    alpha: float
        The fractional power in rand [0 ... 2]

    binarize: bool
        If this is set to True, we binarize the masks


    Returns
    -------
    sourceX: array_like
        A list of extracted source spectrograms

    softMasks: array_like
        A list with the extracted masks
    """

    numBins, numFrames = mixtureX.shape
    numComp = len(sourceA)

    #  Initialize the mixture of the sources / components with a small constant
    mixtureA = EPS + np.zeros((numBins, numFrames))

    softMasks = list()
    sourceX = list()

    # Make superposition
    for k in range(numComp):
        mixtureA += sourceA[k] ** alpha

    # Compute soft masks and spectrogram estimates
    for k in range(numComp):
        currSoftMask = (sourceA[k] ** alpha) / mixtureA
        softMasks.append(currSoftMask.astype(np.float32))

        #  If desired, make this a binary mask
        if binarize:
            tmp = softMasks[k]
            softMasks[k] = tmp[tmp > (1.0/numComp)] * 1

        #  And apply it to the mixture
        sourceX.append(mixtureX * currSoftMask)

    return sourceX, softMasks
