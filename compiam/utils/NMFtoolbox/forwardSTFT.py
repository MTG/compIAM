"""
    Name: forwardSTFT
    Date of Revision: Jun 2019
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
from scipy.fftpack import fft


def forwardSTFT(x, parameter=None):
    """Given a time signal as input, this computes the spectrogram by means of
    the Short-time fourier transform

    Parameters
    ----------
    x: array-like
        The time signal oriented as numSamples x 1

    parameter: dict
        blockSize       The blocksize to use during analysis
        hopSize         The hopsize to use during analysis
        winFunc         The analysis window
        reconstMirror   This switch decides whether to discard the mirror
                        spectrum or not
        appendFrame     This switch decides if we use silence in the
                        beginning and the end


    Returns
    -------
    X: array-like
        The complex valued spectrogram in numBins x numFrames

    A: array-like
        The magnitude spectrogram

    P: array-like
        The phase spectrogram (wrapped in -pi ... +pi)
    """
    parameter = init_parameters(parameter)
    blockSize = parameter['blockSize']
    halfBlockSize = round(blockSize / 2)
    hopSize = parameter['hopSize']
    winFunc = parameter['winFunc']
    reconstMirror = parameter['reconstMirror']
    appendFrame = parameter['appendFrame']

    # the number of bins needs to be corrected
    # if we want to discard the mirror spectrum
    if parameter['reconstMirror']:
        numBins = round(parameter['blockSize'] / 2) + 1
    else:
        numBins = parameter['blockSize']

    # append safety space in the beginning and end
    if appendFrame:
        x = np.concatenate((np.zeros(halfBlockSize), x, np.zeros(halfBlockSize)), axis=0)

    numSamples = len(x)

    # pre-compute the number of frames
    numFrames = round(numSamples / hopSize)

    # initialize with correct size
    X = np.zeros((np.int(numBins), numFrames), dtype=np.complex64)

    counter = 0

    for k in range(0, len(x)-blockSize, hopSize):
        # where to pick
        ind = range(k, k+blockSize)

        # pick signal frame
        snip = x[ind]

        # apply windowing
        snip *= winFunc

        # do FFT
        f = fft(snip, axis=0)

        # if required, remove the upper half of spectrum
        if reconstMirror:
            f = np.delete(f, range(numBins, blockSize), axis=0)

        # store into STFT matrix
        X[:, counter] = f
        counter += 1

    # after constructing the STFT array, remove excessive frames
    X = np.delete(X, range(counter, numFrames), axis=1)

    # compute derived matrices
    # get magnitude
    A = np.abs(X)

    # get phase
    P = np.angle(X)

    # return complex-valued STFT, magnitude STFT, and phase STFT
    return X, A, P


def init_parameters(parameter):
    """Auxiliary function to set the parameter dictionary

    Parameters
    ----------
    parameter: dict
        See the above function forwardSTFT for further information

    Returns
    -------
    parameter: dict
    """
    parameter = dict() if not parameter else parameter
    parameter['blockSize'] = 2048 if 'blockSize' not in parameter else parameter['blockSize']
    parameter['hopSize'] = 512 if 'hopSize' not in parameter else parameter['hopSize']
    parameter['winFunc'] = np.hanning(parameter['blockSize']) if 'winFunc' not in parameter else parameter['winFunc']
    parameter['reconstMirror'] = True if 'reconstMirror' not in parameter else parameter['reconstMirror']
    parameter['appendFrame'] = True if 'appendFrame' not in parameter else parameter['appendFrame']

    return parameter
