"""
    Name: inverseSTFT
    Date of Revision: Jun 2019
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

import numpy as np
from scipy.fftpack import ifft
from copy import deepcopy


def inverseSTFT(X, parameter):
    """Given a valid STFT spectrogram as input, this reconstructs the corresponding
    time-domain signal by  means of the frame-wise inverse FFT and overlap-add
    method described as LSEE-MSTFT in [2].

    References
    ----------
    [2] Daniel W. Griffin and Jae S. Lim, "Signal estimation
        from modified short-time fourier transform", IEEE
        Transactions on Acoustics, Speech and Signal Processing, vol. 32, no. 2,
        pp. 236-243, Apr 1984.

    Parameters
    ----------
    X: array-like
        The complex-valued spectrogram matrix oriented with dimensions
        numBins x numFrames

    parameter: dict
        blockSize      The blocksize to use during synthesis
        hopSize        The hopsize to use during synthesis
        anaWinFunc     The analysis window function
        reconstMirror  This switch decides whether the mirror spectrum
                       should be reconstructed or not
        appendFrame    This switch decides whethter to compensate for
                       zero padding or not
        synWinFunc     The synthesis window function (per default the
                       same as analysis window)
        analyticSig    If this is set to True, we want the analytic signal
        numSamples     The original number of samples

    Returns
    -------
    y: array-like
        The resynthesized signal

    env: array-like
        The envelope used for normalization of the synthesis window
    """

    # get dimensions of STFT array and prepare corresponding output
    numBins, numFrames = X.shape

    # initialize parameters
    parameter = init_parameters(parameter, numBins)

    reconstMirror = parameter['reconstMirror']
    appendFrame = parameter['appendFrame']
    analyticSig = parameter['analyticSig']
    blockSize = parameter['blockSize']
    hopSize = parameter['hopSize']
    numPadBins = blockSize - numBins
    numSamples = numFrames * hopSize + blockSize

    # for simplicity, we assume the analysis and synthesis windows to be equal
    analysisWinFunc = deepcopy(parameter['winFunc'])
    synthesisWinFunc = deepcopy(parameter['winFunc'])

    # prepare helper variables
    halfBlockSize = round(blockSize / 2)

    # check if input parameters are complete
    if analysisWinFunc is None:
        analysisWinFunc = np.hanning(blockSize)

    if synthesisWinFunc is None:
        synthesisWinFunc = np.hanning(blockSize)

    # we need to change the signal scaling in case of the analytic signal
    scale = 2.0 if analyticSig else 1.0

    # decide between analytic and real output
    y = np.zeros(numSamples, dtype=np.complex64) if analyticSig else np.zeros(numSamples, dtype=np.float32)

    # construct normalization function for the synthesis window
    # that equals the denominator in eq. (6) in [1]
    winFuncProd = analysisWinFunc * synthesisWinFunc
    redundancy = round(blockSize / hopSize)

    # construct hopSize-periodic normalization function that will be
    # applied to the synthesis window
    nrmFunc = np.zeros(blockSize)

    # begin with construction outside the support of the window
    for k in range(-redundancy + 1, redundancy):
        nrmFuncInd = hopSize * k
        winFuncInd = np.arange(0, blockSize)
        nrmFuncInd += winFuncInd

        # check which indices are inside the defined support of the window
        validIndex = np.where((nrmFuncInd >= 0) & (nrmFuncInd < blockSize))
        nrmFuncInd = nrmFuncInd[validIndex]
        winFuncInd = winFuncInd[validIndex]

        # accumulate product of analysis and synthesis window
        nrmFunc[nrmFuncInd] += winFuncProd[winFuncInd]

    # apply normalization function
    synthesisWinFunc /= nrmFunc

    # prepare index for output signal construction
    frameInd = np.arange(0, blockSize)

    # then begin frame-wise reconstruction
    for k in range(numFrames):

        # pick spectral frame
        currSpec = deepcopy(X[:, k])

        # if desired, construct artificial mirror spectrum
        if reconstMirror:
            # if the analytic signal is wanted, put zeros instead
            padMirrorSpec = np.zeros(numPadBins)

            if not analyticSig:
                padMirrorSpec = np.conjugate(np.flip(currSpec[1:-1], axis=0))

            # concatenate mirror spectrum to base spectrum
            currSpec = np.concatenate((currSpec, padMirrorSpec), axis=0)

        # transform to time-domain
        snip = ifft(currSpec)

        # treat differently if analytic signal is desired
        if not analyticSig:
            snip = np.real(snip)

        # apply scaling and synthesis window
        snip *= synthesisWinFunc * scale

        # determine overlap-add position
        overlapAddInd = k * hopSize + frameInd

        # and do the overlap add, with synthesis window and scaling factor included
        y[overlapAddInd] += snip

    # check if borders need to be removed
    if appendFrame:
        y = y[halfBlockSize:len(y) - halfBlockSize]

    # check if number of samples was defined from outside
    if parameter['numSamples']:
        y = y[0:parameter['numSamples']]

    return y.reshape(-1, 1), synthesisWinFunc


def init_parameters(parameter, numBins):
    """Auxiliary function to set the parameter dictionary

    Parameters
    ----------
    parameter: dict
        See the above function inverseSTFT for further information

    Returns
    -------
    parameter: dict
    """
    parameter = dict() if not parameter else parameter
    parameter['blockSize'] = 2048 if 'blockSize' not in parameter else parameter['blockSize']
    parameter['hopSize'] = 512 if 'hopSize' not in parameter else parameter['hopSize']
    parameter['winFunc'] = np.hanning(parameter['blockSize']) if 'winFunc' not in parameter else parameter['winFunc']
    parameter['appendFrame'] = True if 'appendFrame' not in parameter else parameter['appendFrame']
    parameter['analyticSig'] = False if 'analyticSig' not in parameter else parameter['analyticSig']

    # this controls if the upper part of the spectrum is given or should be
    # reconctructed by 'mirroring' (flip and conjugate) of the lower spectrum
    if 'reconstMirror' not in parameter:
        if numBins == parameter['blockSize']:
            parameter['reconstMirror'] = False
        elif numBins < parameter['blockSize']:
            parameter['reconstMirror'] = True

    return parameter
