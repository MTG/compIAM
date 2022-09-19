"""
    Name: initActivations
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
import numpy as np

from NMFtoolbox.NEMA import NEMA
from NMFtoolbox.utils import EPS


def initActivations(parameter, strategy):
    """Implements different initialization strategies for NMF activations. The
    strategies 'random' and 'uniform' are self-explaining. The strategy
    'pitched' places gate-like activations at the frames, where certain notes
    are active in the ground truth transcription [2]. The strategy
    'drums' places decaying impulses at the frames where drum onsets are given
    in the ground truth transcription [3].

    References
    ----------
    [2] Jonathan Driedger, Harald Grohganz, Thomas Prätzlich, Sebastian Ewert
    and Meinard Müller "Score-informed audio decomposition and applications"
    In Proceedings of the ACM International Conference on Multimedia (ACM-MM)
    Barcelona, Spain, 2013.

    [3] Christian Dittmar and Meinard Müller -- Reverse Engineering the Amen
    Break - Score-informed Separation and Restoration applied to Drum
    Recordings" IEEE/ACM Transactions on Audio, Speech, and Language Processing,
    24(9): 1531-1543, 2016.

    Parameters
    ----------
    parameter: dict
        numComp           Number of NMF components
        numFrames         Number of time frames
        deltaT            The temporal resolution
        pitches           Optional array of MIDI pitch values
        onsets            Optional array of note onsets (in seconds)
        durations         Optional array of note durations (in seconds)
        drums             Pptional array of drum type indices
        decay             Optional array of decay values per component
        onsetOffsetTol    Optional parameter giving the onset / offset

    strategy: str
        String describing the intialization strategy

    Returns
    -------
    initH: array-like
        Array with initial activation functions
    """

    parameter = init_parameters(parameter)

    if strategy == 'random':
        np.random.seed(0)
        initH = np.random.rand(parameter['numComp'], parameter['numFrames'])

    elif strategy == 'uniform':
        initH = np.ones((parameter['numComp'], parameter['numFrames']))

    elif strategy == 'pitched':
        uniquePitches = np.unique(parameter['pitches'])

        # overwrite
        parameter['numComp'] = uniquePitches.size

        # initialize activations with very small values
        initH = EPS + np.zeros((parameter['numComp'], parameter['numFrames']))

        for k in range(uniquePitches.size):

            # find corresponding note onsets and durations
            ind = np.nonzero(parameter['pitches'] == uniquePitches[k])[0]

            # insert activations
            for g in range(len(ind)):
                currInd = ind[g]

                noteStartInSeconds = parameter['onsets'][currInd]
                noteEndeInSeconds = noteStartInSeconds + parameter['durations'][currInd]

                noteStartInSeconds -= parameter['onsetOffsetTol']
                noteEndeInSeconds += parameter['onsetOffsetTol']

                noteStartInFrames = int(round(noteStartInSeconds / parameter['deltaT']))
                noteEndeInFrames = int(round(noteEndeInSeconds / parameter['deltaT']))

                frameRange = np.arange(noteStartInFrames, noteEndeInFrames + 1)
                frameRange = frameRange[frameRange >= 0]
                frameRange = frameRange[frameRange <= parameter['numFrames']]

                # insert gate-like activation
                initH[k, frameRange-1] = 1

    elif strategy == 'drums':
        uniqueDrums = np.unique(parameter['drums'])

        # overwrite
        parameter['numComp'] = uniqueDrums.size

        # initialize activations with very small values
        initH = EPS + np.zeros((parameter['numComp'], parameter['numFrames']))

        # sanity check
        if uniqueDrums.size == parameter['numComp']:

            # insert impulses at onset positions
            for k in range(len(uniqueDrums)):
                currOns = np.nonzero(parameter['drums'] == uniqueDrums[k])[0]
                currOns = parameter['onsets'][currOns]
                currOns = np.round(currOns/parameter['deltaT']).astype(np.int)
                currOns = currOns[currOns >= 0]
                currOns = currOns[currOns <= parameter['numFrames']]

                initH[uniqueDrums[k].astype(int)-1, currOns-1] = 1

            # add exponential decay
            initH = NEMA(initH, parameter['decay'])

    else:
        raise ValueError('Invalid strategy.')

    return initH


def init_parameters(parameter):
    """Auxiliary function to set the parameter dictionary

    Parameters
    ----------
    parameter: dict
        See the above function initActivations for further information

    Returns
    -------
    parameter: dict
    """
    parameter['decay'] = 0.75 if 'decay' not in parameter else parameter['decay']
    parameter['onsetOffsetTol'] = 0.025 if 'onsetOffsetTol' not in parameter else parameter['onsetOffsetTol']

    return parameter