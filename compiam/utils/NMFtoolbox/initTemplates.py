"""
    Name: initTemplates
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

from NMFtoolbox.midi2freq import midi2freq
from NMFtoolbox.utils import load_matlab_dict, EPS


def initTemplates(parameter, strategy='random'):
    """Implements different initialization strategies for NMF templates. The
    strategies 'random' and 'uniform' are self-explaining. The strategy
    'pitched' uses comb-filter templates as described in [2]. The strategy
    'drums' uses pre-extracted, averaged spectra of desired drum types [3].

    References
    ----------
    [2] Jonathan Driedger, Harald Grohganz, Thomas Prätzlich, Sebastian Ewert
    and Meinard Mueller "Score-informed audio decomposition and applications"
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
        numBins           Number of frequency bins
        numTemplateFrames Number of time frames for 2D-templates
        pitches           Optional array of MIDI pitch values
        drumTypes         Optional list of drum type strings

    strategy: str
        String describing the initialization strategy

    Returns
    -------
    initW: array-like
        List with the desired templates
    """
    # check parameters
    parameter = init_parameters(parameter)
    initW = list()

    if strategy == 'random':
        # fix random seed
        np.random.seed(0)

        for k in range(parameter['numComp']):
            initW.append(np.random.rand(parameter['numBins'], parameter['numTemplateFrames']))

    elif strategy == 'uniform':
        for k in range(parameter['numComp']):
            initW.append(np.ones((parameter['numBins'], parameter['numTemplateFrames'])))

    elif strategy == 'pitched':
        uniquePitches = np.unique(parameter['pitches'])

        # needs to be overwritten
        parameter['numComp'] = uniquePitches.size

        for k in range(uniquePitches.size):
            # initialize as zeros
            initW.append(EPS + np.zeros((parameter['numBins'], parameter['numTemplateFrames'])))

            # then insert non-zero entries in bands around hypothetic harmonics
            curPitchFreqLower_Hz = midi2freq(uniquePitches[k] - parameter['pitchTolDown'])
            curPitchFreqUpper_Hz = midi2freq(uniquePitches[k] + parameter['pitchTolUp'])

            for g in range(parameter['numHarmonics']):
                currPitchFreqLower_Bins = (g + 1) * curPitchFreqLower_Hz / parameter['deltaF']
                currPitchFreqUpper_Bins = (g + 1) * curPitchFreqUpper_Hz / parameter['deltaF']

                binRange = np.arange(int(round(currPitchFreqLower_Bins)) - 1, int(round(currPitchFreqUpper_Bins)))
                binRange = binRange[0:parameter['numBins']]

                # insert 1/f intensity
                initW[k][binRange, :] = 1/(g+1)

    elif strategy == 'drums':

        dictW = load_matlab_dict('../data/dictW.mat', 'dictW')

        if parameter['numBins'] == dictW.shape[0]:
            for k in range(dictW.shape[1]):
                initW.append(dictW[:, k].reshape(-1, 1) * np.linspace(1, 0.1, parameter['numTemplateFrames']))

        # needs to be overwritten
        parameter['numComp'] = len(initW)

    else:
        raise ValueError('Invalid strategy.')

    # do final normalization
    for k in range(parameter['numComp']):
        initW[k] /= (EPS + initW[k].sum())

    return initW


def init_parameters(parameter):
    """Auxiliary function to set the parameter dictionary

    Parameters
    ----------
    parameter: dict
        See the above function initTemplates for further information

    Returns
    -------
    parameter: dict
    """
    parameter['pitchTolUp'] = 0.75 if 'pitchTolUp' not in parameter else parameter['pitchTolUp']
    parameter['pitchTolDown'] = 0.75 if 'pitchTolDown' not in parameter else parameter['pitchTolDown']
    parameter['numHarmonics'] = 25 if 'numHarmonics' not in parameter else parameter['numHarmonics']
    parameter['numTemplateFrames'] = 1 if 'numTemplateFrames' not in parameter else parameter['numTemplateFrames']

    return parameter
