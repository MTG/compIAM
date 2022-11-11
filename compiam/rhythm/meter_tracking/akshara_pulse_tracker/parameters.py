# Copyright 2013,2014 Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of Dunya
#
# Dunya is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free Software
# Foundation (FSF), either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see http://www.gnu.org/licenses/

"""
Created on Sep 12, 2013

@author: Ajay Srinivasamurthy
A file with parameters for rhythm extraction
"""
import collections as coll
import math

import numpy as np

# Frame parameters first
Fs = 44100
hop = 512
frmSize = 1024
Nfft = 4096
zeropadLen = Nfft - frmSize
fBands = np.array(
    [[10, 110], [110, 500], [500, 3000], [3000, 5000], [5000, 10000], [0, 22000]]
)
numBands = fBands.shape[0]
fTicks = np.arange(Nfft / 2 + 1) * Fs / Nfft
songLenMin = 600
roundOffLen = 3
# Init all the named tuples
onsParams = coll.namedtuple(
    "onsParams",
    "frmHop frmLen pdSmooth wtol pkProm thresE maxLen binWidth wtolHistAv Nbins minLen",
)
# Fill in the values now
onsParams.frmHop = float(hop) / Fs  # hard coded, put a check while processing
onsParams.frmLen = float(frmSize) / Fs  # hard coded, check while processing
# Smoothing the onset functions
smoothTime = 80 * 32
onsParams.pdSmooth = round(
    onsParams.frmHop * smoothTime
)  # Smooths over about 10 samples ~ 100ms
onsParams.wtol = math.floor(
    50e-3 / onsParams.frmHop
)  # Peak finding on onset functions peaks within 50 ms are ignored
onsParams.pkProm = 3  # Peak prominence to choose as onsets
onsParams.thresE = 0.05  # Bottom 0.05% of peaks are ignored
onsParams.maxLen = 0.6  # Seconds
onsParams.binWidth = 10e-3
onsParams.wtolHistAv = round(20e-3 / onsParams.binWidth)
onsParams.Nbins = onsParams.maxLen / onsParams.binWidth + 1
onsParams.minLen = 0.1  # Second
# Novelty function parameters
novelParams = coll.namedtuple("novelParams", "frmLen")
novelParams.frmLen = 1024 / Fs  # hard coded, put a check while processing
# Tempogram parameters
TPGen = coll.namedtuple("TPGen", "NormP MinE")
TPGen.NormP = 2
TPGen.MinE = 1e-4
TGons = coll.namedtuple("TGons", "params")
TGons.params = coll.namedtuple("params", "tempoWindow BPM featureRate stepsize")
TGons.params.tempoWindow = 8  # second
stepSizeTempogram = 0.5  # second
TGons.params.BPM = np.arange(40, 600.4, 0.5)  # 0.1 second to 1.3 second
TGons.params.featureRate = 1 / onsParams.frmHop
TGons.params.stepsize = round(stepSizeTempogram / onsParams.frmHop)
"""
TGons.params.ACF.minLag = 0.05  # Second
TGons.params.ACF.maxLag = 2   # Second
TGons.params.ACF.featureRate = 1.0/onsParams.frmHop
TGons.params.ACF.normalization = 'unbiased'
TGons.params.ACF.tempoWindow = 6.0
TGons.params.ACF.stepsize = round(stepSizeTempogram/onsParams.frmHop)
"""
# Tempo curve parameters
TCparams = coll.namedtuple(
    "TCparams",
    "theta BPM binWidth wtolHistAv Nbins minBPM delta octTol smoothParam octCorrectParam",
)
TCparams.theta = 0.005  # Tradeoff between continuity and strength Higher the value, more the stress on continuity
TCparams.BPM = TGons.params.BPM
TCparams.binWidth = onsParams.binWidth
TCparams.wtolHistAv = round(20e-3 / onsParams.binWidth)
TCparams.Nbins = onsParams.maxLen / onsParams.binWidth + 1
TCparams.minBPM = 120  # Below this no pulse will be found
TCparams.delta = pow(
    10, 6
)  # The octave jump tradeoff parameter Higher the value, ocatve jumps are penalized more
TCparams.octTol = 20  # 10 BPM is the octave tolerance to prevent octave jumps
TCparams.smoothParam = round(5 / stepSizeTempogram)
TCparams.octCorrectParam = 0.25
# PLP parameters
# PLP.params.as = 90
# Tempo tolerance parameters
tolParam = coll.namedtuple("tolParam", "tempoLow tempoHigh octaveErr")
tolParam.tempoLow = 0.9
tolParam.tempoHigh = 1.1
tolParam.octaveErr = 1e-1
# Sama Candidate estimation parameters
samaParams = coll.namedtuple(
    "samaParams",
    "pwtol numSeedPeaks numCandPerWindow srchWtolLow srchWtolHigh tolLow tolHigh thres decayCoeff theta ignoreTooClose backSearch alphaDP Nperiods maxRetry",
)
samaParams.pwtol = 0.1  # Peaks within 1% of the ISI are ignored when computing peaks
samaParams.numSeedPeaks = 3  # Seed peak number
samaParams.numCandPerWindow = 5  # Per big window
samaParams.srchWtolLow = 0.2
samaParams.srchWtolHigh = 2.2
samaParams.tolLow = 0.95  # Candidate period tolerances
samaParams.tolHigh = 1.05  # Tolerance
samaParams.thres = 0.05  #
samaParams.decayCoeff = 15  # Probability decay coefficient
samaParams.theta = 0.1  #
samaParams.ignoreTooClose = 0.6  # Close ignore
samaParams.backSearch = 10  # Number of previous peaks to search for
samaParams.alphaDP = 2  # Alpha weighting parameter
samaParams.Nperiods = 3  # No. of periods to search starting from anchor
samaParams.maxRetry = 10  # Not used
# Akshara tracking parameters
aksharaParams = coll.namedtuple(
    "aksharaParams",
    "pwtol numSeedPeaks numCandPerWindow srchWtolLow srchWtolHigh tolLow tolHigh thres decayCoeff theta ignoreTooClose backSearch alphaDP",
)
aksharaParams.pwtol = (
    0.2  # Peaks within 20% of the IAI are ignored when computing peaks
)
aksharaParams.numSeedPeaks = 3  # Number of seeds - Not used
aksharaParams.numCandPerWindow = 5  # Not used
aksharaParams.srchWtolLow = 0.2  # Unused
aksharaParams.srchWtolHigh = 2.2  # Unused
aksharaParams.tolLow = 0.2  #
aksharaParams.tolHigh = 2.2  #
aksharaParams.thres = 0.05  #
aksharaParams.decayCoeff = 15  #
aksharaParams.theta = 0.1  #
aksharaParams.ignoreTooClose = 0.6  #
aksharaParams.backSearch = [5.0, 0.5]  # Number of previous periods to search for
aksharaParams.alphaDP = 3  # Alpha weighting parameter
"""
% Beat Tracking parameters
beatParams.pwtol = 0.2 % Peaks within 20% of the IAI are ignored when computing peaks
beatParams.numSeedPeaks = 3
beatParams.numCandPerWindow = 5
beatParams.srchWtolLow = 0.2
beatParams.srchWtolHigh = 2.2
beatParams.tolLow = 0.2
beatParams.tolHigh = 2.2
beatParams.thres = 0.05
beatParams.decayCoeff = 10
beatParams.theta = 0.1
beatParams.ignoreTooClose = 0.6
beatParams.backSearch = [5 0.5]     % Number of previous peaks to search for
beatParams.alphaDP = 20    % Alpha weighting parameter
"""

# Also adding some structures to use in the future!
akCands = coll.namedtuple("akCands", "Locs ts Wts TransMat TransMatCausal pers")
# Fill in the values now
