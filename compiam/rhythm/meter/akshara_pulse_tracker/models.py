#!/usr/bin/env python
# Copyright 2022 Music Technology Group - Universitat Pompeu Fabra
#
# This file is was adapted from Dunya
#
"""
Originally created on Sep 12, 2013
@author: Ajay Srinivasamurthy
"""

import os
import math
import librosa

import numpy as np

import scipy.stats as scistats
import scipy.signal as scisig
from scipy.fft import fft

from compiam.rhythm.meter.akshara_pulse_tracker import parameters as params
from compiam.utils import get_logger

logger = get_logger(__name__)

# Main things to do
# 1. Segment audio into alapana and kriti: Look at the Mridangam base
# 2. Extract onset features from audio - using essentia
# 2. Estimate the peaks - peak detector xx
# 3. Compute the Tempogram xx
# 4. Estimate the matra/akshara period, tempo tracking - DP based xx
# 5. Track the matra/akshara - Transition matrix and DP based xx

########################################
class cust_pool:
    def __init__(self):
        self.values = {}

    def add(self, name, value):
        if name in self.values:
            self.values[name] = np.append(self.values[name], value)
        else:
            self.values[name] = np.array([value])


##########################################################################################
def smoothNovelty(onsetenv, pd, verbose=True):
    # Smooth beat events
    if np.mod(pd, 2):
        pd = pd + 1
        if verbose:
            logger.info("pd is odd, adding one to pd to make it even")
    ppd = (np.arange(-pd, pd + 1)).astype(float)
    templt = np.exp(-0.5 * (np.power(ppd / (pd / 32.0), 2.0)))
    localscore = np.convolve(templt, onsetenv)
    inds = np.round(templt.size / 2.0) + range(onsetenv.size)
    inds = inds.astype(int)
    localscore = localscore[inds]
    return localscore


##########################################################################################
def hanning(n):
    window = 0.5 - 0.5 * np.cos(2 * math.pi * np.arange(n) / (n - 1))
    return window


###########################################################################################
def normMax(y, alpha=0):
    z = y.copy()
    z = z / (np.max(np.abs(y)) + alpha)
    return z


###########################################################################################
def normalizeFeature(featMat, normP):
    """Assuming that each column is a feature vector, the normalization is along the columns"""
    featNorm = np.zeros(featMat.shape).astype(complex)
    T = featMat.shape[1]
    for t in range(T):
        n = np.linalg.norm(featMat[:, t], normP)
        featNorm[:, t] = featMat[:, t] / n
    return featNorm


###########################################################################################
def compute_fourierCoefficients(s, win, noverlap, f, fs, verbose=True):
    winLen = win.size
    hopSize = winLen - noverlap
    T = np.arange(0, winLen, 1) / fs
    winNum = int(np.round(np.fix((s.size - noverlap) / (winLen - noverlap))))
    x = np.zeros((int(winNum), int(f.size)))
    x = x.astype(complex)
    t = np.arange(winLen / 2.0, s.size - winLen / 2.0, hopSize) / fs
    twoPiT = 2.0 * math.pi * T

    for f0 in range(f.size):
        twoPiFt = f[f0] * twoPiT
        cosine_fn = np.cos(twoPiFt)
        sine_fn = np.sin(twoPiFt)

        for w in range(winNum):
            start = w * hopSize.astype(int)
            stop = start + winLen
            sig = s[range(start, stop)] * win
            co = (sig * cosine_fn).sum()
            si = (sig * sine_fn).sum()
            x[w, f0] = co + 1j * si
        if not np.mod(f0, 100):
            if verbose:
                logger.info(str(f0) + "/" + str(f.size) + "...")
    x = x.transpose()
    return x, f, t


############################################################################################
def tempogram_viaDFT(fn, tempoWindow, featureRate, stepsize, BPM, verbose=True):
    if verbose:
        logger.info("Computing Tempogram...")
    winLen = np.round(tempoWindow * featureRate)
    winLen = winLen + np.mod(winLen, 2) - 1
    window = hanning(winLen)
    ggk = np.zeros(int(np.round(winLen / 2)))
    novelty = np.append(ggk, fn.copy())
    novelty = np.append(novelty, ggk)
    TG, BPM, T = compute_fourierCoefficients(
        novelty, window, winLen - stepsize, BPM / 60.0, featureRate, verbose
    )
    BPM = BPM * 60.0
    T = T - T[0]
    tempogram = TG / math.sqrt(winLen) / sum(window) * winLen
    return tempogram, T, BPM


###########################################################################################
def findpeaks(x, imode="q", pmode="p", wdTol=5, ampTol=0.1, prominence=3.0):
    """
    x needs to be a ndarray column vector
    """
    nx = x.shape[0]
    nxp = nx + 1
    #     if x.shape[1] != 1:
    #         print("Error in input dimension input X. It must be a column vector")
    if pmode == "v":
        x = -x
    dx = x[1:] - x[0:-1]
    (r,) = (dx > 0).nonzero()
    r = r + 1  # 1 indexed
    (f,) = (dx < 0).nonzero()
    f = f + 1  # To make it 1 indexed and equivalent to MATLAB code
    if r.any() and f.any():
        dr = r[:].copy()
        dr[1:] = r[1:] - r[0:-1]
        rc = np.ones(nxp)
        rc[r] = 1 - dr
        rc[0] = 0
        rs = rc.cumsum(axis=0)  # = time since the last rise

        df = f[:].copy()
        df[1:] = f[1:] - f[0:-1]
        fc = np.ones(nxp)
        fc[f] = 1 - df  # import aksharaPulseTrack as ap
        fc[0] = 0
        fs = fc.cumsum(axis=0)  # = time since the last fall

        rp = -np.ones(nxp)
        arr = np.append([0], r.copy())
        dval = np.zeros(1)
        dval = nx - r[-1] - 1
        rrp = dr.copy() - 1
        rrp = np.append(rrp, dval)
        rp[arr] = rrp.copy()
        rq = rp.cumsum(axis=0)  # = time to the next rise

        fp = -np.ones(nxp)
        arr = np.append([0], f.copy())
        dval = np.zeros(1)
        dval = nx - f[-1] - 1
        ffp = df.copy() - 1
        ffp = np.append(ffp, dval)
        fp[arr] = ffp.copy()
        fq = fp.cumsum(axis=0)

        kk = (rs < fs) & (fq < rq) & (np.floor((fq - rs) / 2.0) == 0)
        (k,) = kk.nonzero()
        v = x[k]

        # Purge nearby peaks
        if wdTol > 0:
            kk = (k[1:] - k[0:-1]) <= wdTol
            (j,) = kk.nonzero()
            while any(j):
                (jj,) = (v[j] >= v[j + 1]).nonzero()
                jp = np.zeros(j.size)
                jp[jj] = 1
                jp = jp.astype(int)
                j = j + jp
                k = np.delete(k, [j])
                v = np.delete(v, [j])
                kk = (k[1:] - k[0:-1]) <= wdTol
                (j,) = kk.nonzero()
            pass
        pass

        # Also purge peaks with low prominence andind[k] = I small amplitude
        chosenIndices = np.zeros(v.shape)
        zz = x[-1].copy()
        diffx = np.diff(np.append(x, zz))

        for p in range(len(v)):  # Find prominent peaks and remove non-prominent ones
            pIndex = k[p]
            (lLim,) = (diffx[0 : pIndex - 1] < 0).nonzero()
            (uLim,) = (diffx[pIndex:] > 0).nonzero()
            if (not uLim.size) or (not lLim.size):
                chosenIndices[p] = 1
            else:
                lLim = min(lLim[-1] + 1, nx - 1)
                uLim = min(uLim[0] + pIndex, nx - 1)
                if (abs(v[p] / x[lLim]) > prominence) or (
                    abs(v[p] / x[uLim]) > prominence
                ):
                    chosenIndices[p] = 1
            # A prominent high vakue peak needs to be retained
            if abs(v[p]) > 0.3 * max(x):
                chosenIndices[p] = 1
            # Do an amplitude threshold and remove tiny peaks
            if abs(v[p]) < ampTol:
                chosenIndices[p] = 0

        (ch,) = chosenIndices.nonzero()
        v = v[ch]
        k = k[ch]
        if imode == "q":
            kf = k.copy()
            kf = kf.astype(float)
            b = 0.5 * (x[k + 1] - x[k - 1])
            a = x[k] - b - x[k - 1]
            (j,) = (a > 0).nonzero()  # j=0 on a plateau
            (jtilde,) = (a <= 0).nonzero()  # j=0 on a plateau
            v[j] = x[k[j]] + 0.25 * np.power(b[j], 2.0) / a[j]
            kf[j] = k[j] + 0.5 * b[j] / a[j]
            kf[jtilde] = k[jtilde] + (fq[k[jtilde]] - rs[k[jtilde]]) / 2.0
            # add 0.5 to k if plateau has an even width
            k = kf
    else:  # else for (r.any() and f.any())
        k = np.array([])
        v = np.array([])
    pass
    if pmode == "v":
        v = -v
        # Invert peaks if searching for valleys
    return k, v


##################################################################################
def getNearestIndex(inp, t):
    """
    % INPUT:
    % inp = input scalar
    % t = reference array
    % OUTPUT:
    % ind = returns the nearest index of t that has value closest to inp
    % In other words, ind(p) is the index of t such that inp(p) - t(ind(p)) is
    % minimum
    """
    ind = (np.abs(t - inp)).argmin()
    return ind.astype(int)


#################################################################################
def getNearestIndices(inp, t):
    """
    % INPUT:
    % inp = input array
    % t = reference array
    % OUTPUT:
    % ind = returns the nearest index of t that has value closest to inp
    % In other words, ind(p) is the index of t such that inp(p) - t(ind(p)) is
    % minimum
    """
    ind = np.zeros(inp.size)
    for k in range(inp.size):
        ind[k] = (np.abs(t - inp[k])).argmin()
    return ind.astype(int)


#################################################################################
def getTempoCurve(tg, BPM, minBPM, octTol, theta, delta, verbose=True):
    """
    % Given the tempogram, it returns the best tempo curve using DP
    % tempoGram is a DxN Tempogram matrix computed with D tempo values and N time instances
    % params - parameter structure with following fields
    % BPM (Dx1) = BPM at which the tempogram was computed. Note that BPM(1)
    % corresponds to tempoGram(1,:)
    % ts (1xN) = time stamps at which the tempogram was computed
    % theta (1x1) = smoothing parameter, higher the value, more smooth
    % is the curve. Set theta = 0 for a maximum vote tempo across time
    """
    if verbose:
        logger.info("Computing IAI curve...")
    (D, N) = tg.shape
    DP = np.zeros(tg.shape)
    pIndex = np.zeros(tg.shape)
    DP[:, 0] = tg[:, 0]
    (ind,) = (BPM < minBPM).nonzero()
    tg[ind, :] = -1000000.0
    BPMCOl = (BPM).reshape(((BPM).size, 1))
    xx = np.arange(-octTol, octTol + 1)
    penalGaussian = scistats.norm.pdf(xx, 0, octTol / 3)
    penalGaussian = penalGaussian.reshape((penalGaussian.size, 1))
    penalGaussian = penalGaussian / (penalGaussian.max())
    for i in range(1, N):
        for jj in range(D):
            # Octave jumps to be penalized
            penalArray = np.zeros((D, 1))
            # Lower octave first
            if BPM[jj] / 2.0 > BPM[0]:
                octLow = getNearestIndex(BPM[jj] / 2.0, BPM)
                if octLow <= octTol:
                    gg = np.arange(octLow + octTol)
                    gg = gg.astype(int)
                    penalArray[gg, :] = penalGaussian[-gg.size :]
                else:
                    gg = np.arange(octLow - octTol, octLow + octTol + 1)
                    gg = gg.astype(int)
                    penalArray[gg, :] = penalGaussian
            # Higher octave now
            if BPM[jj] / 2.0 < BPM[0]:
                octHigh = getNearestIndex(BPM[jj] * 2.0, BPM)
                if octHigh >= ((BPM).size - octTol):
                    gg = np.arange(octHigh - octTol, D + 1)
                    gg = gg.astype(int)
                    penalArray[gg, :] = penalGaussian[gg]
                else:
                    gg = np.arange(octHigh - octTol, octHigh + octTol + 1)
                    gg = gg.astype(int)
                    penalArray[gg, :] = penalGaussian
            # Now the DP
            bpmnow = np.abs(BPM[jj] - BPMCOl)
            fnNow = (
                (DP[:, i - 1]).reshape((DP[:, i - 1]).size, 1)
                - theta * bpmnow
                - delta * penalArray
            )
            #             fnNow = (DP[:,i-1]).reshape((DP[:,i-1]).size,1) - theta * bpmnow
            DP[jj, i] = fnNow.max()
            pIndex[jj, i] = fnNow.argmax()
            DP[jj, i] = DP[jj, i] + tg[jj, i]
        if not np.mod(i, 100):
            pass
            logger.info(str(i) + "/" + str(N) + "...")
    # backtracking now
    tc = np.zeros(N)
    zn = (np.zeros(N)).astype(int)
    zn[N - 1] = (DP[:, -1]).argmax()
    tc[N - 1] = BPM[zn[N - 1]]
    for p in range(N - 1, 0, -1):
        zn[p - 1] = pIndex[zn[p], p]
        tc[p - 1] = BPM[zn[p - 1]]
    # Return
    return tc


##################################################################################
def isScaleRelated(a, b, tol):
    a = float(a)
    b = float(b)
    if b > a:  # Swap
        temp = a
        a = b
        b = temp
    if abs((round(a / b) - (a / b))) < tol:
        return True
    else:
        return False


##################################################################################
def correctOctaveErrors(x, per, tol, verbose=True):
    if verbose:
        logger.info("Correcting octave errors in IAI estimation...")
    y = x.copy()
    flag = np.zeros(x.size)
    for k in range(x.size):
        if (np.abs(x[k] - per) / per) > tol:
            if isScaleRelated(x[k], per, tol / 2.0):
                if x[k] > per:
                    scale = np.round(x[k] / per)
                    y[k] = x[k] / scale
                    flag[k] = 1.0
                else:
                    scale = np.round(per / x[k])
                    y[k] = x[k] * scale
                    flag[k] = 1.0
            else:
                y[k] = per
                flag[k] = -1
            pass
        pass
    pass
    return y, flag


class AksharaPulseTracker:
    """Akshara onset detection"""

    def __init__(
        self,
        Nfft=4096,
        frmSize=1024,
        Fs=44100,
        hop=512,
        fBands=np.array(
            [
                [10, 110],
                [110, 500],
                [500, 3000],
                [3000, 5000],
                [5000, 10000],
                [0, 22000],
            ]
        ),
        songLenMin=600,
        octCorrectParam=0.25,
        tempoWindow=8,
        stepSizeTempogram=0.5,
        BPM=np.arange(40, 600.4, 0.5),
        minBPM=120,
        octTol=20,
        theta=0.005,
        delta=pow(10, 6),
        maxLen=0.6,
        binWidth=10e-3,
        thres=0.05,
        ignoreTooClose=0.6,
        decayCoeff=15,
        backSearch=[5.0, 0.5],
        alphaDP=3,
        smoothTime=2560,
        pwtol=0.2,
    ):

        self.Nfft = Nfft
        self.frmSize = frmSize
        self.Fs = Fs
        self.hop = hop
        self.fBands = fBands
        self.songLenMin = songLenMin
        self.octCorrectParam = octCorrectParam
        self.tempoWindow = tempoWindow
        self.stepSizeTempogram = stepSizeTempogram
        self.BPM = BPM
        self.minBPM = minBPM
        self.octTol = octTol
        self.theta = theta
        self.delta = delta
        self.maxLen = maxLen
        self.binWidth = binWidth
        self.thres = thres
        self.ignoreTooClose = ignoreTooClose
        self.decayCoeff = decayCoeff
        self.backSearch = backSearch
        self.alphaDP = alphaDP
        self.smoothTime = smoothTime
        self.pwtol = pwtol

        # Deduce other parameters
        self.fTicks = np.arange(Nfft / 2 + 1) * Fs / Nfft
        self.numBands = fBands.shape[0]
        self.frmHop = float(hop) / Fs
        self.pdSmooth = round(self.frmHop * smoothTime)
        self.featureRate = 1 / self.frmHop
        self.stepSize = round(stepSizeTempogram / self.frmHop)
        self.Nbins = maxLen / binWidth + 1
        self.wtolHistAv = round(20e-3 / binWidth)

    def extract(self, file_path, verbose=True):
        """Run extraction of akshara pulses from input audio file

        :param file_path: path to audio for extraction
        :param verbose: verbose level

        :returns: array of akshara pulses
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError("Target audio not found.")

        # Get onset functions
        onsFns = self.getOnsetFunctions(
            file_path,
            self.Nfft,
            self.frmSize,
            self.Fs,
            self.fTicks,
            self.hop,
            self.numBands,
            self.fBands,
            verbose,
        )
        onsFn = onsFns[:, 6].copy()
        onsTs = onsFns[:, 0].copy()
        onsFnLow = onsFns[:, 1].copy()
        onsFn = normMax(smoothNovelty(onsFn, self.pdSmooth, verbose))
        onsFnLow = normMax(smoothNovelty(onsFnLow, self.pdSmooth, verbose))
        sectStart = np.array([0.0])
        sectEnd = np.array([])

        # Find if segmentation is needed
        if onsTs[-1] > params.songLenMin:
            offsetIndex = self.getKritiStartBoundary(onsFnLow, onsTs, verbose)
            offsetTime = onsTs[offsetIndex] - onsTs[0]
            sectStart = np.append(sectStart, [offsetTime])
            sectEnd = np.append(sectEnd, onsTs[offsetIndex] - onsTs[0])
        else:
            offsetIndex = 0
            offsetTime = onsTs[offsetIndex] - onsTs[0]
        onsFn = onsFn[offsetIndex:]
        onsTs = onsTs[offsetIndex:]
        sectEnd = np.append(sectEnd, onsTs[-1])

        # Construct tempogram
        TG, TCts, BPM = tempogram_viaDFT(
            onsFn.copy(),
            self.tempoWindow,
            self.featureRate,
            self.stepSize,
            self.BPM,
            verbose,
        )
        TG = np.abs(normalizeFeature(TG, 2))

        # Estimate the tempo curve - IAI curve
        TCRaw = getTempoCurve(
            TG.copy(), BPM, self.minBPM, self.octTol, self.theta, self.delta, verbose
        )
        TCperRaw = 60.0 / TCRaw

        # Estimate akshara/matra period
        mmpFromTC = self.getMatraPeriodEstimateFromTC(
            TCperRaw, self.Nbins, self.minBPM, self.wtolHistAv, verbose
        )
        TCper, TCcorrFlag = correctOctaveErrors(
            TCperRaw, mmpFromTC, self.octCorrectParam, verbose
        )
        TC = 60.0 / TCper

        # Candidate estimation
        (
            akCandLocs,
            akCandTs,
            akCandWts,
            akCandTransMat,
        ) = self.estimateAksharaCandidates(
            onsTs,
            onsFn.copy(),
            TCper,
            TCts,
            mmpFromTC,
            self.pwtol,
            self.thres,
            self.ignoreTooClose,
            self.decayCoeff,
            verbose,
        )
        Locs = akCandLocs
        ts = akCandTs
        Wts = akCandWts
        TransMat = akCandTransMat
        TransMatCausal = np.triu(akCandTransMat + akCandTransMat.transpose())
        pers = TCper[getNearestIndices(akCandTs, TCts)]

        # Candidate selection
        aksharaLocs, aksharaTimes = self.DPSearch(
            TransMatCausal, ts, pers, Locs, Wts, self.backSearch, self.alphaDP, verbose
        )

        # Correct for all the offsets now and save to file
        aksharaTimes = aksharaTimes + offsetTime
        TCts = TCts + offsetTime

        return aksharaTimes

    def getOnsetFunctions(
        self, file_path, Nfft, frmSize, Fs, fTicks, hop, numBands, fBands, verbose=True
    ):
        zeropadLen = Nfft - frmSize
        zz = np.zeros((zeropadLen,), dtype="float32")

        frameCounter = 0
        bufferFrame = np.zeros(
            round(Nfft / 2 + 1),
        )
        if verbose:
            logger.info("Reading audio file...")

        # audio = ess.MonoLoader(filename=file_path)()
        audio, _ = librosa.load(file_path, sr=Fs)

        # fft = ess.FFT(size=Nfft)  # this gives us a complex FFT
        # c2p = ess.CartesianToPolar()  # and this turns it into a pair (magnitude, phase)

        pool = cust_pool()
        fTicks = fTicks
        poolName = "features.flux"

        if verbose:
            logger.info("Extracting Onset functions...")

        for i in range(audio.shape[0]):
            frame = audio[i * hop : i * hop + frmSize]
            if len(frame) < frmSize:
                break
            frmTime = hop / Fs * frameCounter + frmSize / (2.0 * Fs)
            zpFrame = np.hstack((frame, zz))
            hammFrame = np.hamming(len(zpFrame)) * zpFrame
            spectrum = fft(hammFrame)
            l = round(len(spectrum) / 2 + 1)
            mag = np.abs(spectrum)[:l]
            phase = np.angle(spectrum)[:l]

            magFlux = mag - bufferFrame
            bufferFrame = np.copy(mag)  # Copying for the next iteration to compute flux
            for bands in range(numBands):
                chosenInd = (fTicks >= fBands[bands, 0]) & (fTicks <= fBands[bands, 1])
                magFluxBand = magFlux[chosenInd]
                magFluxBand = (magFluxBand + abs(magFluxBand)) / 2
                oFn = magFluxBand.sum()
                if math.isnan(oFn):
                    if verbose:
                        logger.warning("NaN found here")
                pass
                pool.add(poolName + str(bands), oFn)
            pass
            pool.add("features.time", frmTime)
            frameCounter += 1
            if not np.mod(frameCounter, 10000):
                pass
                if verbose:
                    logger.info(str(frameCounter) + "/" + str(audio.size / hop) + "...")
        if verbose:
            logger.info("Total frames processed = " + str(frameCounter))
        timeStamps = pool.values["features.time"]
        all_feat = timeStamps
        for bands in range(numBands):
            feat_flux = [pool.values[poolName + str(bands)]]
            all_feat = np.vstack((all_feat, feat_flux))
        pass
        return np.transpose(all_feat)

    def getKritiStartBoundary(self, onsFn, onsTs, verbose=True):
        if verbose:
            logger.info("Obtaining Start of piece boundary...")
        peakLocs, peakVals = findpeaks(
            onsFn, imode="n", pmode="p", wdTol=0, ampTol=0.4, prominence=10.0
        )
        offsetIndex = peakLocs[0]
        offsetIndex = getNearestIndex(
            onsTs[offsetIndex] - 3.0, onsTs
        )  # Start three seconds
        return offsetIndex

    def getMatraPeriodEstimateFromTC(
        self, TCper, Nbins, minBPM, wtolHistAv, verbose=True
    ):
        if verbose:
            logger.info("Computing akshara pulse period...")
        histFn, binEdges = np.histogram(TCper, int(Nbins), (0.0, 60.0 / minBPM))
        binCentres = np.zeros(histFn.size)
        for p in range(histFn.size):
            binCentres[p] = (binEdges[p] + binEdges[p + 1]) / 2.0
        wtol = int(wtolHistAv)
        peaks, peakVals = findpeaks(
            histFn, imode="n", pmode="p", wdTol=wtol + 1.0, ampTol=0.0, prominence=1e-6
        )
        sortInd = np.argsort(-peakVals)
        topHistBins = peaks[sortInd]
        (kk,) = (topHistBins > (histFn.size - wtol - 1)).nonzero()
        topHistBins = np.delete(topHistBins, kk)
        (kk,) = (topHistBins < (wtol + 1)).nonzero()
        topHistBins = np.delete(topHistBins, kk)
        topHistBins = (topHistBins[0]).copy()
        topHistBins = topHistBins.astype(int)
        indRange = range(topHistBins - wtol - 1, topHistBins + wtol)
        wdw = (binCentres[indRange]).copy()
        wdwWts = (histFn[indRange]).copy()
        wdwWts = wdwWts.astype(float)
        wdwWts = wdwWts / wdwWts.sum()
        matraEst = (wdw * wdwWts).sum()
        return matraEst

    def estimateAksharaCandidates(
        self,
        tstamps,
        onsFn,
        TCper,
        TCts,
        medIAI,
        pwtol,
        thres,
        ignoreTooClose,
        decayCoeff,
        verbose=True,
    ):
        if verbose:
            logger.info("Estimating akshara candidates...")
        ts = tstamps[1] - tstamps[0]
        medIAISamp = medIAI / ts
        wtolPeaks = np.floor(pwtol * medIAISamp)
        peakLocs, peakVals = findpeaks(
            onsFn, imode="q", pmode="p", wdTol=wtolPeaks, ampTol=thres
        )
        # We get time ordered peaks
        Npeaks = peakLocs.size
        tPeaks = ts * peakLocs
        transMat = np.zeros((Npeaks, Npeaks))
        akPers = np.round(TCper[getNearestIndices(tPeaks, TCts)] / ts)
        for k in range(peakLocs.size):
            distVals = np.abs(peakLocs[k] - peakLocs) / akPers[k]
            farAwayParam = np.floor(distVals)
            dtVals = distVals - farAwayParam
            (iind,) = (dtVals > 0.5).nonzero()
            dtVals[iind] = 1.0 - dtVals[iind]
            (closeIndices,) = (
                np.abs(peakLocs[k] - peakLocs) < ignoreTooClose * medIAISamp
            ).nonzero()
            farAwayParam[closeIndices] = 100.0  # Arbitrarily large
            transMat[k, :] = np.exp(
                -(farAwayParam - 1) / decayCoeff
            ) * scistats.norm.pdf(dtVals, 0, 0.1)
            transMat[k, :] = transMat[k, :] / (transMat[k, :]).sum()
        pass
        akCandLocs = peakLocs
        akCandTs = tPeaks
        akCandWts = peakVals
        akCandTransMat = transMat
        return akCandLocs, akCandTs, akCandWts, akCandTransMat

    def DPSearch(
        self, TransMatCausal, ts, pers, Locs, Wts, backSearch, alphaDP, verbose=True
    ):
        if verbose:
            logger.info("Searching through candidates...")
        TM = (TransMatCausal).copy()
        ts = (ts).copy()
        pers = (pers).copy()
        Locs = (Locs).copy()
        D, N = TM.shape
        if D != N:
            if verbose:
                logger.warning("Transition Matrix not square!!!!")
            return -1
        backlink = -np.ones(Locs.size)
        cumscore = (Wts).copy()
        startIndex = getNearestIndex(backSearch[0] * pers.max(), ts) + 1

        for t in range(startIndex, cumscore.size):
            startSearch = getNearestIndex(ts[t] - pers[t] * backSearch[0], ts)
            endSearch = getNearestIndex(ts[t] - pers[t] * backSearch[1], ts)
            timerange = range(startSearch, endSearch + 1)
            scorecands = cumscore[timerange] + alphaDP * TM[timerange, t]
            # CAUTION, See the 100!
            val = scorecands.max()
            Ind = scorecands.argmax()
            cumscore[t] = val + Wts[t]
            backlink[t] = timerange[Ind]
        pass
        # Backtrace
        aksharaLocs = np.array([cumscore.argmax()]).astype(int)
        while (backlink[int(aksharaLocs[0])] > 0) and (aksharaLocs.size < N):
            aksharaLocs = np.append(backlink[int(aksharaLocs[0])], aksharaLocs)
        aksharaLocs = aksharaLocs.astype(int)
        aksharaTimes = ts[aksharaLocs]
        return aksharaLocs, aksharaTimes
