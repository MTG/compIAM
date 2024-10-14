# Copyright 2023 Music Technology Group - Universitat Pompeu Fabra
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

from scipy.fft import fft

from compiam.rhythm.meter.akshara_pulse_tracker.models import (
    cust_pool,
    smoothNovelty,
    normMax,
    normalizeFeature,
    tempogram_viaDFT,
    findpeaks,
    getNearestIndex,
    getNearestIndices,
    getTempoCurve,
    correctOctaveErrors,
)
from compiam.rhythm.meter.akshara_pulse_tracker import parameters as params
from compiam.utils import get_logger

logger = get_logger(__name__)


class AksharaPulseTracker:
    """Akshara onset detection. CompMusic Rhythm Extractor."""

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
        """Akshara onset detection init method"""
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

    def extract(self, input_data, input_sr=44100, verbose=True):
        """Run extraction of akshara pulses from input audio file

        :param input_data: path to audio file or numpy array like audio signal
        :param input_sr: sampling rate of the input array of data (if any). This variable is only
            relevant if the input is an array of data instead of a filepath
        :param verbose: verbose level

        :returns: dict containing estimation for sections, matra period, akshara pulses, and tempo curve
        """
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError("Target audio not found.")
            audio, _ = librosa.load(input_data, sr=self.Fs)
        elif isinstance(input_data, np.ndarray):
            logger.warning(
                f"Resampling... (input sampling rate is {input_sr}Hz, make sure this is correct)"
            )
            audio = librosa.resample(input_data, orig_sr=input_sr, target_sr=self.Fs)
        else:
            raise ValueError("Input must be path to audio signal or an audio array")

        # Get onset functions
        onsFns = self.getOnsetFunctions(
            audio,
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

        # Get sections
        sections = {"startTime": 0, "endTime": 0, "label": ""}
        sections["startTime"] = np.round(sectStart, params.roundOffLen).tolist()
        sections["endTime"] = np.round(sectEnd, params.roundOffLen).tolist()
        labelStr = ("Alapana", "Kriti")
        if sectEnd.size == 2:
            sections["label"] = labelStr
        else:
            sections["label"] = labelStr[1]

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
        TCper, _ = correctOctaveErrors(
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
        # TransMat = akCandTransMat
        TransMatCausal = np.triu(akCandTransMat + akCandTransMat.transpose())
        pers = TCper[getNearestIndices(akCandTs, TCts)]

        # Candidate selection
        _, aksharaPulses = self.DPSearch(
            TransMatCausal, ts, pers, Locs, Wts, self.backSearch, self.alphaDP, verbose
        )

        # Correct for all the offsets now
        aksharaPulses = aksharaPulses + offsetTime
        TCts = TCts + offsetTime

        APcurve = [[TCts[t], TCper[t]] for t in range(TCts.size)]

        return {
            "sections": sections,
            "aksharaPeriod": np.round(mmpFromTC, params.roundOffLen).item(0),
            "aksharaPulses": aksharaPulses.tolist(),
            "APcurve": APcurve,
        }

    def getOnsetFunctions(
        self, audio, Nfft, frmSize, Fs, fTicks, hop, numBands, fBands, verbose=True
    ):
        zeropadLen = Nfft - frmSize
        zz = np.zeros((zeropadLen,), dtype="float32")

        frameCounter = 0
        bufferFrame = np.zeros(
            round(Nfft / 2 + 1),
        )
        if verbose:
            logger.info("Reading audio file...")

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
