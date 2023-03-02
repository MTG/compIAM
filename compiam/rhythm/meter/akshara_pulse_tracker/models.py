# Copyright 2023 Music Technology Group - Universitat Pompeu Fabra
#
# This file is was adapted from Dunya
#
"""
Originally created on Sep 12, 2013
@author: Ajay Srinivasamurthy
"""

import math

import numpy as np
import scipy.stats as scistats

from compiam.utils import get_logger

logger = get_logger(__name__)


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
