import math
import numpy as np
import pandas as pd 

import librosa

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

import essentia.standard as estd

from scipy.ndimage import gaussian_filter1d

from compiam.melody.pattern.sancara_search.extraction.sequence import get_stability_mask, add_center_to_mask
from compiam.melody.pattern.sancara_search.extraction.io import get_timeseries, write_timeseries

def pitch_to_cents(p, tonic):
    """
    Convert pitch value, <p> to cents above <tonic>.

    :param p: Pitch value in Hz
    :type p: float
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Pitch value, <p> in cents above <tonic>
    :rtype: float
    """
    return 1200*math.log(p/tonic, 2) if p else None


def cents_to_pitch(c, tonic):
    """
    Convert cents value, <c> to pitch in Hz

    :param c: Pitch value in cents above <tonic>
    :type c: float/int
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Pitch value, <c> in Hz 
    :rtype: float
    """
    return (2**(c/1200))*tonic


def pitch_seq_to_cents(pseq, tonic):
    """
    Convert sequence of pitch values to sequence of 
    cents above <tonic> values

    :param pseq: Array of pitch values in Hz
    :type pseq: np.array
    :param tonic: Tonic value in Hz
    :type tonic: float

    :return: Sequence of original pitch value in cents above <tonic>
    :rtype: np.array
    """
    return np.vectorize(lambda y: pitch_to_cents(y, tonic))(pseq)


def interpolate_below_length(arr, val, gap):
    """
    Interpolate gaps of value, <val> of 
    length equal to or shorter than <gap> in <arr>
    
    :param arr: Array to interpolate
    :type arr: np.array
    :param val: Value expected in gaps to interpolate
    :type val: number
    :param gap: Maximum gap length to interpolate, gaps of <val> longer than <g> will not be interpolated
    :type gap: number

    :return: interpolated array
    :rtype: np.array
    """
    s = np.copy(arr)
    is_zero = s == val
    cumsum = np.cumsum(is_zero).astype('float')
    diff = np.zeros_like(s)
    diff[~is_zero] = np.diff(cumsum[~is_zero], prepend=0)
    for i,d in enumerate(diff):
        if d <= gap:
            s[int(i-d):i] = np.nan
    interp = pd.Series(s).interpolate(method='linear', axis=0)\
                         .ffill()\
                         .bfill()\
                         .values
    return interp


def extract_pitch_track(audio_path, frameSize, hopSize, gap_interp, smooth, sr):

    audio_loaded, _ = librosa.load(audio_path, sr=sr)

    # Run spleeter on track to remove the background
    separator = Separator('spleeter:2stems')
    audio_loader = AudioAdapter.default()
    waveform, _ = audio_loader.load(audio_path, sample_rate=sr)
    prediction = separator.separate(waveform=waveform)
    clean_vocal = prediction['vocals']

    # Prepare audio for pitch extraction
    audio_mono = clean_vocal.sum(axis=1) / 2
    audio_mono_eqloud = estd.EqualLoudness(sampleRate=sr)(audio_mono)

    # Extract pitch using Melodia algorithm from Essentia
    pitch_extractor = estd.PredominantPitchMelodia(frameSize=frameSize, hopSize=hopSize)
    raw_pitch, _ = pitch_extractor(audio_mono_eqloud)
    raw_pitch_ = np.append(raw_pitch, 0.0)
    time = np.linspace(0.0, len(audio_mono_eqloud) / sr, len(raw_pitch))

    timestep = time[4]-time[3] # resolution of time track

    # Gap interpolation
    if gap_interp:
        print(f'Interpolating gaps of {gap_interp} or less')
        raw_pitch = interpolate_below_length(raw_pitch_, 0, int(gap_interp/timestep))
        
    # Gaussian smoothing
    if smooth:
        print(f'Gaussian smoothing with sigma={smooth}')
        pitch = gaussian_filter1d(raw_pitch, smooth)
    else:
        pitch = raw_pitch[:]

    return pitch, raw_pitch, timestep, time


def silence_stability_from_file(inpath, outpath, tonic=None, min_stability_length_secs=1, stab_hop_secs=0.2, freq_var_thresh_stab=8, gap_interp=0.250):

    pitch, time, timestep = get_timeseries(inpath)
    pitch_interp = interpolate_below_length(pitch, 0, (gap_interp/timestep))

    print('Computing stability/silence mask')
    if tonic:
        pi = pitch_seq_to_cents(pitch_interp, tonic)
    else:
        pi = pitch_interp
    stable_mask = get_stability_mask(pi, min_stability_length_secs, stab_hop_secs, freq_var_thresh_stab, timestep)
    silence_mask = (pitch_interp == 0).astype(int)
    silence_mask = add_center_to_mask(silence_mask)
    silence_and_stable_mask = np.array([int(any([i,j])) for i,j in zip(silence_mask, stable_mask)])
    write_timeseries([time, silence_and_stable_mask], outpath)

