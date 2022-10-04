import os
import math

import numpy as np
import essentia.standard as estd

from compiam.melody.ftanet.model import FTANet
from compiam.melody.ftanet.pitch_processing import batchize_test, get_est_arr
from compiam.melody.ftanet.cfp import cfp_process

class FTANetCarnatic(object):
    """FTA-Net melody extraction tuned to Carnatic Music
    """
    def __init__(self, filepath):
        """ FTA-Net melody extraction init method
        :param model_path: path to file to the model weights
        """
        if not os.path.exists(filepath + '.data-00000-of-00001'):
            raise ValueError("""
                Given path to model weights not found. Make sure you enter the path correctly.
                A training process for the FTA-Net tuned to Carnatic is under development right
                now and will be added to the library soon. Meanwhile, we provide the weights in the
                latest repository version (https://github.com/MTG/compIAM) so make sure you have these
                available before loading the Carnatic FTA-Net.
            """)
        self.filepath = filepath
        self.model = FTANet().load_model()
        self.model.load_weights(filepath).expect_partial()

    def predict(self, path_to_audio, sample_rate=8000, hop_size=80, batch_size=5):
        """ Extract melody from filename
        :param filename: path to file to extract
        :param sample_rate: sample rate of extraction process
        :param hop_size: hop size between frequency estimations
        :param batch_size: batches of seconds that are passed through the model 
            (defaulted to 5, increase if enough computational power, reduce if
            needed)
        :returns: a 2-D list with time-stamps and pitch values per timestamp
        """
        xlist = []
        timestamps = []
        print('CFP process in {}'.format(path_to_audio))
        y, _ = estd.MonoLoader(path_to_audio, sampleRate=8000)()
        audio_len = len(y)
        batch_min = 8000*60*batch_size
        freqs = []
        if len(y) > batch_min:
            iters = math.ceil(len(y)/batch_min)
            for i in np.arange(iters):
                if i < iters-1:
                    audio_in = y[batch_min*i:batch_min*(i+1)]
                if i == iters-1:
                    audio_in = y[batch_min*i:]
                feature, _, time_arr = cfp_process(audio_in, sr=sample_rate, hop=hop_size)
                data = batchize_test(feature, size=128)
                xlist.append(data)
                timestamps.append(time_arr)

                estimation = get_est_arr(self.ftanet, xlist, timestamps, batch_size=16)
                if i == 0:
                    freqs = estimation[:, 1]
                else:
                    freqs = np.concatenate((freqs, estimation[:, 1]))
        else:
            feature, _, time_arr = cfp_process(y, sr=sample_rate, hop=hop_size)
            data = batchize_test(feature, size=128)
            xlist.append(data)
            timestamps.append(time_arr)
            # Getting estimatted pitch
            estimation = get_est_arr(self.ftanet, xlist, timestamps, batch_size=16)
            freqs = estimation[:, 1]
        TStamps = np.linspace(0, audio_len/sample_rate, len(freqs))

        ### TODO: Write code to re-sample in case sampling frequency is initialized different than 8k
        return np.array([TStamps, freqs]).transpose().toList()

    def normalise_pitch(pitch, tonic, bins_per_octave=120, max_value=4):
        """ Normalize pitch given a tonic
        :param pitch: a 2-D list with time-stamps and pitch values per timestamp
        :param tonic: recording tonic to normalize the pitch to
        :param bins_per_octave: number of frequency bins per octave
        :param max_value: maximum value to clip the normalized pitch to
        :returns: a 2-D list with time-stamps and normalized to a given tonic 
            pitch values per timestamp
        """
        return _normalise_pitch(
            pitch, tonic, bins_per_octave=bins_per_octave, max_value=max_value)


class Melodia:
    """ Melodia predominant melody extraction
    """
    def __init__(self, binResolution=10, filterIterations=3, frameSize=2048, guessUnvoiced=False,
                 harmonicWeight=0.8, hopSize=128, magnitudeCompression=1, magnitudeThreshold=40,
                 maxFrequency=20000, minDuration=100, minFrequency=80, numberHarmonics=20, 
                 peakDistributionThreshold=0.9, peakFrameThreshold=0.9, pitchContinuity=27.5625,
                 referenceFrequency=55, sampleRate=44100, timeContinuity=100, voiceVibrato=False,
                 voicingTolerance=0.2):
        """ Melodia predominant melody extraction init method
        For a complete and detailed list of the parameters see the documentation on the 
        following link: https://essentia.upf.edu/reference/std_PredominantPitchMelodia.html
        """
        self.binResolution = binResolution
        self.filterIterations = filterIterations
        self.frameSize = frameSize
        self.guessUnvoiced = guessUnvoiced
        self.harmonicWeight = harmonicWeight
        self.hopSize = hopSize
        self.magnitudeCompression = magnitudeCompression
        self.magnitudeThreshold = magnitudeThreshold
        self.maxFrequency = maxFrequency
        self.minDuration = minDuration
        self.minFrequency = minFrequency
        self.numberHarmonics = numberHarmonics
        self.peakDistributionThreshold = peakDistributionThreshold
        self.peakFrameThreshold = peakFrameThreshold
        self.pitchContinuity = pitchContinuity
        self.referenceFrequency = referenceFrequency
        self.sampleRate = sampleRate
        self.timeContinuity = timeContinuity
        self.voiceVibrato = voiceVibrato
        self.voicingTolerance = voicingTolerance

    def extract(self, filename):
        """ Extract the melody from a given file
        :param filename: path to file to extract
        :returns: a 2-D list with time-stamps and pitch values per timestamp
        """
        audio = estd.EqloudLoader(filename=filename)()
        extractor = estd.PredominantPitchMelodia(
            binResolution=self.binResolution,
            filterIterations=self.filterIterations,
            frameSize=self.frameSize, 
            guessUnvoiced=self.guessUnvoiced,
            harmonicWeight=self.harmonicWeight,
            hopSize=self.hopSize,
            magnitudeCompression=self.magnitudeCompression, 
            magnitudeThreshold=self.magnitudeThreshold,
            maxFrequency=self.maxFrequency,
            minDuration=self.minDuration,
            minFrequency=self.minFrequency,
            numberHarmonics=self.numberHarmonics,
            peakDistributionThreshold=self.peakDistributionThreshold,
            peakFrameThreshold=self.peakFrameThreshold,
            pitchContinuity=self.pitchContinuity,
            referenceFrequency=self.referenceFrequency,
            sampleRate=self.sampleRate,
            timeContinuity=self.timeContinuity,
            voiceVibrato=self.voiceVibrato,
            voicingTolerance=self.voicingTolerance)
        pitch, _ = extractor(audio)
        TStamps = np.array(range(0, len(pitch))) * np.float(self.parameters['hopSize']) / self.parameters['sampleRate']
        return np.array([TStamps, pitch]).transpose().toList()

    def normalise_pitch(pitch, tonic, bins_per_octave=120, max_value=4):
        """ Normalize pitch given a tonic
        :param pitch: a 2-D list with time-stamps and pitch values per timestamp
        :param tonic: recording tonic to normalize the pitch to
        :param bins_per_octave: number of frequency bins per octave
        :param max_value: maximum value to clip the normalized pitch to
        :returns: a 2-D list with time-stamps and normalized to a given tonic 
            pitch values per timestamp
        """
        return _normalise_pitch(
            pitch, tonic, bins_per_octave=bins_per_octave, max_value=max_value)


class TonicIndianMultiPitch:
    """ MultiPitch approach to extract the tonic from IAM music signals
    """
    def __init__(self, binResolution=10, frameSize=2048, harmonicWeight=0.8, hopSize=128,
                 magnitudeCompression=1, magnitudeThreshold=40, maxTonicFrequency=375,
                 minTonicFrequency=100, numberHarmonics=20, referenceFrequency=55, sampleRate=44100):
        """ Tonic extraction init method
        For a complete and detailed list of the parameters see the documentation on the 
        following link: https://essentia.upf.edu/reference/std_TonicIndianArtMusic.html
        """
        self.binResolution = binResolution
        self.frameSize = frameSize
        self.harmonicWeight = harmonicWeight
        self.hopSize = hopSize
        self.magnitudeCompression = magnitudeCompression
        self.magnitudeThreshold = magnitudeThreshold
        self.maxTonicFrequency = maxTonicFrequency
        self.minTonicFrequency = minTonicFrequency
        self.numberHarmonics = numberHarmonics
        self.referenceFrequency = referenceFrequency
        self.sampleRate = sampleRate

    def extract(self, filename):
        """ Extract the tonic from a given file
        :param filename: path to file to extract
        :returns: a floating point number representing the tonic of the input recording
        """
        audio = estd.MonoLoader(filename=filename)()
        extractor = estd.TonicIndianArtMusic(
            binResolution=self.binResolution,
            frameSize=self.frameSize,
            harmonicWeight=self.harmonicWeight,
            hopSize=self.hopSize,
            magnitudeCompression=self.magnitudeCompression, 
            magnitudeThreshold=self.magnitudeThreshold,
            maxTonicFrequency=self.maxTonicFrequency,
            minTonicFrequency=self.minTonicFrequency,
            numberHarmonics=self.numberHarmonics,
            referenceFrequency=self.referenceFrequency,
            sampleRate=self.sampleRate)
        return extractor(audio)


###############
# Melody utils
###############
def _normalise_pitch(pitch, tonic, bins_per_octave=120, max_value=4):
    """ Normalize pitch given a tonic
    :param pitch: a 2-D list with time-stamps and pitch values per timestamp
    :param tonic: recording tonic to normalize the pitch to
    :param bins_per_octave: number of frequency bins per octave
    :param max_value: maximum value to clip the normalized pitch to
    :returns: a 2-D list with time-stamps and normalized to a given tonic 
        pitch values per timestamp
    """
    pitch_values = pitch[:, 1]
    eps = np.finfo(np.float).eps
    normalised_pitch = bins_per_octave * np.log2(2.0 * (pitch_values + eps) / tonic)
    indexes = np.where(normalised_pitch <= 0)
    normalised_pitch[indexes] = 0
    indexes = np.where(normalised_pitch > max_value)
    normalised_pitch[indexes] = max_value
    return np.array([pitch[:, 0], normalised_pitch]).transpose().toList()
