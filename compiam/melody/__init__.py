import math

import numpy as np
import essentia.standard as estd

from compiam.melody.ftanet.model import FTANet


class ftanetCarnatic:
    def __init__(self, filepath):
        self.filepath = filepath
        self.model = FTANet().load_model()
        self.model.load_weights(filepath)


    def predict(self, path_to_audio, sample_rate=8000, hop_size=80, batch_size=5):
        """Extract melody from filename
        Args:
            filename (str): path to file to extract
        """
        from ftanet.pitch_processing import batchize_test, get_est_arr
        from ftanet.cfp import cfp_process
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

    def normalise_pitch(self, pitch, tonic, bins_per_octave=120, max_value=4):
        """Normalize pitch given a tonic
        Args:
            pitch (list): list of pitch values and time-stamps
            tonic (float): TODO
            bins_per_octave (int): cents per bin
            max_value (int): TODO
        """
        pitch_values = pitch[:, 1]
        eps = np.finfo(np.float).eps
        normalised_pitch = bins_per_octave * np.log2(2.0 * (pitch_values + eps) / tonic)
        indexes = np.where(normalised_pitch <= 0)
        normalised_pitch[indexes] = 0
        indexes = np.where(normalised_pitch > max_value)
        normalised_pitch[indexes] = max_value
        return np.array([pitch[:, 0], normalised_pitch]).transpose().toList()


class Melodia:
    """Melodia predominant melody extraction
    """
    def __init__(self):
        """Melodia predominant melody extraction init method
        """
        self.parameters = {
            'binResolution': 10,
            'filterIterations': 3,
            'frameSize': 2048,
            'guessUnvoiced':  False,
            'harmonicWeight': 0.8,
            'hopSize': 128,
            'magnitudeCompression': 1,
            'magnitudeThreshold': 40,
            'maxFrequency': 20000,
            'minDuration': 100,
            'minFrequency': 80,
            'numberHarmonics': 20,
            'peakDistributionThreshold': 0.9,
            'peakFrameThreshold': 0.9,
            'pitchContinuity': 27.5625,
            'referenceFrequency': 55,
            'sampleRate': 44100,
            'timeContinuity': 100,
            'voiceVibrato': False,
            'voicingTolerance': 0.2}
        self.extractor = estd.PredominantPitchMelodia(
            binResolution=self.parameters['binResolution'],
            filterIterations=self.parameters['filterIterations'],
            frameSize=self.parameters['frameSize'], 
            guessUnvoiced=self.parameters['guessUnvoiced'],
            harmonicWeight=self.parameters['harmonicWeight'],
            hopSize=self.parameters['hopSize'],
            magnitudeCompression=self.parameters['magnitudeCompression'], 
            magnitudeThreshold=self.parameters['magnitudeThreshold'],
            maxFrequency=self.parameters['maxFrequency'],
            minDuration=self.parameters['minDuration'],
            minFrequency=self.parameters['minFrequency'], 
            numberHarmonics=self.parameters['numberHarmonics'],
            peakDistributionThreshold=self.parameters['peakDistributionThreshold'],
            peakFrameThreshold=self.parameters['peakFrameThreshold'], 
            pitchContinuity=self.parameters['pitchContinuity'],
            referenceFrequency=self.parameters['referenceFrequency'],
            sampleRate=self.parameters['sampleRate'], 
            timeContinuity=self.parameters['timeContinuity'],
            voiceVibrato=self.parameters['voiceVibrato'],
            voicingTolerance=self.parameters['voicingTolerance'])

    def get_parameters(self):
        """Returns the current parameters to run the method
        """
        return self.parameters
    
    def update_parameters(self, binResolution=10, filterIterations=3, frameSize=2048, guessUnvoiced=False, 
                          harmonicWeight=0.8, hopSize=128, magnitudeCompression=1, magnitudeThreshold=40,
                          maxFrequency=20000, minDuration=100, minFrequency=80, numberHarmonics=20, 
                          peakDistributionThreshold=0.9, peakFrameThreshold=0.9, pitchContinuity=27.5625,
                          referenceFrequency=55, sampleRate=44100, timeContinuity=100, voiceVibrato=False,
                          voicingTolerance=0.2):
        """Update the parameters and re-initialize method
        Args:
            Melodia parameters
        """
        self.parameters = {
            'binResolution': binResolution,
            'filterIterations': filterIterations,
            'frameSize': frameSize,
            'guessUnvoiced':  guessUnvoiced,
            'harmonicWeight': harmonicWeight,
            'hopSize': hopSize,
            'magnitudeCompression': magnitudeCompression,
            'magnitudeThreshold': magnitudeThreshold,
            'maxFrequency': maxFrequency,
            'minDuration': minDuration,
            'minFrequency': minFrequency,
            'numberHarmonics': numberHarmonics,
            'peakDistributionThreshold': peakDistributionThreshold,
            'peakFrameThreshold': peakFrameThreshold,
            'pitchContinuity': pitchContinuity,
            'referenceFrequency': referenceFrequency,
            'sampleRate': sampleRate,
            'timeContinuity': timeContinuity,
            'voiceVibrato': voiceVibrato,
            'voicingTolerance': voicingTolerance}
        self.extractor = estd.PredominantPitchMelodia(
            binResolution=self.parameters['binResolution'],
            filterIterations=self.parameters['filterIterations'],
            frameSize=self.parameters['frameSize'], 
            guessUnvoiced=self.parameters['guessUnvoiced'],
            harmonicWeight=self.parameters['harmonicWeight'],
            hopSize=self.parameters['hopSize'],
            magnitudeCompression=self.parameters['magnitudeCompression'], 
            magnitudeThreshold=self.parameters['magnitudeThreshold'],
            maxFrequency=self.parameters['maxFrequency'],
            minDuration=self.parameters['minDuration'],
            minFrequency=self.parameters['minFrequency'], 
            numberHarmonics=self.parameters['numberHarmonics'],
            peakDistributionThreshold=self.parameters['peakDistributionThreshold'],
            peakFrameThreshold=self.parameters['peakFrameThreshold'], 
            pitchContinuity=self.parameters['pitchContinuity'],
            referenceFrequency=self.parameters['referenceFrequency'],
            sampleRate=self.parameters['sampleRate'], 
            timeContinuity=self.parameters['timeContinuity'],
            voiceVibrato=self.parameters['voiceVibrato'],
            voicingTolerance=self.parameters['voicingTolerance'])

    def extract(self, filename):
        audio = estd.EqloudLoader(filename=filename)()
        pitch, _ = self.extractor(audio)
        TStamps = np.array(range(0, len(pitch))) * np.float(self.parameters['hopSize']) / self.parameters['sampleRate']
        return np.array([TStamps, pitch]).transpose().toList()


class TonicIndianMultiPitch:
    """MultiPitch approach to extract the tonic from IAM music signals
    """
    def __init__(self):
        """MultiPitch approach to extract the tonic from IAM music signals init method
        """
        self.parameters = {
            'binResolution': 10,
            'frameSize': 2048,
            'harmonicWeight': 0.8,
            'hopSize': 128,
            'magnitudeCompression': 1,
            'magnitudeThreshold': 40,
            'maxTonicFrequency': 375,
            'minTonicFrequency': 100,
            'numberHarmonics': 20,
            'referenceFrequency': 55,
            'sampleRate': 44100}
        self.extractor = estd.TonicIndianArtMusic(
            binResolution=self.parameters['binResolution'],
            frameSize=self.parameters['frameSize'], 
            harmonicWeight=self.parameters['harmonicWeight'],
            hopSize=self.parameters['hopSize'],
            magnitudeCompression=self.parameters['magnitudeCompression'], 
            magnitudeThreshold=self.parameters['magnitudeThreshold'],
            maxTonicFrequency=self.parameters['maxTonicFrequency'],
            minTonicFrequency=self.parameters['minTonicFrequency'],
            numberHarmonics=self.parameters['numberHarmonics'],
            referenceFrequency=self.parameters['referenceFrequency'],
            sampleRate=self.parameters['sampleRate'])

    def get_parameters(self):
        """Returns the current parameters to run the method
        """
        return self.parameters
    
    def update_parameters(self, binResolution=10, frameSize=2048, harmonicWeight=0.8, hopSize=128, 
                          magnitudeCompression=1, magnitudeThreshold=40, maxTonicFrequency=375,
                          minTonicFrequency=100, numberHarmonics=20, referenceFrequency=55, 
                          sampleRate=44100):
        """Update the parameters and re-initialize method
        Args:
            TonicIndianArtMusic parameters
        """
        self.parameters = {
            'binResolution': binResolution,
            'frameSize': frameSize,
            'harmonicWeight': harmonicWeight,
            'hopSize': hopSize,
            'magnitudeCompression': magnitudeCompression,
            'magnitudeThreshold': magnitudeThreshold,
            'maxTonicFrequency': maxTonicFrequency,
            'minTonicFrequency': minTonicFrequency,
            'numberHarmonics': numberHarmonics,
            'referenceFrequency': referenceFrequency,
            'sampleRate': sampleRate}
        self.extractor = estd.PredominantPitchMelodia(
            binResolution=self.parameters['binResolution'],
            frameSize=self.parameters['frameSize'], 
            harmonicWeight=self.parameters['harmonicWeight'],
            hopSize=self.parameters['hopSize'],
            magnitudeCompression=self.parameters['magnitudeCompression'], 
            magnitudeThreshold=self.parameters['magnitudeThreshold'],
            maxTonicFrequency=self.parameters['maxTonicFrequency'],
            minTonicFrequency=self.parameters['minTonicFrequency'],
            numberHarmonics=self.parameters['numberHarmonics'],
            referenceFrequency=self.parameters['referenceFrequency'],
            sampleRate=self.parameters['sampleRate'])

    def extract(self, filename):
        """Extract tonic from filename
        Args:
            filename (str): path to file to extract
        """
        audio = estd.MonoLoader(filename=filename)()
        return self.extractor(audio)
