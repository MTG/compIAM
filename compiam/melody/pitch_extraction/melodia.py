import os

import numpy as np

from compiam.utils.pitch import normalisation


class Melodia:
    """Melodia predominant melody extraction"""

    def __init__(
        self,
        binResolution=10,
        filterIterations=3,
        frameSize=2048,
        guessUnvoiced=False,
        harmonicWeight=0.8,
        hopSize=128,
        magnitudeCompression=1,
        magnitudeThreshold=40,
        maxFrequency=20000,
        minDuration=100,
        minFrequency=80,
        numberHarmonics=20,
        peakDistributionThreshold=0.9,
        peakFrameThreshold=0.9,
        pitchContinuity=27.5625,
        referenceFrequency=55,
        sampleRate=44100,
        timeContinuity=100,
        voiceVibrato=False,
        voicingTolerance=0.2,
    ):
        """Melodia predominant melody extraction init method
        For a complete and detailed list of the parameters see the documentation on the
        following link: https://essentia.upf.edu/reference/std_PredominantPitchMelodia.html
        """
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global estd
            import essentia.standard as estd
        except:
            raise ImportError(
                "In order to use this tool you need to have essentia installed. "
                "Please install essentia using: pip install essentia"
            )
        ###

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

    def extract(self, file_path):
        """Extract the melody from a given file.

        :param file_path: path to file to extract.
        :returns: a 2-D list with time-stamps and pitch values per timestamp.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError("Target audio not found.")
        audio = estd.EqloudLoader(filename=file_path)()
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
            voicingTolerance=self.voicingTolerance,
        )
        pitch, _ = extractor(audio)
        TStamps = np.array(range(0, len(pitch))) * float(self.hopSize) / self.sampleRate
        return np.array([TStamps, pitch]).transpose()

    @staticmethod
    def normalise_pitch(pitch, tonic, bins_per_octave=120, max_value=4):
        """Normalize pitch given a tonic.

        :param pitch: a 2-D list with time-stamps and pitch values per timestamp.
        :param tonic: recording tonic to normalize the pitch to.
        :param bins_per_octave: number of frequency bins per octave.
        :param max_value: maximum value to clip the normalized pitch to.
        :returns: a 2-D list with time-stamps and normalized to a given tonic
            pitch values per timestamp.
        """
        return normalisation(
            pitch, tonic, bins_per_octave=bins_per_octave, max_value=max_value
        )
