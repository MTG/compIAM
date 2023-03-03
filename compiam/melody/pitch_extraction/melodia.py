import os

import numpy as np

from compiam.utils.pitch import normalisation, resampling
from compiam.io import write_csv
from compiam.utils import get_logger

logger = get_logger(__name__)


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

    def extract(self, input_data, input_sr=44100, out_step=None):
        """Extract the melody from a given file.

        :param input_data: path to audio file or numpy array like audio signal
        :param input_sr: sampling rate of the input array of data (if any). This variable is only
            relevant if the input is an array of data instead of a filepath.
        :param out_step: particular time-step duration if needed at output
        :returns: a 2-D list with time-stamps and pitch values per timestamp.
        """
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError("Target audio not found.")
            audio = estd.EqloudLoader(filename=input_data, sampleRate=self.sampleRate)()
        elif isinstance(input_data, np.ndarray):
            logger.warning(
                f"Resampling... (input sampling rate is {input_sr}Hz, make sure this is correct)"
            )
            resample_audio = estd.Resample(
                inputSampleRate=input_sr, outputSampleRate=self.sampleRate
            )()
            input_data = resample_audio(input_data)
            audio = estd.EqualLoudness(signal=input_data)()
        else:
            raise ValueError("Input must be path to audio signal or an audio array")

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
        output = np.array([TStamps, pitch]).transpose()

        if out_step is not None:
            new_len = int((len(audio) / self.sampleRate) // out_step)
            return resampling(output, new_len)

        return output

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

    @staticmethod
    def save_pitch(data, output_path):
        """Calling the write_csv function in compiam.io to write the output pitch curve in a fle

        :param data: the data to write
        :param output_path: the path where the data is going to be stored

        :returns: None
        """
        return write_csv(data, output_path)
