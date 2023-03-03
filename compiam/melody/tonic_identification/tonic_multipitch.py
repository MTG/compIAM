import os

import numpy as np

from compiam.utils import get_logger

logger = get_logger(__name__)


class TonicIndianMultiPitch:
    """MultiPitch approach to extract the tonic from IAM music signals."""

    def __init__(
        self,
        binResolution=10,
        frameSize=2048,
        harmonicWeight=0.8,
        hopSize=128,
        magnitudeCompression=1,
        magnitudeThreshold=40,
        maxTonicFrequency=375,
        minTonicFrequency=100,
        numberHarmonics=20,
        referenceFrequency=55,
        sampleRate=44100,
    ):
        """Tonic extraction init method.
        For a complete and detailed list of the parameters see the documentation on the
        following link: https://essentia.upf.edu/reference/std_TonicIndianArtMusic.html.
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

    def extract(self, input_data, input_sr=44100):
        """Extract the tonic from a given file.

        :param input_data: path to audio file or numpy array like audio signal
        :param input_sr: sampling rate of the input array of data (if any). This variable is only
            relevant if the input is an array of data instead of a filepath.
        :returns: a floating point number representing the tonic of the input recording.
        """
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError("Target audio not found.")
            audio = estd.MonoLoader(filename=input_data, sampleRate=self.sampleRate)()
        elif isinstance(input_data, np.ndarray):
            logger.warning(
                f"Resampling... (input sampling rate is {input_sr}Hz, make sure this is correct)"
            )
            resampling = estd.Resample(
                inputSampleRate=input_sr, outputSampleRate=self.sampleRate
            )()
            audio = resampling(input_data)
        else:
            raise ValueError("Input must be path to audio signal or an audio array")

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
            sampleRate=self.sampleRate,
        )
        return extractor(audio)
