import os

import numpy as np

from compiam.utils import get_logger

logger = get_logger(__name__)


class TonicIndianMultiPitch:
    """MultiPitch approach to extract the tonic from IAM music signals."""

    def __init__(
        self,
        bin_resolution=10,
        frame_size=2048,
        harmonic_weight=0.8,
        hop_size=128,
        magnitude_compression=1,
        magnitude_threshold=40,
        max_tonic_frequency=375,
        min_tonic_frequency=100,
        num_harmonics=20,
        ref_frequency=55,
        sample_rate=44100,
    ):
        """Tonic extraction init method.
        For a complete and detailed list of the parameters see the documentation on the
        following link: https://essentia.upf.edu/reference/std_TonicIndianArtMusic.html.
        Naming convention of the arguments has been stadardized to compIAM-friendly format.
        """
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global estd
            import essentia.standard as estd
        except:
            raise ImportError(
                "In order to use this tool you need to have essentia installed. "
                "Install compIAM with essentia support: pip install 'compiam[essentia]'"
            )
        ###

        self.bin_resolution = bin_resolution
        self.frame_size = frame_size
        self.harmonic_weight = harmonic_weight
        self.hop_size = hop_size
        self.magnitude_compression = magnitude_compression
        self.magnitude_threshold = magnitude_threshold
        self.max_tonic_frequency = max_tonic_frequency
        self.min_tonic_frequency = min_tonic_frequency
        self.num_harmonics = num_harmonics
        self.ref_frequency = ref_frequency
        self.sample_rate = sample_rate

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
            audio = estd.MonoLoader(filename=input_data, sampleRate=self.sample_rate)()
        elif isinstance(input_data, np.ndarray):
            logger.warning(
                f"Resampling... (input sampling rate is {input_sr}Hz, make sure this is correct)"
            )
            resampling = estd.Resample(
                inputSampleRate=input_sr, outputSampleRate=self.sample_rate
            )()
            audio = resampling(input_data)
        else:
            raise ValueError("Input must be path to audio signal or an audio array")

        extractor = estd.TonicIndianArtMusic(
            binResolution=self.bin_resolution,
            frameSize=self.frame_size,
            harmonicWeight=self.harmonic_weight,
            hopSize=self.hop_size,
            magnitudeCompression=self.magnitude_compression,
            magnitudeThreshold=self.magnitude_threshold,
            maxTonicFrequency=self.max_tonic_frequency,
            minTonicFrequency=self.min_tonic_frequency,
            numberHarmonics=self.num_harmonics,
            referenceFrequency=self.ref_frequency,
            sampleRate=self.sample_rate,
        )
        return extractor(audio)
