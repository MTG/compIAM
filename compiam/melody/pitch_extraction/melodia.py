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
        bin_resolution=10,
        filter_iterations=3,
        frame_size=2048,
        guess_unvoiced=False,
        harmonic_weight=0.8,
        hop_size=128,
        magnitude_compression=1,
        magnitude_threshold=40,
        max_frequency=20000,
        min_duration=100,
        min_frequency=80,
        num_harmonics=20,
        peak_distribution_threshold=0.9,
        peak_frame_threshold=0.9,
        pitch_continuity=27.5625,
        reference_frequency=55,
        sample_rate=44100,
        time_continuity=100,
        voice_vibrato=False,
        voicing_tolerance=0.2,
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
                "Install compIAM with essentia support: pip install 'compiam[essentia]'"
            )
        ###

        self.bin_resolution = bin_resolution
        self.filter_iterations = filter_iterations
        self.frame_size = frame_size
        self.guess_unvoiced = guess_unvoiced
        self.harmonic_weight = harmonic_weight
        self.hop_size = hop_size
        self.magnitude_compression = magnitude_compression
        self.magnitude_threshold = magnitude_threshold
        self.max_frequency = max_frequency
        self.min_duration = min_duration
        self.min_frequency = min_frequency
        self.num_harmonics = num_harmonics
        self.peak_distribution_threshold = peak_distribution_threshold
        self.peak_frame_threshold = peak_frame_threshold
        self.pitch_continuity = pitch_continuity
        self.reference_frequency = reference_frequency
        self.sample_rate = sample_rate
        self.time_continuity = time_continuity
        self.voice_vibrato = voice_vibrato
        self.voicing_tolerance = voicing_tolerance

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
            audio = estd.EqloudLoader(
                filename=input_data, sampleRate=self.sample_rate
            )()
        elif isinstance(input_data, np.ndarray):
            logger.warning(
                f"Resampling... (input sampling rate is {input_sr}Hz, make sure this is correct)"
            )
            resample_audio = estd.Resample(
                inputSampleRate=input_sr, outputSampleRate=self.sample_rate
            )()
            input_data = resample_audio(input_data)
            audio = estd.EqualLoudness(signal=input_data)()
        else:
            raise ValueError("Input must be path to audio signal or an audio array")

        extractor = estd.PredominantPitchMelodia(
            binResolution=self.bin_resolution,
            filterIterations=self.filter_iterations,
            frameSize=self.frame_size,
            guessUnvoiced=self.guess_unvoiced,
            harmonicWeight=self.harmonic_weight,
            hopSize=self.hop_size,
            magnitudeCompression=self.magnitude_compression,
            magnitudeThreshold=self.magnitude_threshold,
            maxFrequency=self.max_frequency,
            minDuration=self.min_duration,
            minFrequency=self.min_frequency,
            numberHarmonics=self.num_harmonics,
            peakDistributionThreshold=self.peak_distribution_threshold,
            peakFrameThreshold=self.peak_frame_threshold,
            pitchContinuity=self.pitch_continuity,
            referenceFrequency=self.reference_frequency,
            sampleRate=self.sample_rate,
            timeContinuity=self.time_continuity,
            voiceVibrato=self.voice_vibrato,
            voicingTolerance=self.voicing_tolerance,
        )
        pitch, _ = extractor(audio)
        TStamps = (
            np.array(range(0, len(pitch))) * float(self.hop_size) / self.sample_rate
        )
        output = np.array([TStamps, pitch]).transpose()

        if out_step is not None:
            new_len = int((len(audio) / self.sample_rate) // out_step)
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
