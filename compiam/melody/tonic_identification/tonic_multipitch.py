import os


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

    def extract(self, file_path):
        """Extract the tonic from a given file.

        :param file_path: path to file to extract.
        :returns: a floating point number representing the tonic of the input recording.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError("Target audio not found.")
        audio = estd.MonoLoader(filename=file_path)()
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
