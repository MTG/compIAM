import numpy as np


def log_frequencies(bands_per_octave, fmin, fmax, fref=440.0):
    """Frequencies of a logarithmically spaced grid, aligned to a reference frequency.

    :param bands_per_octave: number of bands per octave.
    :param fmin: minimum frequency (in Hz).
    :param fmax: maximum frequency (in Hz).
    :param fref: reference frequency (in Hz).

    :returns: array of frequencies (in Hz).
    """
    left = np.floor(np.log2(float(fmin) / fref) * bands_per_octave)
    right = np.ceil(np.log2(float(fmax) / fref) * bands_per_octave)
    frequencies = fref * 2.0 ** (np.arange(left, right) / float(bands_per_octave))
    frequencies = frequencies[np.searchsorted(frequencies, fmin) :]
    return frequencies[: np.searchsorted(frequencies, fmax, "right")]


def frequencies_to_bins(frequencies, bin_frequencies, unique_bins=False):
    """Map frequencies to the closest bins.

    :param frequencies: frequencies to map (in Hz).
    :param bin_frequencies: frequencies of the bins (in Hz).
    :param unique_bins: keep only unique bins.

    :returns: array of bin indices.
    """
    indices = bin_frequencies.searchsorted(frequencies)
    indices = np.clip(indices, 1, len(bin_frequencies) - 1)
    left = bin_frequencies[indices - 1]
    right = bin_frequencies[indices]
    indices -= frequencies - left < right - frequencies
    return np.unique(indices) if unique_bins else indices


def logarithmic_filterbank(
    bin_frequencies,
    bands_per_octave=12,
    fmin=30.0,
    fmax=17000.0,
    norm_filters=True,
    unique_filters=True,
):
    """Filterbank of overlapping triangular filters with logarithmically spaced center frequencies.

    :param bin_frequencies: frequencies of the spectrogram bins (in Hz).
    :param bands_per_octave: number of filters per octave.
    :param fmin: minimum frequency (in Hz).
    :param fmax: maximum frequency (in Hz).
    :param norm_filters: normalise the area of the filters to 1.
    :param unique_filters: remove filters that would be identical due to the limited frequency resolution.

    :returns: 2-D array (bins x bands) of filter weights.
    """
    bin_frequencies = np.asarray(bin_frequencies, dtype=np.float64)
    frequencies = log_frequencies(bands_per_octave, fmin, fmax)
    bins = frequencies_to_bins(frequencies, bin_frequencies, unique_filters)
    if len(bins) < 3:
        raise ValueError("Not enough bins to create a triangular filter.")

    filterbank = np.zeros((len(bin_frequencies), len(bins) - 2), dtype=np.float32)
    for band in range(len(bins) - 2):
        start, center, stop = bins[band : band + 3]
        if stop - start < 2:
            center = start
            stop = start + 1
        center -= start
        stop -= start
        data = np.zeros(stop)
        data[:center] = np.linspace(0, 1, center, endpoint=False)
        data[center:] = np.linspace(1, 0, stop - center, endpoint=False)
        data = data.astype(np.float32)
        if norm_filters:
            data /= np.sum(data)
        # filters that exceed the spectrum are cropped
        stop = min(start + len(data), len(bin_frequencies))
        filterbank[start:stop, band] = np.maximum(
            data[: stop - start], filterbank[start:stop, band]
        )
    return filterbank


class PreProcessor:
    """Log-filtered magnitude spectrogram of an audio signal, used as input of the TCN.

    The signal is framed at `fps` frames per second, windowed with a Hann window and transformed
    with an FFT. The magnitude is then filtered with a logarithmic filterbank and compressed with
    a logarithm.
    """

    def __init__(
        self,
        sample_rate=44100,
        frame_size=2048,
        num_bands=12,
        log=np.log,
        add=1e-6,
        fps=100,
    ):
        """Pre-processor init method.

        :param sample_rate: sampling rate of the signals to process (in Hz).
        :param frame_size: size of the analysis frames (in samples).
        :param num_bands: number of filters per octave.
        :param log: function to compress the magnitudes with.
        :param add: value added to the magnitudes before compressing them.
        :param fps: frames per second.
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.log = log
        self.add = add
        self.fps = fps
        self.hop_size = sample_rate / float(fps)
        self.window = np.hanning(frame_size)
        num_bins = frame_size >> 1
        bin_frequencies = np.arange(num_bins) * float(sample_rate) / frame_size
        self.filterbank = logarithmic_filterbank(bin_frequencies, num_bands)

    def __call__(self, signal, block_size=1024):
        """Compute the features of a mono signal.

        :param signal: 1-D array with the audio signal, in the range [-1, 1].
        :param block_size: number of frames processed at once.

        :returns: 2-D array (frames x bands) with the features.
        """
        signal = np.asarray(signal)
        if signal.ndim != 1:
            raise ValueError("Signal must be mono, got shape %s." % (signal.shape,))
        num_frames = int(np.ceil(len(signal) / self.hop_size))
        # frames are centered in their reference sample
        offset = self.frame_size // 2
        starts = (np.arange(num_frames) * self.hop_size).astype(np.int64) - offset
        padded = np.pad(signal, (offset, self.frame_size))
        features = np.empty((num_frames, self.filterbank.shape[1]), dtype=np.float32)
        for begin in range(0, num_frames, block_size):
            block = starts[begin : begin + block_size, np.newaxis] + offset
            frames = padded[block + np.arange(self.frame_size)]
            stft = np.fft.rfft(frames * self.window, axis=-1)
            magnitude = np.abs(stft[:, : self.frame_size >> 1].astype(np.complex64))
            features[begin : begin + block_size] = self.log(
                np.dot(magnitude, self.filterbank) + self.add
            )
        return features
