import torch
import torchaudio


class PreProcessor(torch.nn.Module):
    """Log-mel spectrogram used as input of the beat tracker."""

    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        hop_length=441,
        f_min=30,
        f_max=11000,
        n_mels=128,
        mel_scale="slaney",
        normalized="frame_length",
        power=1,
        log_multiplier=1000,
    ):
        """Pre-processor init method.

        :param sample_rate: sampling rate of the signals to process (in Hz).
        :param n_fft: size of the FFT (in samples).
        :param hop_length: hop size between frames (in samples).
        :param f_min: minimum frequency of the mel bands (in Hz).
        :param f_max: maximum frequency of the mel bands (in Hz).
        :param n_mels: number of mel bands.
        :param mel_scale: scale of the mel bands.
        :param normalized: normalisation of the STFT.
        :param power: exponent of the magnitude spectrogram.
        :param log_multiplier: magnitudes are multiplied by this value before the log(1 + x) compression.
        """
        super().__init__()
        self.spect_class = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            mel_scale=mel_scale,
            normalized=normalized,
            power=power,
        )
        self.log_multiplier = log_multiplier

    def forward(self, x):
        """Compute the features of a mono signal.

        :param x: 1-D array or tensor with the audio signal, sampled at `sample_rate`.

        :returns: tensor of shape (frames, n_mels) with the features.
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        return torch.log1p(self.log_multiplier * self.spect_class(x).T)
