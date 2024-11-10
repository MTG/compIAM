import os
import librosa
import numpy as np
from tqdm import tqdm

from compiam.exceptions import ModelNotTrainedError

# Shared functions with FTANetCarnatic for vocals
from compiam.utils.pitch import normalisation, resampling
from compiam.melody.pitch_extraction.ftanet_carnatic.pitch_processing import (
    est,
    std_normalize,
)
from compiam.melody.pitch_extraction.ftanet_carnatic.cfp import cfp_process

from compiam.io import write_csv
from compiam.utils import get_logger, WORKDIR
from compiam.utils.download import download_remote_model

logger = get_logger(__name__)


class FTAResNetCarnatic(object):
    """FTA-ResNet melody extraction tuned to Carnatic Music."""

    def __init__(
        self,
        model_path=None,
        download_link=None,
        download_checksum=None,
        sample_rate=44100,
        gpu="-1",
    ):
        """FTA-Net melody extraction init method.

        :param model_path: path to file to the model weights.
        :param download_link: link to the remote pre-trained model.
        :param download_checksum: checksum of the model file.
        :param sample_rate: Sample rate to which the audio is sampled for extraction.
        """
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global torch
            import torch

            global FTAnet
            from compiam.melody.pitch_extraction.ftaresnet_carnatic.model import FTAnet

        except:
            raise ImportError(
                "In order to use this tool you need to have torch installed. "
                "Install compIAM with torch support: pip install 'compiam[torch]'"
            )
        ###

        ## Setting up GPU if specified
        self.gpu = gpu
        self.device = None
        self.select_gpu(gpu)

        self.model = self._build_model()
        self.sample_rate = sample_rate
        self.trained = False

        self.model_path = model_path
        self.download_link = download_link
        self.download_checksum = download_checksum
        if self.model_path is not None:
            self.load_model(self.model_path)

        self.sample_rate = sample_rate

    def _build_model(self):
        """Build the FTA-Net model."""
        ftanet = FTAnet().to(self.device)
        ftanet.eval()
        return ftanet

    def load_model(self, model_path):
        """Load pre-trained model weights."""
        if not os.path.exists(model_path):
            self.download_model(model_path)  # Downloading model weights
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model_path = model_path
        self.trained = True

    def download_model(self, model_path=None, force_overwrite=False):
        """Download pre-trained model."""
        download_path = (
            os.sep + os.path.join(*model_path.split(os.sep)[:-2])
            if model_path is not None
            else os.path.join(WORKDIR, "models", "melody", "ftaresnet_carnatic")
        )
        # Creating model folder to store the weights
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        download_remote_model(
            self.download_link,
            self.download_checksum,
            download_path,
            force_overwrite=force_overwrite,
        )

    def predict(
        self,
        input_data,
        input_sr=44100,
        hop_size=441,
        time_frame=128,
        out_step=None,
        gpu="-1",
    ):
        """Extract melody from input_data.
        Implementation taken (and slightly adapted) from https://github.com/yushuai/FTANet-melodic.

        :param input_data: path to audio file or numpy array like audio signal.
        :param input_sr: sampling rate of the input array of data (if any). This variable is only
            relevant if the input is an array of data instead of a filepath.
        :param hop_size: hop size between frequency estimations.
        :param batch_size: batches of seconds that are passed through the model
            (defaulted to 5, increase if enough computational power, reduce if
            needed).
        :param out_step: particular time-step duration if needed at output
        :param gpu: Id of the available GPU to use (-1 by default, to run on CPU), use string: '0', '1', etc.
        :returns: a 2-D list with time-stamps and pitch values per timestamp.
        """
        ## Setting up GPU if any
        if gpu != self.gpu:
            self.select_gpu(gpu)

        if self.trained is False:
            raise ModelNotTrainedError(
                """Model is not trained. Please load model before running inference!
                You can load the pre-trained instance with the load_model wrapper."""
            )

        # Loading and resampling audio
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError("Target audio not found.")
            audio, _ = librosa.load(input_data, sr=self.sample_rate)
        elif isinstance(input_data, np.ndarray):
            logger.warning(
                f"Resampling... (input sampling rate is assumed {input_sr}Hz, \
                    make sure this is correct and change input_sr otherwise)"
            )
            audio = librosa.resample(
                input_data, orig_sr=input_sr, target_sr=self.sample_rate
            )
        else:
            raise ValueError("Input must be path to audio signal or an audio array")

        audio_shape = audio.shape
        if len(audio_shape) > 1:
            audio_channels = min(audio_shape)
            if audio_channels == 1:
                audio = audio.flatten()
            else:
                audio = np.mean(audio, axis=np.argmin(audio_shape))

        # Extracting pitch
        audio_len = len(audio) // hop_size
        with torch.no_grad():
            prediction_pitch = torch.zeros(321, audio_len).to(self.device)
            for i in tqdm(range(0, audio_len, time_frame)):
                W, Cen_freq, _ = cfp_process(
                    y=audio[i * hop_size : (i + time_frame) * hop_size + 1],
                    sr=self.sample_rate,
                    hop=hop_size,
                )
                value = W.shape[-1]
                W = np.concatenate(
                    (W, np.zeros((3, 320, 128 - W.shape[-1]))), axis=-1
                )  # Padding
                W_norm = std_normalize(W)
                w = np.stack((W_norm, W_norm))
                prediction_pitch[:, i : i + value] = self.model(
                    torch.Tensor(w).to(self.device)
                )[0][0][0][:, :value]

        # Convert to Hz
        frame_time = hop_size / self.sample_rate
        y_hat = est(
            prediction_pitch.to("cpu"),
            Cen_freq,
            torch.linspace(
                0,
                frame_time * ((audio_len // time_frame) * time_frame),
                ((audio_len // time_frame) * time_frame),
            ),
        )

        # Filter low frequency values
        freqs = y_hat[:, 1]
        freqs[freqs < 50] = 0

        # Format output
        TStamps = y_hat[:, 0]
        output = np.array([TStamps, freqs]).transpose()

        if out_step is not None:
            new_len = int((len(audio) / self.sample_rate) // out_step)
            output = resampling(output, new_len)
            return output

        return output

    @staticmethod
    def normalise_pitch(pitch, tonic, bins_per_octave=120, max_value=4):
        """Normalise pitch given a tonic.

        :param pitch: a 2-D list with time-stamps and pitch values per timestamp.
        :param tonic: recording tonic to normalize the pitch to.
        :param bins_per_octave: number of frequency bins per octave.
        :param max_value: maximum value to clip the normalized pitch to.
        :returns: a 2-D list with time-stamps and normalised to a given tonic
            pitch values per timestamp.
        """
        return normalisation(
            pitch, tonic, bins_per_octave=bins_per_octave, max_value=max_value
        )

    @staticmethod
    def save_pitch(data, output_path):
        """Calling the write_csv function in compiam.io to write the output pitch curve in a file

        :param data: the data to write
        :param output_path: the path where the data is going to be stored

        :returns: None
        """
        return write_csv(data, output_path)

    def select_gpu(self, gpu="-1"):
        """Select the GPU to use for inference.

        :param gpu: Id of the available GPU to use (-1 by default, to run on CPU), use string: '0', '1', etc.
        :returns: None
        """
        if int(gpu) == -1:
            self.device = torch.device("cpu")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:" + str(gpu))
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps:" + str(gpu))
            else:
                self.device = torch.device("cpu")
                logger.warning("No GPU available. Running on CPU.")
        self.gpu = gpu
