import os
import numpy as np

from compiam.exceptions import ModelNotTrainedError

from compiam.utils.download import download_remote_model
from compiam.utils import get_logger, WORKDIR
from compiam.io import write_csv

logger = get_logger(__name__)


class TCNTracker(object):
    """TCN beat tracker tuned to Carnatic Music."""

    def __init__(
        self,
        post_processor="joint",
        model_version=42,
        model_path=None,
        download_link=None,
        download_checksum=None,
        gpu=-1,
    ):
        """TCN beat tracker init method.

        :param post_processor: Post-processing method to use. Choose from 'joint', or 'sequential'.
        :param model_version: Version of the pre-trained model to use. Choose from 42, 52, or 62.
        :param model_path: path to file to the model weights.
        :param download_link: link to the remote pre-trained model.
        :param download_checksum: checksum of the model file.
        :param gpu: Id of the available GPU to use (-1 by default, to run on CPU), use string: '0', '1', etc.
        """
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global torch
            import torch
        except ImportError:
            raise ImportError(
                "Torch is required to use TCNTracker. "
                "Install compIAM with torch support: pip install 'compiam[torch]'"
            )
        ###
        global MultiTracker, PreProcessor, joint_tracker, sequential_tracker
        from compiam.rhythm.meter.tcn_carnatic.model import MultiTracker
        from compiam.rhythm.meter.tcn_carnatic.pre import PreProcessor
        from compiam.rhythm.meter.tcn_carnatic.post import (
            joint_tracker,
            sequential_tracker,
        )

        if post_processor not in ["joint", "sequential"]:
            raise ValueError(
                f"Invalid post_processor: {post_processor}. Choose from 'joint', or 'sequential'."
            )
        if model_version not in [42, 52, 62]:
            raise ValueError(
                f"Invalid model_version: {model_version}. Choose from 42, 52, or 62."
            )

        self.gpu = gpu
        self.device = None
        self.select_gpu(gpu)

        self.model_path = model_path
        self.model_version = f"multitracker_{model_version}.pth"
        self.download_link = download_link
        self.download_checksum = download_checksum

        self.sample_rate = 44100
        self.fps = 100
        self.pad_frames = 2
        self.pre_processor = PreProcessor(sample_rate=self.sample_rate, fps=self.fps)

        self.trained = False
        self.model = self._build_model()
        if self.model_path is not None:
            self.load_model(self.model_path)

        self.post_processor = (
            joint_tracker if post_processor == "joint" else sequential_tracker
        )

    def _build_model(self):
        """Build the TCN model."""
        model = MultiTracker().to(self.device)
        model.eval()
        return model

    def load_model(self, model_path):
        """Load pre-trained model weights.

        :param model_path: path to the folder with the model weights.
        """
        if not os.path.exists(os.path.join(model_path, self.model_version)):
            self.download_model(model_path)  # Downloading model weights

        self.model.load_weights(
            os.path.join(model_path, self.model_version), self.device
        )

        self.model_path = model_path
        self.trained = True

    def download_model(self, model_path=None, force_overwrite=True):
        """Download pre-trained model.

        :param model_path: path to the folder to download the model weights to.
        :param force_overwrite: if True, overwrite existing files.
        """
        download_path = (
            model_path
            if model_path is not None
            else os.path.join(WORKDIR, "models", "rhythm", "tcn-carnatic")
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
        sr=44100,
        min_bpm=55,
        max_bpm=230,
        beats_per_bar=(3, 5, 7, 8),
    ):
        """Run inference on input audio file.

        :param input_data: path to audio file or numpy array like audio signal.
        :param sr: sampling rate of the input audio signal, only used if the input is an array (default: 44100).
        :param min_bpm: minimum BPM for beat tracking (default: 55).
        :param max_bpm: maximum BPM for beat tracking (default: 230).
        :param beats_per_bar: list of possible beats per bar for downbeat tracking (default: [3, 5, 7, 8]).

        :returns: a 2-D array with the beat times (in seconds) and their position in the bar.
        """
        if self.trained is False:
            raise ModelNotTrainedError(
                """Model is not trained. Please load model before running inference!
                You can load the pre-trained instance with the load_model wrapper."""
            )

        features = self.preprocess_audio(input_data, sr)
        x = torch.from_numpy(features).to(self.device)
        with torch.inference_mode():
            output = self.model(x)
        beats_act = output["beats"].reshape(-1).detach().cpu().numpy()
        downbeats_act = output["downbeats"].reshape(-1).detach().cpu().numpy()

        return self.post_processor(
            beats_act,
            downbeats_act,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            fps=self.fps,
            beats_per_bar=beats_per_bar,
        )

    def preprocess_audio(self, input_data, input_sr=44100):
        """Preprocess input audio file to extract features for inference.

        :param input_data: path to the input audio file, or numpy array like audio signal.
        :param input_sr: sampling rate of the input audio signal, only used if the input is an array.

        :returns: Preprocessed features as a numpy array.
        """
        import librosa

        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError("Target audio not found.")
            audio, _ = librosa.load(input_data, sr=self.sample_rate, mono=True)
        elif isinstance(input_data, np.ndarray):
            audio = input_data.astype(np.float32)
            if audio.ndim == 2:  # (channels, samples)
                audio = librosa.to_mono(audio)
            elif audio.ndim != 1:
                raise ValueError(f"Expected 1D or 2D audio, got shape {audio.shape}")
            if input_sr != self.sample_rate:
                audio = librosa.resample(
                    audio, orig_sr=input_sr, target_sr=self.sample_rate
                )
        else:
            raise ValueError("Input must be path to audio signal or an audio array")

        x = self.pre_processor(audio)

        pad_start = np.repeat(x[:1], self.pad_frames, axis=0)
        pad_stop = np.repeat(x[-1:], self.pad_frames, axis=0)
        x_padded = np.concatenate((pad_start, x, pad_stop))

        return x_padded[np.newaxis, np.newaxis]

    @staticmethod
    def save_beats(data, output_path):
        """Calling the write_csv function in compiam.io to write the output beat track in a file

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
