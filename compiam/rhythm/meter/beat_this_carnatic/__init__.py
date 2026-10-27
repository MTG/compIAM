import os
import numpy as np

from compiam.exceptions import ModelNotTrainedError

from compiam.utils.download import download_remote_model
from compiam.utils import get_logger, WORKDIR
from compiam.io import write_csv

logger = get_logger(__name__)


class BeatThisCarnatic(object):
    """Beat This! beat and downbeat tracker fine-tuned to Carnatic Music."""

    def __init__(
        self,
        post_processor="minimal",
        model_path=None,
        download_link=None,
        download_checksum=None,
        gpu=-1,
    ):
        """Beat This! beat tracker init method.

        :param post_processor: Post-processing method to use. Choose from 'minimal' (peak picking), or 'dbn'
            (Dynamic Bayesian Network).
        :param model_path: path to the folder with the model weights.
        :param download_link: link to the remote pre-trained model.
        :param download_checksum: checksum of the model file.
        :param gpu: Id of the available GPU to use (-1 by default, to run on CPU), use string: '0', '1', etc.
        """
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global torch
            import torch
            import torchaudio  # noqa: F401
            import einops  # noqa: F401
            import rotary_embedding_torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "Torch is required to use BeatThisCarnatic. "
                "Install compIAM with torch support: pip install 'compiam[torch]'"
            )
        ###
        global BeatTracker, PreProcessor, minimal_tracker, dbn_tracker
        global split_spectrogram, merge_chunks
        from compiam.rhythm.meter.beat_this_carnatic.model import BeatTracker
        from compiam.rhythm.meter.beat_this_carnatic.pre import PreProcessor
        from compiam.rhythm.meter.beat_this_carnatic.post import (
            minimal_tracker,
            dbn_tracker,
        )
        from compiam.rhythm.meter.beat_this_carnatic.utils import (
            split_spectrogram,
            merge_chunks,
        )

        if post_processor not in ["minimal", "dbn"]:
            raise ValueError(
                f"Invalid post_processor: {post_processor}. Choose from 'minimal', or 'dbn'."
            )

        self.gpu = gpu
        self.device = None
        self.select_gpu(gpu)

        self.model_path = model_path
        self.model_version = "beat_this_carnatic.pth"
        self.download_link = download_link
        self.download_checksum = download_checksum

        self.sample_rate = 22050
        self.fps = 50
        self.chunk_size = 1500
        self.border_size = 6
        self.pre_processor = PreProcessor(sample_rate=self.sample_rate)

        self.trained = False
        self.model = self._build_model()
        if self.model_path is not None:
            self.load_model(self.model_path)

        self.post_processor = (
            minimal_tracker if post_processor == "minimal" else dbn_tracker
        )

    def _build_model(self):
        """Build the beat tracking model."""
        model = BeatTracker().to(self.device)
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
            else os.path.join(WORKDIR, "models", "rhythm", "beat-this-carnatic")
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
        :param min_bpm: minimum BPM for beat tracking, only used with the 'dbn' post-processor (default: 55).
        :param max_bpm: maximum BPM for beat tracking, only used with the 'dbn' post-processor (default: 230).
        :param beats_per_bar: list of possible beats per bar, only used with the 'dbn' post-processor
            (default: [3, 5, 7, 8]).

        :returns: a 2-D array with the beat times (in seconds) and their position in the bar.
        """
        if self.trained is False:
            raise ModelNotTrainedError(
                """Model is not trained. Please load model before running inference!
                You can load the pre-trained instance with the load_model wrapper."""
            )

        spect = self.preprocess_audio(input_data, sr)
        beats_act, downbeats_act = self._predict_frames(spect)

        if self.post_processor is dbn_tracker:
            return self.post_processor(
                beats_act,
                downbeats_act,
                fps=self.fps,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
                beats_per_bar=beats_per_bar,
            )
        return self.post_processor(beats_act, downbeats_act, fps=self.fps)

    def _predict_frames(self, spect):
        """Framewise beat and downbeat logits of a whole piece, predicted in overlapping chunks.

        :param spect: tensor of shape (frames, bins) with the features of the piece.

        :returns: tuple with the beat and downbeat logits, each a 1-D tensor of size frames.
        """
        chunks, starts = split_spectrogram(
            spect, self.chunk_size, border_size=self.border_size
        )
        with torch.inference_mode():
            predictions = [self.model(chunk.unsqueeze(0)) for chunk in chunks]
        return tuple(
            merge_chunks(
                [p[key][0].float() for p in predictions],
                starts,
                len(spect),
                self.chunk_size,
                border_size=self.border_size,
            )
            for key in ("beat", "downbeat")
        )

    def preprocess_audio(self, input_data, input_sr=44100):
        """Preprocess input audio file to extract features for inference.

        :param input_data: path to the input audio file, or numpy array like audio signal.
        :param input_sr: sampling rate of the input audio signal, only used if the input is an array.

        :returns: Preprocessed features as a tensor of shape (frames, bins).
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

        return self.pre_processor(audio).to(self.device)

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
        if hasattr(self, "model"):
            self.model.to(self.device)
