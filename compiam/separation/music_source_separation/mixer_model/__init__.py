import os

import numpy as np

from compiam.exceptions import ModelNotTrainedError
from compiam.utils import get_logger, WORKDIR
from compiam.utils.download import download_remote_model


logger = get_logger(__name__)


class MixerModel(object):
    """Leakage-aware multi-source separation model for Carnatic Music."""

    def __init__(
        self,
        model_path=None,
        download_link=None,
        download_checksum=None,
        sample_rate=24000,
        gpu="-1",
    ):
        """Leakage-aware singing voice separation init method.

        :param model_path: path to file to the model weights.
        :param download_link: link to the remote pre-trained model.
        :param download_checksum: checksum of the model file.
        :param sample_rate: sample rate to which the audio is sampled for extraction.
        :param gpu: Id of the available GPU to use (-1 by default, to run on CPU), use string: '0', '1', etc.
        """
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global torch
            import torch

            global nn
            import torch.nn as nn

            global torchaudio
            import torchaudio

            global MDXModel, ConvTDFNet
            from compiam.separation.music_source_separation.mixer_model.models import (
                MDXModel,
                ConvTDFNet,
            )

        except:
            raise ImportError(
                "In order to use this tool you need to have torch and torchaudio installed. "
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

        self.chunk_size = self.model.chunk_size

    def forward(self, x):
        """Forward pass of the mixer model"""
        return self.model(x)

    def _build_model(self):
        """Build the MDXNet mixer model."""
        mdxnet = MDXModel().to(self.device)
        mdxnet.eval()
        return mdxnet

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            self.download_model(model_path)  # Downloading model weights
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model_path = model_path
        self.trained = True

    def separate(
        self,
        input_data,
        input_sr=44100,
        gpu="-1",
    ):
        """Separate singing voice and violin from mixture.

        :param input_data: Audio signal to separate.
        :param input_sr: sampling rate of the input array of data (if any). This variable is only
            relevant if the input is an array of data instead of a filepath.
        :param gpu: Id of the available GPU to use (-1 by default, to run on CPU), use string: '0', '1', etc.
        :return: Singing voice and violin signals.
        """
        ## Setting up GPU if specified
        self.gpu = gpu
        self.device = None
        self.select_gpu(gpu)

        if self.trained is False:
            raise ModelNotTrainedError(
                """ Model is not trained. Please load model before running inference!
                You can load the pre-trained instance with the load_model wrapper."""
            )

        # Loading and resampling audio
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError("Target audio not found.")
            audio, input_sr = torchaudio.load(input_data)
        elif isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data).to(torch.float32).to(self.device)
        elif isinstance(input_data, torch.Tensor):
            input_data = input_data.to(torch.float32).to(self.device)
        else:
            raise ValueError("Input must be path to audio signal or an audio array")
        
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)

        if len(input_data.shape) == 3:
            if input_data.shape[0] != 1:
                raise ValueError("Batching is not supported. Please provide a single audio signal.")
            input_data = input_data.squeeze(0)
        
        # resample audio
        if input_sr != self.sample_rate:
            logger.warning(
                f"Resampling... (input sampling rate is assumed {input_sr}Hz, \
                    make sure this is correct and change input_sr otherwise)"
            )
            audio = torchaudio.transforms.Resample(
                orig_freq=input_sr, new_freq=self.sample_rate
            )(input_data)

        # downsampling to mono
        if audio.shape[0] == 2:
            audio = audio.mean(dim=0, keepdim=True)
            logger.info(
                f"Downsampling to mono... your audio is stereo, \
                    and the model is trained on mono audio."
            )

        # audio has shape B, 1, N
        audio = audio.reshape(-1)
        predictions = []
        pad_length = self.chunk_size - (audio.shape[-1] % self.chunk_size)
        audio = torch.nn.functional.pad(audio, (0, pad_length))

        for i in range(0, audio.shape[-1], self.chunk_size):
            audio_chunk = audio[i : i + self.chunk_size].reshape(
                1, 1, -1
            )  # TODO Batching
            predictions.append(self.forward(audio_chunk))

        result = torch.cat(predictions, dim=-1)
        result = result[:, :, :-pad_length]

        vocal_separation = torchaudio.transforms.Resample(
            orig_freq=self.sample_rate, new_freq=input_sr
        )(result[:, 0, :])
        violin_separation = torchaudio.transforms.Resample(
            orig_freq=self.sample_rate, new_freq=input_sr
        )(result[:, 1, :])
        
        vocal_separation = vocal_separation.detach().cpu().numpy().reshape(-1)
        violin_separation = violin_separation.detach().cpu().numpy().reshape(-1)
        return (vocal_separation, violin_separation)

    def download_model(self, model_path=None, force_overwrite=False):
        """Download pre-trained model."""
        download_path = (
            os.sep + os.path.join(*model_path.split(os.sep)[:-2])
            if model_path is not None
            else os.path.join(WORKDIR, "models", "separation", "mixer_model")
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
