import os
import tqdm
import librosa
import math

import numpy as np

from compiam.separation.singing_voice_extraction.cold_diff_sep.model.vad import VAD

from compiam.exceptions import ModelNotTrainedError
from compiam.utils import get_logger, WORKDIR
from compiam.utils.download import download_remote_model

logger = get_logger(__name__)


class ColdDiffSep(object):
    """Leakage-aware singing voice separation model for Carnatic Music."""

    def __init__(
        self, model_path=None, download_link=None, download_checksum=None, gpu="-1"
    ):
        """Leakage-aware singing voice separation init method.

        :param model_path: path to file to the model weights.
        :param download_link: link to the remote pre-trained model.
        :param download_checksum: checksum of the model file.
        :param gpu: Id of the available GPU to use (-1 by default, to run on CPU), use string: '0', '1', etc.
        """
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global tf
            import tensorflow as tf

            global DiffWave
            from compiam.separation.singing_voice_extraction.cold_diff_sep.model import (
                DiffWave,
            )

            global UnetConfig
            from compiam.separation.singing_voice_extraction.cold_diff_sep.model.config import (
                Config as UnetConfig,
            )

            global get_mask
            from compiam.separation.singing_voice_extraction.cold_diff_sep.model.clustering import (
                get_mask,
            )

            global compute_stft, compute_signal_from_stft, next_power_of_2, get_overlap_window
            from compiam.separation.singing_voice_extraction.cold_diff_sep.model.signal_processing import (
                get_overlap_window,
                compute_stft,
                compute_signal_from_stft,
                next_power_of_2,
            )

        except:
            raise ImportError(
                "In order to use this tool you need to have tensorflow installed. "
                "Install compIAM with tensorflow support: pip install 'compiam[tensorflow]'"
            )
        ###

        ## Setting up GPU if specified
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.gpu = gpu

        self.unet_config = UnetConfig()

        self.model = DiffWave(self.unet_config)
        self.sample_rate = self.unet_config.sr
        self.trained = False

        self.model_path = model_path
        self.download_link = download_link
        self.download_checksum = download_checksum
        if self.model_path is not None:
            self.load_model(self.model_path)

    def load_model(self, model_path):
        if ".data-00000-of-00001" not in model_path:
            path_to_check = model_path + ".data-00000-of-00001"
        if not os.path.exists(path_to_check):
            self.download_model(model_path)  # Downloading model weights
        self.model.restore(model_path).expect_partial()
        self.model_path = model_path
        self.trained = True

    def separate(
        self,
        input_data,
        input_sr=44100,
        clusters=5,
        scheduler=4,
        chunk_size=3,
        gpu="-1",
    ):
        """Separate singing voice from mixture.

        :param input_data: Audio signal to separate.
        :param input_sr: sampling rate of the input array of data (if any). This variable is only
            relevant if the input is an array of data instead of a filepath.
        :param clusters: Number of clusters to use to build the separation masks.
        :param scheduler: Scheduler factor to weight the clusters to be more or less restirctive with the interferences.
        :param gpu: Id of the available GPU to use (-1 by default, to run on CPU), use string: '0', '1', etc.
        :return: Singing voice signal.
        """
        ## Setting up GPU if any
        if gpu != self.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            self.gpu = gpu

        if self.trained is False:
            raise ModelNotTrainedError(
                """ Model is not trained. Please load model before running inference!
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
        mixture = tf.convert_to_tensor(audio, dtype=tf.float32)

        if len(mixture.shape) == 3:
            if mixture.shape[0] != 1:
                raise ValueError("Batching is not supported. Please provide a single audio signal.")
            else:
                mixture = mixture.squeeze(0)
                mixture = tf.reduce_mean(mixture, axis=0, keepdims=False)  # Removing dimension
                logger.info(
                    f"Downsampling to mono... your audio is stereo, \
                        and the model is trained on mono audio."
                )
        if len(mixture.shape) > 3:
            raise ValueError("Input must be a single, unbatched audio")

        if mixture.shape[0] <= 2:
            mixture = tf.reduce_mean(mixture, axis=0, keepdims=False)
            logger.info(
                f"Downsampling to mono... your audio is stereo, \
                    and the model is trained on mono audio."
            )
            
        output_voc = np.zeros(mixture.shape)
        hopsized_chunk = int((chunk_size * self.sample_rate) / 2)
        runs = math.floor(mixture.shape[0] / hopsized_chunk)
        trim_low = 0
        for trim in tqdm.tqdm(np.arange((runs * 2) - 1)):
            try:
                trim_high = int(trim_low + (hopsized_chunk * 2))

                # Get input mixture spectrogram
                mix_trim = mixture[trim_low:trim_high]
                mix_mag, mix_phase = compute_stft(mix_trim[None], self.unet_config)
                new_len = next_power_of_2(mix_mag.shape[1])
                mix_mag_trim = mix_mag[:, :new_len, :]
                mix_phase_trim = mix_phase[:, :new_len, :]

                # Get and stack cold diffusion steps
                diff_feat = self.model(mix_mag_trim, mode="train")
                diff_feat = tf.transpose(diff_feat, [1, 0, 2, 3])
                diff_feat_t = tf.squeeze(
                    tf.reshape(
                        diff_feat, [1, 8, diff_feat.shape[-2] * diff_feat.shape[-1]]
                    ),
                    axis=0,
                ).numpy()

                # Normalize features, all energy curves having same range
                normalized_feat = []
                for j in np.arange(diff_feat_t.shape[1]):
                    normalized_curve = diff_feat_t[:, j] / (
                        np.max(np.abs(diff_feat_t[:, j])) + 1e-6
                    )
                    normalized_feat.append(normalized_curve)
                normalized_feat = np.array(normalized_feat, dtype=np.float32)

                # Compute mask using unsupervised clustering and reshape to magnitude spec shape
                mask = get_mask(normalized_feat, clusters, scheduler)
                mask = tf.convert_to_tensor(
                    mask, dtype=tf.float32
                )  # Move mask to tensor and cast to float
                mask = tf.reshape(mask, mix_mag_trim.shape)

                # Getting last step of computed features and applying mask
                diff_feat_t = tf.reshape(diff_feat_t[-1, :], mix_mag_trim.shape)
                output_signal = tf.math.multiply(diff_feat_t, mask)

                # Silence unvoiced regions
                output_signal = compute_signal_from_stft(
                    output_signal, mix_phase_trim, self.unet_config
                )
                # From here on, pred_audio is numpy
                pred_audio = tf.squeeze(output_signal, axis=0).numpy()
                vad = VAD(
                    pred_audio,
                    sr=self.sample_rate,
                    nFFT=512,
                    win_length=0.025,
                    hop_length=0.01,
                    threshold=0.99,
                )
                if np.sum(vad) / len(vad) < 0.25:
                    pred_audio = np.zeros(pred_audio.shape)

                # Get boundary
                boundary = None
                boundary = "start" if trim == 0 else None
                boundary = "end" if trim == runs - 2 else None

                placehold_voc = np.zeros(output_voc.shape)
                placehold_voc[trim_low : trim_low + pred_audio.shape[0]] = (
                    pred_audio * get_overlap_window(pred_audio, boundary=boundary)
                )
                output_voc += placehold_voc
                trim_low += pred_audio.shape[0] // 2

            except:
                output_voc = output_voc * (
                    np.max(np.abs(mixture.numpy()))
                    / (np.max(np.abs(output_voc)) + 1e-6)
                )
                output_voc = output_voc[:trim_low]
                return output_voc

        return output_voc * (
            np.max(np.abs(mixture.numpy())) / (np.max(np.abs(output_voc)) + 1e-6)
        )

    def download_model(self, model_path=None, force_overwrite=False):
        """Download pre-trained model."""
        download_path = (
            os.sep + os.path.join(*model_path.split(os.sep)[:-2])
            if model_path is not None
            else os.path.join(WORKDIR, "models", "separation", "cold_diff_sep")
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
