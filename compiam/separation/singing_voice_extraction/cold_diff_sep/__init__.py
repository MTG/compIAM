import os
import tqdm
import gdown
import zipfile
import librosa
import math

import numpy as np
import soundfile as sf

from compiam.separation.singing_voice_extraction.cold_diff_sep.model.vad import VAD

from compiam.exceptions import ModelNotTrainedError
from compiam.utils import get_logger, WORKDIR

logger = get_logger(__name__)


class ColdDiffSep(object):
    """Leakage-aware singing voice separation model for Carnatic Music."""

    def __init__(self, model_path=None, config_path=None, sample_rate=22050):
        """Leakage-aware singing voice separation init method.

        :param model_path: path to file to the model weights.
        :param config_path: path to config for the model.
        :param sample_rate: sample rate to which the audio is sampled for extraction.
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
                "Please install tensorflow using: pip install tensorflow==2.7.2"
            )
        ###

        self.unet_config = UnetConfig()

        self.model = DiffWave(self.unet_config)
        self.sample_rate = self.unet_config.sr
        self.trained = False

        self.model_path = model_path
        if self.model_path is not None:
            self.load_model(self.model_path)

    def load_model(self, model_path):
        if ".data-00000-of-00001" not in model_path:
            path_to_check = model_path + ".data-00000-of-00001"
        if not os.path.exists(path_to_check):
            self.download_model(model_path)  # Dowloading model weights
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
        :param gpu: Id of the available GPU to use (-1 by default, to run on CPU)
        :return: Singing voice signal.
        """
        ## Setting up GPU if any
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

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
        if mixture.shape[0] == 2:
            mixture = tf.reduce_mean(mixture, axis=0)

        output_voc = np.zeros(mixture.shape)
        hopsized_chunk = int((chunk_size * 22050) / 2)
        runs = math.floor(mixture.shape[0] / hopsized_chunk)
        trim_low = 0
        for trim in tqdm.tqdm(np.arange((runs * 2) - 1)):
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
                sr=22050,
                nFFT=512,
                win_length=0.025,
                hop_length=0.01,
                theshold=0.99,
            )
            if np.sum(vad) / len(vad) < 0.25:
                pred_audio = np.zeros(pred_audio.shape)

            # Get boundary
            boundary = None
            boundary = "start" if trim == 0 else None
            boundary = "end" if trim == runs - 2 else None

            placehold_voc = np.zeros(output_voc.shape)
            placehold_voc[
                trim_low : trim_low + pred_audio.shape[0]
            ] = pred_audio * get_overlap_window(pred_audio, boundary=boundary)
            output_voc += placehold_voc
            trim_low += pred_audio.shape[0] // 2

        output_voc = output_voc * (
            np.max(np.abs(mixture.numpy())) / (np.max(np.abs(output_voc)) + 1e-6)
        )

        return output_voc

        # TODO: write a function to store audio
        # Building intuitive filename with model config
        # filefolder = os.path.join(args.input_signal.split("/")[:-1])
        # filename = args.input_signal.split("/")[-1].split(".")[:-1]
        # filename = filename[0] if len(filename) == 1 else ".".join(filename)
        # filename = filename + "_" + str(clusters) + "_" + str(scheduler) + "pred_voc"
        # sf.write(
        #    os.path.join(filefolder, filename + ".wav"),
        #    output_voc,
        #    22050) # Writing to file

    def download_model(self, model_path=None):
        """Download pre-trained model."""
        url = "https://drive.google.com/uc?id=1yj9iHTY7nCh2qrIM2RIUOXhLXt1K8WcE&export=download"
        unzip_path = (
            os.sep + os.path.join(*model_path.split(os.sep)[:-2])
            if model_path is not None
            else os.path.join(WORKDIR, "models", "separation", "cold_diff_sep")
        )
        if not os.path.exists(unzip_path):
            os.makedirs(unzip_path)
        output = os.path.join(unzip_path, "saraga-8.zip")
        gdown.download(url, output, quiet=False)

        # Unzip file
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        # Delete zip file after extraction
        os.remove(output)
        logger.warning("Files downloaded and extracted successfully.")
