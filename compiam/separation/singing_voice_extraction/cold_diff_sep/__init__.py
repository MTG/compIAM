import os
import tqdm
import json
import math

import numpy as np
import soundfile as sf

from scipy.signal import get_window

from compiam.separation.singing_voice_extraction.cold_diff_sep.model.vad import VAD
from compiam.separation.singing_voice_extraction.cold_diff_sep.model.signal_processing import (
    compute_stft,
    compute_signal_from_stft,
    next_power_of_2
)
from compiam.exceptions import ModelNotTrainedError, ModelNotFoundError
from compiam.utils import get_logger, load_and_resample

logger = get_logger(__name__)


class ColdDiffSep(object):
    """Leakage-aware singing voice separation model for Carnatic Music."""

    def __init__(
        self,
        model_path=None,
        config_path=None,
        sample_rate=22050
    ):
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
            from compiam.separation.singing_voice_extraction.cold_diff_sep.model import DiffWave
            global UnetConfig
            from compiam.separation.singing_voice_extraction.cold_diff_sep.model.config import Config as UnetConfig
            global get_mask
            from compiam.separation.singing_voice_extraction.cold_diff_sep.model.clustering import get_mask

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
            raise ModelNotFoundError(
                """Given path to model weights not found. Make sure you enter the path correctly."""
            )
        self.model.restore(model_path).expect_partial()
        self.model_path = model_path
        self.trained = True

    def separate(
        self,
        input_data,
        input_sr=44100,
        clusters=5,
        scheduler=4,
        gpu="-1"
    ):
        """Separate singing voice from mixture.

        :param input_data: Audio signal to separate.
        TODO: add missing params
        :return: Singing voice signal.
        """
        ## Setting up GPU if any
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        if self.trained is False:
            raise ModelNotTrainedError(
                """
                Model is not trained. Please load model before running inference!
                You can load the pre-trained instance with the load_model wrapper.
            """
            )
        
        # Loading and resampling audio
        audio = load_and_resample(input_data, input_sr, self.sample_rate)
        mixture = tf.convert_to_tensor(audio, dtype=tf.float32)
        if mixture.shape[0] == 2:
            mixture = tf.reduce_mean(mixture, axis=0)

        TRIMS = self.batch_size
        output_voc = np.zeros(mixture.shape)
        hopsized_batch = int((TRIMS*22050) / 2)
        runs = math.floor(mixture.shape[0] / hopsized_batch)
        trim_low = 0
        for trim in tqdm.tqdm(np.arange((runs*2)-1)):
            trim_high = int(trim_low + (hopsized_batch*2))

            # Get input mixture spectrogram
            mix_trim = mixture[trim_low:trim_high]
            mix_mag, mix_phase = compute_stft(mix_trim[None], self.unet_config)
            new_len = next_power_of_2(mix_mag.shape[1])
            mix_mag_trim = mix_mag[:, :new_len, :]
            mix_phase_trim = mix_phase[:, :new_len, :]

            # Get and stack cold diffusion steps
            diff_feat = self.model(mix_mag_trim, mode="train")
            diff_feat = tf.transpose(diff_feat, [1, 0, 2, 3])
            diff_feat_t = tf.squeeze(tf.reshape(diff_feat, [1, 8, diff_feat.shape[-2]*diff_feat.shape[-1]]), axis=0).numpy()

            # Normalize features, all energy curves having same range
            normalized_feat = []
            for j in np.arange(diff_feat_t.shape[1]):
                normalized_curve = diff_feat_t[:, j] / np.max(np.abs(diff_feat_t[:, j]))
                normalized_feat.append(normalized_curve)
            normalized_feat = np.array(normalized_feat, dtype=np.float32)

            # Compute mask using unsupervised clustering and reshape to magnitude spec shape
            mask = get_mask(normalized_feat, clusters, scheduler)
            mask = tf.reshape(mask, mix_mag_trim.shape)

            # Getting last step of computed features and applying mask
            diff_feat_t = tf.reshape(diff_feat_t[-1, :], mix_mag_trim.shape)
            output_signal = tf.math.multiply(diff_feat_t, mask)

            # Silence unvoiced regions
            output_signal = compute_signal_from_stft(output_signal, mix_phase_trim, self.unet_config)
            pred_audio = tf.squeeze(output_signal, axis=0).numpy()
            vad = VAD(pred_audio, sr=22050, nFFT=512, win_length=0.025, hop_length=0.01, theshold=0.99)
            if np.sum(vad) / len(vad) < 0.25:
                pred_audio = np.zeros(pred_audio.shape)

            # Get boundary
            boundary = None
            boundary = "start" if trim == 0 else None
            boundary = "end" if trim == runs-2 else None

            placehold_voc = np.zeros(output_voc.shape)
            placehold_voc[trim_low:trim_low+pred_audio.shape[0]] = pred_audio * get_window(pred_audio, boundary=boundary)
            output_voc += placehold_voc
            trim_low += pred_audio.shape[0] // 2

        output_voc = output_voc * (np.max(np.abs(mixture.numpy())) / np.max(np.abs(output_voc)))
        
        return output_voc
        
        # TODO: write a function to store audio
        # Building intuitive filename with model config
        #filefolder = os.path.join(args.input_signal.split("/")[:-1])
        #filename = args.input_signal.split("/")[-1].split(".")[:-1]
        #filename = filename[0] if len(filename) == 1 else ".".join(filename)
        #filename = filename + "_" + str(clusters) + "_" + str(scheduler) + "pred_voc"
        #sf.write(
        #    os.path.join(filefolder, filename + ".wav"),
        #    output_voc,
        #    22050) # Writing to file

