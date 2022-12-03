import os
import math
import librosa

import numpy as np
from compiam.exceptions import ModelNotFoundError, ModelNotTrainedError

from compiam.utils.pitch import normalisation, resampling
from compiam.melody.pitch_extraction.ftanet_carnatic.pitch_processing import (
    batchize_test,
    get_est_arr,
)
from compiam.melody.pitch_extraction.ftanet_carnatic.cfp import cfp_process


class FTANetCarnatic(object):
    """FTA-Net melody extraction tuned to Carnatic Music."""

    def __init__(self, model_path=None, sample_rate=8000):
        """FTA-Net melody extraction init method.

        :param model_path: path to file to the model weights.
        :param sample_rate: Sample rate to which the audio is sampled for extraction.
        """
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global tf
            import tensorflow as tf

        except:
            raise ImportError(
                "In order to use this tool you need to have tensorflow installed. "
                "Please install tensorflow using: pip install tensorflow==2.7.2"
            )
        ###

        self.model = self._build_model()
        self.sample_rate = sample_rate
        self.trained = False

        self.model_path = model_path
        if self.model_path is not None:
            self.load_model(self.model_path)

    @staticmethod
    def SF_Module(x_list, n_channel, reduction, limitation):
        """Selection and fusion module.
        Implementation taken from https://github.com/yushuai/FTANet-melodic

        :param x_list: list of tensor inputs.
        :param n_channel: number of feature channels.
        :param reduction: the rate to which the data is compressed.
        :param limitation: setting a compressing limit.
        :returns: a tensor with the fused and selected feature map.
        """
        ## Split
        fused = None
        for x_s in x_list:
            if fused == None:
                fused = x_s
            else:
                fused = tf.keras.layers.Add()([fused, x_s])

        ## Fuse
        fused = tf.keras.layers.GlobalAveragePooling2D()(fused)
        fused = tf.keras.layers.BatchNormalization()(fused)
        fused = tf.keras.layers.Dense(
            max(n_channel // reduction, limitation), activation="selu"
        )(fused)

        ## Select
        masks = []
        for i in range(len(x_list)):
            masks.append(tf.keras.layers.Dense(n_channel)(fused))
        mask_stack = tf.keras.layers.Lambda(
            tf.keras.backend.stack, arguments={"axis": -1}
        )(masks)
        # (n_channel, n_kernel)
        mask_stack = tf.keras.layers.Softmax(axis=-2)(mask_stack)

        selected = None
        for i, x_s in enumerate(x_list):
            mask = tf.keras.layers.Lambda(lambda z: z[:, :, i])(mask_stack)
            mask = tf.keras.layers.Reshape((1, 1, n_channel))(mask)
            x_s = tf.keras.layers.Multiply()([x_s, mask])
            if selected == None:
                selected = x_s
            else:
                selected = tf.keras.layers.Add()([selected, x_s])
        return selected

    @staticmethod
    def FTA_Module(x, shape, kt, kf):
        """Selection and fusion module.
        Implementation taken from https://github.com/yushuai/FTANet-melodic

        :param x: input tensor.
        :param shape: the shape of the input tensor.
        :param kt: kernel size for time attention.
        :param kf: kernel size for frequency attention.
        :returns: the resized input, the time-attention map,
            and the frequency-attention map.
        """
        x = tf.keras.layers.BatchNormalization()(x)

        ## Residual
        x_r = tf.keras.layers.Conv2D(
            shape[2], (1, 1), padding="same", activation="relu"
        )(x)

        ## Time Attention
        # Attn Map (1, T, C), FC
        a_t = tf.keras.layers.Lambda(tf.keras.backend.mean, arguments={"axis": -3})(x)
        a_t = tf.keras.layers.Conv1D(shape[2], kt, padding="same", activation="selu")(
            a_t
        )
        a_t = tf.keras.layers.Conv1D(shape[2], kt, padding="same", activation="selu")(
            a_t
        )  # 2
        a_t = tf.keras.layers.Softmax(axis=-2)(a_t)
        a_t = tf.keras.layers.Reshape((1, shape[1], shape[2]))(a_t)
        # Reweight
        x_t = tf.keras.layers.Conv2D(
            shape[2], (3, 3), padding="same", activation="selu"
        )(x)
        x_t = tf.keras.layers.Conv2D(
            shape[2], (5, 5), padding="same", activation="selu"
        )(x_t)
        x_t = tf.keras.layers.Multiply()([x_t, a_t])

        # Frequency Attention
        # Attn Map (F, 1, C), Conv1D
        a_f = tf.keras.layers.Lambda(tf.keras.backend.mean, arguments={"axis": -2})(x)
        a_f = tf.keras.layers.Conv1D(shape[2], kf, padding="same", activation="selu")(
            a_f
        )
        a_f = tf.keras.layers.Conv1D(shape[2], kf, padding="same", activation="selu")(
            a_f
        )
        a_f = tf.keras.layers.Softmax(axis=-2)(a_f)
        a_f = tf.keras.layers.Reshape((shape[0], 1, shape[2]))(a_f)
        # Reweight
        x_f = tf.keras.layers.Conv2D(
            shape[2], (3, 3), padding="same", activation="selu"
        )(x)
        x_f = tf.keras.layers.Conv2D(
            shape[2], (5, 5), padding="same", activation="selu"
        )(x_f)
        x_f = tf.keras.layers.Multiply()([x_f, a_f])

        return x_r, x_t, x_f

    def _build_model(self, input_shape=(320, 128, 3)):
        """Building the entire FTA-Net.
        Implementation taken from https://github.com/yushuai/FTANet-melodic

        :param input_shape: input shape.
        :returns: a tensorflow Model instance of the FTA-Net.
        """
        visible = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.BatchNormalization()(visible)

        ## Bottom
        # bm = BatchNormalization()(x)
        bm = x
        bm = tf.keras.layers.Conv2D(
            16, (4, 1), padding="valid", strides=(4, 1), activation="selu"
        )(
            bm
        )  # 80
        bm = tf.keras.layers.Conv2D(
            16, (4, 1), padding="valid", strides=(4, 1), activation="selu"
        )(
            bm
        )  # 20
        bm = tf.keras.layers.Conv2D(
            16, (4, 1), padding="valid", strides=(4, 1), activation="selu"
        )(
            bm
        )  # 5
        bm = tf.keras.layers.Conv2D(
            1, (5, 1), padding="valid", strides=(5, 1), activation="selu"
        )(
            bm
        )  # 1

        shape = input_shape
        x_r, x_t, x_f = self.FTA_Module(x, (shape[0], shape[1], 32), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 32, 4, 4)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x_r, x_t, x_f = self.FTA_Module(x, (shape[0] // 2, shape[1] // 2, 64), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 64, 4, 4)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

        x_r, x_t, x_f = self.FTA_Module(x, (shape[0] // 4, shape[1] // 4, 128), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 128, 4, 4)

        x_r, x_t, x_f = self.FTA_Module(x, (shape[0] // 4, shape[1] // 4, 128), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 128, 4, 4)

        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x_r, x_t, x_f = self.FTA_Module(x, (shape[0] // 2, shape[1] // 2, 64), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 64, 4, 4)

        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x_r, x_t, x_f = self.FTA_Module(x, (shape[0], shape[1], 32), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 32, 4, 4)

        x_r, x_t, x_f = self.FTA_Module(x, (shape[0], shape[1], 1), 3, 3)
        x = self.SF_Module([x_r, x_t, x_f], 1, 4, 4)
        x = tf.keras.layers.Concatenate(axis=1)([bm, x])

        # Softmax
        x = tf.keras.layers.Lambda(tf.keras.backend.squeeze, arguments={"axis": -1})(x)
        x = tf.keras.layers.Softmax(axis=-2)(x)
        return tf.keras.models.Model(inputs=visible, outputs=x)

    def load_model(self, model_path):
        if ".data-00000-of-00001" not in model_path:
            path_to_check = model_path + ".data-00000-of-00001"
        if not os.path.exists(path_to_check):
            raise ModelNotFoundError(
                """
                Given path to model weights not found. Make sure you enter the path correctly.
                A training process for the FTA-Net tuned to Carnatic is under development right
                now and will be added to the library soon. Meanwhile, we provide the weights in the
                latest repository version (https://github.com/MTG/compIAM) so make sure you have these
                available before loading the Carnatic FTA-Net.
            """
            )
        self.model.load_weights(model_path).expect_partial()
        self.model_path = model_path
        self.trained = True

    def predict(self, file_path, hop_size=80, batch_size=5, out_step=None):
        """Extract melody from file_path.
        Implementation taken (and slightly adapted) from https://github.com/yushuai/FTANet-melodic.

        :param file_path: path to file to extract.
        :param sample_rate: sample rate of extraction process.
        :param hop_size: hop size between frequency estimations.
        :param batch_size: batches of seconds that are passed through the model
            (defaulted to 5, increase if enough computational power, reduce if
            needed).
        :param out_step: particular time-step duration if needed at output
        :returns: a 2-D list with time-stamps and pitch values per timestamp.
        """
        if self.trained is False:
            raise ModelNotTrainedError("""
                Model is not trained. Please load model before running inference!
                You can load the pre-trained instance with the load_model wrapper.
            """)

        if not os.path.exists(file_path):
            raise FileNotFoundError("Target audio not found.")

        xlist = []
        timestamps = []

        y, _ = librosa.load(file_path, sr=self.sample_rate)
        audio_len = len(y)
        batch_min = self.sample_rate * 60 * batch_size
        freqs = []
        if len(y) > batch_min:
            iters = math.ceil(len(y) / batch_min)
            for i in np.arange(iters):
                if i < iters - 1:
                    audio_in = y[batch_min * i : batch_min * (i + 1)]
                if i == iters - 1:
                    audio_in = y[batch_min * i :]
                feature, _, time_arr = cfp_process(
                    audio_in, sr=self.sample_rate, hop=hop_size
                )
                data = batchize_test(feature, size=128)
                xlist.append(data)
                timestamps.append(time_arr)

                estimation = get_est_arr(self.model, xlist, timestamps, batch_size=16)
                if i == 0:
                    freqs = estimation[:, 1]
                else:
                    freqs = np.concatenate((freqs, estimation[:, 1]))
        else:
            feature, _, time_arr = cfp_process(y, sr=self.sample_rate, hop=hop_size)
            data = batchize_test(feature, size=128)
            xlist.append(data)
            timestamps.append(time_arr)
            # Getting estimatted pitch
            estimation = get_est_arr(self.model, xlist, timestamps, batch_size=16)
            freqs = estimation[:, 1]
        TStamps = np.linspace(0, audio_len / self.sample_rate, len(freqs))

        freqs[freqs<50] = 0
        
        output = np.array([TStamps, freqs]).transpose()

        if out_step is not None:
            new_len = int((audio_len / self.sample_rate) // out_step)
            return resampling(output, new_len)

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
