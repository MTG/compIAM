# unet.py
# Source: https://github.com/hojonathanho/diffusion/blob/master/
# diffusion_tf/models/unet.py


from compiam.separation.singing_voice_extraction.cold_diff_sep.model.unet_utils import (
    get_timestep_embedding,
)

import tensorflow as tf
import keras.backend as K
from tensorflow.keras import layers, models


class TimestepEmbedding(layers.Layer):
    def __init__(self, dim):
        super(TimestepEmbedding, self).__init__()
        self.dim = dim

    def call(self, t):
        return get_timestep_embedding(t, self.dim)


class Upsample(layers.Layer):
    def __init__(self, channels, with_conv=True):
        super(Upsample, self).__init__()
        self.channels = channels
        self.with_conv = with_conv
        self.conv = layers.Conv2DTranspose(
            self.channels, (3, 3), padding="same", strides=2
        )

    def call(self, inputs):
        batch_size, height, width, _ = inputs.shape
        x = self.conv(inputs)
        assert x.shape == [batch_size, height * 2, width * 2, self.channels]
        return x


class Downsample(layers.Layer):
    def __init__(self, channels, with_conv=True):
        super(Downsample, self).__init__()
        self.with_conv = with_conv
        self.channels = channels
        self.conv = layers.Conv2D(self.channels, (3, 3), padding="same", strides=2)
        self.avg_pool = layers.AveragePooling2D(strides=2, padding="same")

    def call(self, inputs):
        batch_size, height, width, _ = inputs.shape
        if self.with_conv:
            x = self.conv(inputs)
        else:
            x = self.avg_pool(inputs)
        assert x.shape == [batch_size, height // 2, width // 2, self.channels]
        return x


class ResNetBlock(layers.Layer):
    def __init__(
        self, in_ch, cond_track=None, out_ch=None, conv_shortcut=False, dropout=0.0
    ):
        super(ResNetBlock, self).__init__()
        self.in_ch = in_ch
        self.cond_track = cond_track
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut
        self.dropout = dropout

        if self.out_ch is None:
            self.out_ch = self.in_ch
        self.c_not_out_ch = self.in_ch != self.out_ch

        # Layers.
        self.group_norm1 = tf.keras.layers.BatchNormalization()
        self.non_linear1 = layers.Activation("swish")
        self.conv1 = layers.Conv2D(self.out_ch, (3, 3), padding="same")

        self.non_linear2 = layers.Activation("swish")
        self.dense2 = layers.Dense(self.out_ch)

        self.group_norm3 = tf.keras.layers.BatchNormalization()
        self.non_linear3 = layers.Activation("swish")
        self.dropout3 = layers.Dropout(self.dropout)
        self.conv3 = layers.Conv2D(self.out_ch, (3, 3), padding="same")
        if self.cond_track is not None:
            self.downsample_cond = layers.Conv2D(
                self.out_ch, (3, 3), padding="same", strides=2 * self.cond_track
            )
            self.proj_cond = tf.keras.layers.Conv1D(self.out_ch, 1, padding="same")

        self.conv4 = layers.Conv2D(self.out_ch, (3, 3), padding="same")
        self.dense4 = layers.Dense(self.out_ch)

    def call(self, inputs, temb, cond=None):
        x = inputs

        x = self.group_norm1(x)
        x = self.non_linear1(x)
        x = self.conv1(x)

        # Add in timestep embedding.
        x += self.dense2(self.non_linear2(temb))[:, None, None, :]

        x = self.group_norm3(x)
        x = self.non_linear3(x)
        x = self.dropout3(x)

        if cond is not None:
            if self.cond_track is not None:
                cond = self.downsample_cond(cond)
                x = self.conv3(x) + self.proj_cond(cond)
            else:
                x = self.conv3(x)

        if self.c_not_out_ch:
            if self.conv_shortcut:
                inputs = self.conv4(inputs)
            else:
                inputs = self.dense4(inputs)
        assert x.shape == inputs.shape
        return inputs + x


class AttentionBlock(layers.Layer):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.channels = channels

        self.avg_pool = tf.keras.layers.Lambda(
            lambda x: K.mean(x, axis=3, keepdims=True)
        )
        self.max_pool = tf.keras.layers.Lambda(
            lambda x: K.max(x, axis=3, keepdims=True)
        )
        self.cbam_feature = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False,
        )

    def call(self, inputs):
        x = inputs
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        cbam = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])
        cbam = self.cbam_feature(cbam)
        return tf.keras.layers.multiply([inputs, cbam])


class UNet(models.Model):
    def __init__(
        self,
        config,
        num_res_blocks=2,
        attn_resolutions=(8, 16, 32),
        channels=16,
        ch_mult=(1, 2, 4, 8, 16, 32, 64),
        dropout=0.2,
        resample_with_conv=False,
    ):
        super(UNet, self).__init__()
        self.config = config
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.channels = channels
        self.ch_mult = ch_mult
        self.dropout = dropout
        self.resample_with_conv = resample_with_conv
        self.num_resolutions = len(self.ch_mult)

        self.in_embed = [
            TimestepEmbedding(self.channels),
            layers.Dense(self.channels * 4),
            layers.Activation("swish"),
            layers.Dense(self.channels * 4),
        ]

        self.upsample_cond = [
            tf.keras.layers.Conv2DTranspose(
                self.channels * 2, [32, 3], [1, 1], padding="same"
            )
            for _ in range(1)
        ]
        # mel-downsampler
        self.downsample_cond = [
            tf.keras.layers.Conv2D(1, [16, 3], [1, 8], padding="same") for _ in range(2)
        ]

        # Downsampling.
        self.pre_process = layers.Conv2D(self.channels, (3, 3), padding="same")
        self.downsampling = []
        cond_track = 1
        input_track = self.channels
        channel_track = self.channels
        for i_level in range(len(ch_mult)):
            downsampling_block = []
            # Residual blocks for this resolution.
            for _ in range(self.num_res_blocks):
                if input_track in self.attn_resolutions:
                    downsampling_block.append(
                        ResNetBlock(
                            in_ch=channel_track,
                            cond_track=cond_track,
                            out_ch=self.channels * self.ch_mult[i_level],
                            dropout=self.dropout,
                        )
                    )
                else:
                    downsampling_block.append(
                        ResNetBlock(
                            in_ch=channel_track,
                            cond_track=cond_track,
                            out_ch=self.channels * self.ch_mult[i_level],
                            dropout=self.dropout,
                        )
                    )
            if i_level != self.num_resolutions - 1:
                downsampling_block.append(
                    Downsample(
                        channels=self.channels * self.ch_mult[i_level],
                        with_conv=self.resample_with_conv,
                    )
                )
                cond_track *= 2
                input_track //= 2
            channel_track = self.channels * self.ch_mult[i_level]
            self.downsampling.append(downsampling_block)

        # Middle.
        self.middle = [
            ResNetBlock(in_ch=channel_track, dropout=self.dropout),
            ResNetBlock(in_ch=channel_track, dropout=self.dropout),
        ]

        # Upsampling.
        self.upsampling = []
        channel_track = self.channels * self.ch_mult[-1] * 2
        for i_level in reversed(range(self.num_resolutions)):
            upsampling_block = []
            # Residual blocks for this resolution.
            for _ in range(self.num_res_blocks + 1):
                if input_track in self.attn_resolutions:
                    upsampling_block.append(
                        ResNetBlock(
                            in_ch=channel_track,
                            cond_track=cond_track,
                            out_ch=self.channels * self.ch_mult[i_level],
                            dropout=0.2,
                        )
                    )
                else:
                    upsampling_block.append(
                        ResNetBlock(
                            in_ch=channel_track,
                            cond_track=cond_track,
                            out_ch=self.channels * self.ch_mult[i_level],
                            dropout=0.2,
                        )
                    )
            # Upsample.
            if i_level != 0:
                upsampling_block.append(
                    Upsample(
                        channels=self.channels * self.ch_mult[i_level],
                        with_conv=self.resample_with_conv,
                    )
                )
                cond_track //= 2
                input_track *= 2
            channel_track = self.channels * self.ch_mult[i_level]
            self.upsampling.append(upsampling_block)

        # End.
        self.end = [
            layers.Conv2D(self.channels, (3, 3), padding="same"),
            layers.Conv2D(1, (3, 3), (1, 1), padding="same"),
        ]

    def call(self, inputs, temb, cond=None):

        x = inputs[..., None]

        if cond is not None:
            cond = self.vectorize_layer(cond)
            cond = self.word_embedding(cond)
            if len(cond.shape) < 3:
                cond = cond[None]
            cond = tf.transpose(cond[..., None], [0, 2, 1, 3])

            for upsample in self.upsample_cond:
                cond = tf.nn.leaky_relu(upsample(cond), 0.4)
            cond = tf.transpose(cond, [0, 3, 1, 2])
            for downsample in self.downsample_cond:
                cond = tf.nn.leaky_relu(downsample(cond), 0.4)
            cond = tf.transpose(cond, [0, 2, 1, 3])

        for lay in self.in_embed:
            temb = lay(temb)
        # Downsampling.
        hs = [self.pre_process(x)]
        for block in self.downsampling:
            for idx_block in range(self.num_res_blocks):
                if isinstance(block[idx_block], list):
                    if cond is not None:
                        h = block[idx_block][0](hs[-1], temb, cond)
                    else:
                        h = block[idx_block][0](hs[-1], temb)
                    h = block[idx_block][1](h)
                    hs.append(h)
                else:
                    if cond is not None:
                        h = block[idx_block](hs[-1], temb, cond)
                    else:
                        h = block[idx_block](hs[-1], temb)
                    hs.append(h)
            if len(block) > self.num_res_blocks:
                for extra_lay in block[self.num_res_blocks :]:
                    hs.append(extra_lay(hs[-1]))

        # Middle.
        h = hs[-1]
        for _, lay in enumerate(self.middle):
            h = lay(h, temb)

        # Upsampling.
        for block in self.upsampling:
            # Residual blocks for this resolution.
            for idx_block in range(self.num_res_blocks + 1):
                if isinstance(block[idx_block], list):
                    if cond is not None:
                        h = block[idx_block][0](
                            tf.concat([h, hs.pop()], axis=-1), temb, cond
                        )
                    else:
                        h = block[idx_block][0](tf.concat([h, hs.pop()], axis=-1), temb)
                    h = block[idx_block][1](h)
                else:
                    if cond is not None:
                        h = block[idx_block](
                            tf.concat([h, hs.pop()], axis=-1), temb, cond
                        )
                    else:
                        h = block[idx_block](tf.concat([h, hs.pop()], axis=-1), temb)
            # Upsample.
            if len(block) > self.num_res_blocks + 1:
                for extra_lay in block[self.num_res_blocks + 1 :]:
                    h = extra_lay(h)

        # End.
        for lay in self.end:
            h = lay(h)

        h = tf.keras.activations.sigmoid(h)
        h = tf.squeeze(h, axis=-1)

        return tf.multiply(inputs, h)
