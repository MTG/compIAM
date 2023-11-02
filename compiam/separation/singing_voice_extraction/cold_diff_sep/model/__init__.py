import numpy as np
import tensorflow as tf

from compiam.separation.singing_voice_extraction.cold_diff_sep.model.unet import UNet


class DiffWave(tf.keras.Model):
    """Code copied and modified from DiffWave: A Versatile Diffusion Model for Audio Synthesis.
    Zhifeng Kong et al., 2020.
    """

    def __init__(self, config):
        """Initializer.
        Args:
            config: Config, model configuration.
        """
        super(DiffWave, self).__init__()
        self.config = config
        self.net = UNet(config)

    def call(self, signal, mode="predict", step_stop=0):
        """Generate denoised audio.
        Args:
            signal: tf.Tensor, [B, T], starting signal for transformation.
        Returns:
            signal: tf.Tensor, [B, T], predicted output.
        """
        base = tf.ones([tf.shape(signal)[0]], dtype=tf.int32)
        if mode == "train":
            features = []
        for t in range(self.config.iter, step_stop, -1):
            signal = self.pred_noise(signal, base * t)
            if mode == "train":
                features.append(signal)
        if mode == "train":
            return tf.convert_to_tensor(features)
        else:
            return signal

    def diffusion(self, mixture, vocal, alpha_bar):
        """Compute conditions"""
        diffusion_step = lambda x: self._diffusion(x[0], x[1], x[2])
        return tf.map_fn(
            fn=diffusion_step,
            elems=[mixture, vocal, alpha_bar],
            fn_output_signature=(tf.float32),
        )

    def _diffusion(self, mixture, vocals, alpha_bar):
        """Trans to next state with diffusion process.
        Args:
            signal: tf.Tensor, [B, T], signal.
            alpha_bar: Union[float, tf.Tensor: [B]], cumprod(1 -beta).
            eps: Optional[tf.Tensor: [B, T]], noise.
        Return:
            tuple,
                noised: tf.Tensor, [B, T], noised signal.
                eps: tf.Tensor, [B, T], noise.
        """
        mix_mag = self.check_shape(
            self.check_shape(
                tf.abs(
                    tf.signal.stft(
                        mixture,
                        frame_length=self.config.win,
                        frame_step=self.config.hop,
                        fft_length=self.config.win,
                        window_fn=tf.signal.hann_window,
                    )
                ),
                0,
            ),
            1,
        )
        # print(mix_mag.shape)
        vocal_mag = self.check_shape(
            self.check_shape(
                tf.abs(
                    tf.signal.stft(
                        vocals,
                        frame_length=self.config.win,
                        frame_step=self.config.hop,
                        fft_length=self.config.win,
                        window_fn=tf.signal.hann_window,
                    )
                ),
                0,
            ),
            1,
        )
        return (
            tf.dtypes.cast(alpha_bar, tf.float32) * vocal_mag
            + tf.dtypes.cast(1 - tf.sqrt(alpha_bar), tf.float32) * mix_mag
        )

    @staticmethod
    def check_shape(data, dim):
        n = data.shape[dim]
        if n % 2 != 0:
            n = data.shape[dim] - 1
        if dim == 0:
            return data[:n, :]
        else:
            return data[:, :n]

    def pred_noise(self, signal, timestep):
        """Predict noise from signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            timestep: tf.Tensor, [B], timesteps of current markov chain.
            mel: tf.Tensor, [B, T // hop, M], conditional mel-spectrogram.
        Returns:
            tf.Tensor, [B, T], predicted noise.
        """
        return self.net(signal, timestep)

    def pred_signal(self, signal, eps, alpha, alpha_bar):
        """Compute mean and stddev of denoised signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            eps: tf.Tensor, [B, T], estimated noise.
            alpha: float, 1 - beta.
            alpha_bar: float, cumprod(1 - beta).
        Returns:
            tuple,
                mean: tf.Tensor, [B, T], estimated mean of denoised signal.
        """
        signal = tf.dtypes.cast(signal, tf.float64)
        eps = tf.dtypes.cast(eps, tf.float64)

        # Compute mean (our estimation) using diffusion formulation
        mean = (
            signal
            - (1 - alpha) / tf.dtypes.cast(tf.sqrt(1 - alpha_bar), tf.float64) * eps
        ) / tf.dtypes.cast(tf.sqrt(alpha), tf.float64)
        stddev = np.sqrt((1 - alpha_bar / alpha) / (1 - alpha_bar) * (1 - alpha))
        return mean, stddev

    def write(self, path, optim=None):
        """Write checkpoint with `tf.train.Checkpoint`.
        Args:
            path: str, path to write.
            optim: Optional[tf.keras.optimizers.Optimizer]
                , optional optimizer.
        """
        kwargs = {"model": self}
        if optim is not None:
            kwargs["optim"] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        ckpt.save(path)

    def restore(self, path, optim=None):
        """Restore checkpoint with `tf.train.Checkpoint`.
        Args:
            path: str, path to restore.
            optim: Optional[tf.keras.optimizers.Optimizer]
                , optional optimizer.
        """
        kwargs = {"model": self}
        if optim is not None:
            kwargs["optim"] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        return ckpt.restore(path)
