import numpy as np
import tensorflow as tf


def get_window(signal, boundary=None):
    window_out = np.ones(signal.shape)
    midpoint = window_out.shape[0] // 2
    if boundary == "start":
        window_out[midpoint:] = np.linspace(1, 0, window_out.shape[0]-midpoint)
    elif boundary == "end":
        window_out[:midpoint] = np.linspace(0, 1, window_out.shape[0]-midpoint)
    else:
        window_out[:midpoint] = np.linspace(0, 1, window_out.shape[0]-midpoint)
        window_out[midpoint:] = np.linspace(1, 0, window_out.shape[0]-midpoint)
    return window_out


def compute_stft(signal, unet_config):
    signal_stft = check_shape_3d(
        check_shape_3d(
            tf.signal.stft(
                signal,
                frame_length=unet_config.model.win,
                frame_step=unet_config.model.hop,
                fft_length=unet_config.model.win,
                window_fn=tf.signal.hann_window), 1), 2)
    mag = tf.abs(signal_stft)
    phase = tf.math.angle(signal_stft)
    return mag, phase


def compute_signal_from_stft(spec, phase, config):
    polar_spec = tf.complex(tf.multiply(spec, tf.math.cos(phase)), tf.zeros(spec.shape)) + \
        tf.multiply(tf.complex(spec, tf.zeros(spec.shape)), tf.complex(tf.zeros(phase.shape), tf.math.sin(phase)))
    return tf.signal.inverse_stft(
        polar_spec,
        frame_length=config.model.win,
        frame_step=config.model.hop,
        window_fn=tf.signal.inverse_stft_window_fn(
            config.model.hop,
            forward_window_fn=tf.signal.hann_window))


def log2(x, base):
    return int(np.log(x) / np.log(base))


def next_power_of_2(n):
    # decrement `n` (to handle the case when `n` itself is a power of 2)
    n = n - 1
    # calculate the position of the last set bit of `n`
    lg = log2(n, 2)
    # next power of two will have a bit set at position `lg+1`.
    return 1 << lg #+ 1


def check_shape_3d(data, dim):
    n = data.shape[dim]
    if n % 2 != 0:
        n = data.shape[dim] - 1
    if dim==0:
        return data[:n, :, :]
    if dim==1:
        return data[:, :n, :]
    if dim==2:
        return data[:, :, :n]


def load_audio(paths):
    mixture = tf.io.read_file(paths[0])
    vocals = tf.io.read_file(paths[1])
    mixture_audio, _ = tf.audio.decode_wav(mixture, desired_channels=1)
    vocal_audio, _ = tf.audio.decode_wav(vocals, desired_channels=1)
    return tf.squeeze(mixture_audio, axis=-1), tf.squeeze(vocal_audio, axis=-1)
