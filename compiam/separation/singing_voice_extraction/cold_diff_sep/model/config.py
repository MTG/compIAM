import numpy as np
import tensorflow as tf

class Config:
    """Configuration for DiffWave implementation.
    """
    def __init__(self):
        self.model_type = None
        
        self.sr = 22050

        self.hop = 256
        self.win = 1024

        # mel-scale filter bank
        self.mel = 80
        self.fmin = 0
        self.fmax = 8000

        self.eps = 1e-5

        # sample size
        self.frames = (self.hop + 6) * 128  # 16384
        self.batch = 8

        # leaky relu coefficient
        self.leak = 0.4

        # embdding config
        self.embedding_size = 128
        self.embedding_proj = 512
        self.embedding_layers = 2
        self.embedding_factor = 4

        # upsampler config
        self.upsample_stride = [4, 1]
        self.upsample_kernel = [32, 3]
        self.upsample_layers = 4
        # computed hop size
        # block config
        self.channels = 64
        self.kernel_size = 3
        self.dilation_rate = 2
        self.num_layers = 30
        self.num_cycles = 3

        # noise schedule
        self.iter = 8                 # 20, 40, 50
        self.noise_policy = 'linear'
        self.noise_start = 1e-4
        self.noise_end = 0.5           # 0.02 for 200

    def beta(self):
        """Generate beta-sequence.
        Returns:
            List[float], [iter], beta values.
        """
        mapper = {
            'linear': self._linear_sched,
        }
        if self.noise_policy not in mapper:
            raise ValueError('invalid beta policy')
        return mapper[self.noise_policy]()

    def _linear_sched(self):
        """Linearly generated noise.
        Returns:
            List[float], [iter], beta values.
        """
        return np.linspace(
            self.noise_start, self.noise_end, self.iter, dtype=np.float32)

    def window_fn(self):
        """Return window generator.
        Returns:
            Callable, window function of tf.signal
                , which corresponds to self.win_fn.
        """
        mapper = {
            'hann': tf.signal.hann_window,
            'hamming': tf.signal.hamming_window
        }
        if self.win_fn in mapper:
            return mapper[self.win_fn]
        
        raise ValueError('invalid window function: ' + self.win_fn)
