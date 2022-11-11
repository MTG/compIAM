import numpy as np

# parameters for data loader
batch_size = 32
params = {"batch_size": batch_size, "shuffle": True, "num_workers": 4}
max_epochs = 500

# input-output parameters
fs = 16000

winsize_sec = 0.04
winsize = int(winsize_sec * fs)
hopsize_sec = 0.02
hopsize = int(hopsize_sec * fs)
nfft = int(2 ** (np.ceil(np.log2(winsize))))

input_len_sec = 8
input_len = int(input_len_sec / hopsize_sec)
input_hop_sec = 0.5
input_hop = int(input_hop_sec / hopsize_sec)
input_height = 40

classes_dict = {
    "voc": [1.0, 2.0, 4.0, 8.0],
    "pakh": [1.0, 2.0, 4.0, 8.0, 16.0],
    "net": [1.0, 2.0, 4.0, 8.0, 16.0],
}

# minimum section duration for smoothing s.t.m. estimates
min_sec_dur = 5  # in seconds
min_sec_dur /= input_hop_sec
