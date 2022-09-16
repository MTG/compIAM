# Data Load
data = load_data('data_name')

# Data augment
from compiam.utils.augment import pitch_shift, spectral_shape, stroke_remix, time_scale, attack_remix

pitch_shift(in_path, out_dir, fs, ps, shifts, n_jobs=n_jobs)
spectral_shape(in_path, out_dir, fs, gain_set, winDur, hopDur, params, n_jobs=n_jobs)
stroke_remix(in_path, out_dir, fs, gain_set, templates, winDur, hopDur, params, n_jobs=n_jobs)
time_scale(in_path, out_dir, fs, ts, params, n_jobs=n_jobs)
attack_remix(in_path, out_dir, fs, G, winDur, hopDur, params, n_jobs=n_jobs)

# Train
# TODO

# Evaluate  
# TODO

# Predict
# Predict
from compiam.utils import load_model

model = load_model('rhythm:4way-tabla')

path_to_audio = '/Volumes/Shruti/asplab2/4way-tabla-transcription/dataset/train/audios/KB_9.wav'
onsets, labels = model.predict(path_to_audio)

# Visualise
from compiam.visualisation.audio import plot_waveform

plot_waveform(path_to_audio, 0, 3, dict(zip(onsets, labels)), filepath='test.png')

# Output
from compiam.io import write_csv

write_csv([onsets, labels], out_path, header=['onset','label'])