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


# Evaluate  


# Predict
from compiam.rhythm.tabla_transcription.models import classify_strokes

onsets, labels = classify_strokes(path_to_audio, predict_thresh, saved_model_dir, device)

# Visualise
from compiam.visualisation.audio import plot_waveform

plot_waveform(path_to_audio, t1, t2, dict(zip(onsets, labels)))

# Output
with open(out_file, 'w') as f:
	for o,l in zip(onsets, labels):
		writer = csv.writer(f)
		writer.writerow(row)
