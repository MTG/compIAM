from augmentation.augment_data_ar import augment_data_attack_remix
from augmentation.augment_data_ps import augment_data_pitch_shift
from augmentation.augment_data_sf import augment_data_spectral_shape
from augmentation.augment_data_sr import augment_data_stroke_remix
from augmentation.augment_data_ts import augment_data_time_scale

# TODO: think about global variables like this
n_jobs = 4

def pitch_shift(in_path, out_dir, fs, ps, shifts, n_jobs=n_jobs):
	_ = Parallel(n_jobs=n_jobs)(delayed(augment_data_pitch_shift)(in_path, out_dir, fs, ps)
		for ps in shifts)

def spectral_shape(in_path, out_dir, fs, gain_set, winDur, hopDur, params, n_jobs=n_jobs):
	_ = Parallel(n_jobs=n_jobs)(delayed(augment_data_spectral_shape)(in_path, out_dir, fs, gain_set, winDur=winDur, hopDur=hopDur)
		for gain_set in params)

def stroke_remix(in_path, out_dir, fs, gain_set, templates, winDur, hopDur, params, n_jobs=n_jobs):
	_ = Parallel(n_jobs=n_jobs)(delayed(augment_data_stroke_remix)(in_path, out_dir, fs, gain_set, templates, winDur, hopDur)
		for gain_set in params)

def time_scale(in_path, out_dir, fs, ts, params, n_jobs=n_jobs):
	_ = Parallel(n_jobs=n_jobs)(delayed(augment_data_time_scale)(in_path, out_dir, fs, ts) 
		for ts in params)

def attack_remix(in_path, out_dir, fs, G, winDur, hopDur, params, n_jobs=n_jobs):
	_ = Parallel(n_jobs=n_jobs)(delayed(augment_data_attack_remix)(in_path, out_dir, fs, G, winDur, hopDur)
		for G in params)