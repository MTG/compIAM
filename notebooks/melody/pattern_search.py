audio_path = "/Volumes/Shruti/asplab2/cae-invar/audio/lara_wim/spleeter/2018_11_13_am_Sec_4_P1_Atana_V.2.mp3"

# TODO, for each stage include a visualisation

# Pattern Extraction for a Given Audio
from compiam import load_model

# Feature Extraction
# CAE features 
cae = load_model("melody:cae-carnatic")

ampl, phase = cae.extract_features(audio_path)

# Pitch Track 
ftanet = load_model('melody:ftanet-carnatic')
pitch_track = ftanet.predict(audio_path)
pitch = pitch_track[:,1]
time  = pitch_track[:,0]
timestep  = time[2]-time[1]

# Stability Track
from compiam.utils.pitch import extract_stability_mask
import numpy as np

stability_mask = extract_stability_mask(
	pitch=pitch, 
	min_stab_sec=1.0, 
	hop_sec=0.2, 
	var=60,
	timestep=timestep)

# Silence Mask
silence_mask = pitch==0

exclusion_mask = np.logical_or(silence_mask==1, stability_mask==1)

# Self Similarity
from compiam.melody.pattern import self_similarity

ss = self_similarity(ampl, exclusion_mask=exclusion_mask, timestep=timestep, hop_length=cae.hop_length, sr=cae.sr)
X, orig_sparse_lookup, sparse_orig_lookup, boundaries_orig, boundaries_sparse = ss





#    ## Output
#    metadata = {
#        'orig_size': (len(data), len(data)),
#        'sparse_size': (matrix.shape[0], matrix.shape[0]),
#        'orig_sparse_lookup': orig_sparse_lookup,
#        'sparse_orig_lookup': sparse_orig_lookup,
#        'boundaries_orig': boundaries_orig,
#        'boundaries_sparse': boundaries_sparse,
#        'audio_path': file,
#        'pitch_path': pitch_file,
#        'stability_path': mask_file,
#        'raga': raga,
#        'tonic': tonic,
#        'title': title
#    }

#    out_path_mat = os.path.join(out_dir, 'self_sim.npy')
#    out_path_meta = os.path.join(out_dir, 'metadata.pkl')
#    out_path_feat = os.path.join(out_dir, "features.pyc.bz")

#    create_if_not_exists(out_dir)

#    print(f"Saving features to {out_path_feat}..")
#    save_pyc_bz(results, out_path_feat)

#    print(f"Saving self sim matrix to {out_path_mat}..")
#    np.save(out_path_mat, matrix)

#    print(f'Saving metadata to {out_path_meta}')
#    write_pkl(metadata, out_path_meta)