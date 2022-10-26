# Data augment
from compiam.utils.augment import (
    pitch_shift,
    spectral_shape,
    stroke_remix,
    time_scale,
    attack_remix,
)

path_to_audio = (
    "/Volumes/Shruti/asplab2/4way-tabla-transcription/dataset/train/audios/KB_9.wav"
)
out_dir = "/Volumes/Shruti/asplab2/compIAM/output/KB_9_test/"

pitch_shift(in_path, out_dir)
spectral_shape(in_path, out_dir)
stroke_remix(in_path, out_dir)
time_scale(in_path, out_dir)
attack_remix(in_path, out_dir)

# Train
# TODO

# Evaluate
# TODO

# Predict
from compiam import load_model

model = load_model("rhythm:4way-tabla")

path_to_audio = (
    "/Volumes/Shruti/asplab2/4way-tabla-transcription/dataset/train/audios/KB_9.wav"
)
onsets, labels = model.predict(path_to_audio)

# Visualise
from compiam.visualisation.audio import plot_waveform

plot_waveform(path_to_audio, 0, 3, dict(zip(onsets, labels)), filepath="test.png")


# Output
from compiam.io import write_csv

write_csv([onsets, labels], "test.csv", header=["onset", "label"])
