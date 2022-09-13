# Data Load
data = load_data('data_name')

# Data augment
from compiam.utils.augment import pitch_shift, spectral_shape, stroke_remix, time_scale, attack_remix
from compiam.utils.core import train_test_split

pitch_shift(data)

# Train/Test split


# Train


# Evaluate


# Predict
from compiam.rhythm.tabla_transcription.models import load_model

model = load_model('model_name')

# Output

