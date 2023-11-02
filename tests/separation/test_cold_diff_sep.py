import os
import pytest
import librosa

import numpy as np

import compiam
from compiam.data import TESTDIR
from compiam.exceptions import ModelNotTrainedError


def _separate():
    from compiam.separation.singing_voice_extraction import ColdDiffSep

    cold_diff_sep = ColdDiffSep()
    with pytest.raises(ModelNotTrainedError):
        cold_diff_sep.separate(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))
    cold_diff_sep.trained = True
    with pytest.raises(FileNotFoundError):
        cold_diff_sep.separate(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))
    cold_diff_sep = compiam.load_model("separation:cold-diff-sep", data_home=TESTDIR)
    separation_1 = cold_diff_sep.separate(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav"))

    audio_in, sr = librosa.load(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav"))
    separation_2 = cold_diff_sep.separate(audio_in, input_sr=sr)
    os.rmdir(os.path.join(TESTDIR, "models"))


@pytest.mark.tensorflow
def test_predict_tf():
    _separate()


@pytest.mark.essentia_tensorflow
def test_predict_ess_tf():
    _separate()


@pytest.mark.full_ml
def test_predict_full():
    _separate()


@pytest.mark.all
def test_predict_all():
    _separate()
