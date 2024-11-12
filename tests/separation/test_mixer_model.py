import os
import pytest
import shutil

import numpy as np

import compiam
from compiam.data import TESTDIR
from compiam.exceptions import ModelNotTrainedError


def _separate():
    from compiam.separation.music_source_separation import MixerModel

    mixer_model = MixerModel()
    with pytest.raises(ModelNotTrainedError):
        mixer_model.separate(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))
    mixer_model.trained = True
    with pytest.raises(FileNotFoundError):
        mixer_model.separate(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))

    mixer_model = compiam.load_model("separation:mixer-model", data_home=TESTDIR)
    audio_in, sr = np.array(np.ones([2, 44150 * 10]), dtype=np.float32), 44100
    separation = mixer_model.separate(audio_in, input_sr=sr)
    assert isinstance(separation, tuple)
    assert isinstance(separation[0], np.ndarray)
    assert isinstance(separation[1], np.ndarray)
    shutil.rmtree(os.path.join(TESTDIR, "models"))


@pytest.mark.torch
def test_predict_torch():
    _separate()


@pytest.mark.full_ml
def test_predict_full():
    _separate()


@pytest.mark.all
def test_predict_all():
    _separate()
