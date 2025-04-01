import os
import pytest
import shutil

import numpy as np

import compiam
from compiam.data import TESTDIR
from compiam.exceptions import ModelNotTrainedError


def _separate():
    from compiam.separation.singing_voice_extraction import ConvTDFVocalFineTune

    convtdf_vocal = ConvTDFVocalFineTune()
    with pytest.raises(ModelNotTrainedError):
        convtdf_vocal.separate(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))
    convtdf_vocal.trained = True
    with pytest.raises(FileNotFoundError):
        convtdf_vocal.separate(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))

    convtdf_vocal = compiam.load_model("separation:convtdf-vocal-finetune", data_home=TESTDIR)
    audio_in, sr = np.array(np.ones([1, 44100]), dtype=np.float32), 44100
    separation = convtdf_vocal.separate(audio_in, input_sr=sr)
    assert isinstance(separation, np.ndarray)
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
