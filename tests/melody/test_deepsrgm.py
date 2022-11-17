import os
import pytest

import numpy as np

from compiam import load_model
from compiam.data import WORKDIR


def _load_model():
    deepsrgm = load_model("melody:deepsrgm")
    with pytest.raises(FileNotFoundError):
        deepsrgm.predict(
            os.path.join(WORKDIR, "tests", "resources", "melody", "hola.wav")
        )
    with pytest.raises(ValueError):
        deepsrgm.predict(
            os.path.join(WORKDIR, "tests", "resources", "melody", "pitch_test.wav")
        )
    raga_mapping = deepsrgm.mapping
    assert raga_mapping == {
        0: 'Bhairav',
        1: 'Madhukauns',
        2: 'Mōhanaṁ',
        3: 'Hamsadhvāni',
        4: 'Varāḷi',
        5: 'Dēś',
        6: 'Kamās',
        7: 'Yaman kalyāṇ',
        8: 'Bilahari',
        9: 'Ahira bhairav'
    }


def _get_features():
    deepsrgm = load_model("melody:deepsrgm")
    with pytest.raises(ValueError):
        feat = deepsrgm.get_features(
            os.path.join(WORKDIR, "tests", "resources", "melody", "hola.wav")
        )
    

@pytest.mark.torch
def test_predict_tf():
    _load_model()


@pytest.mark.essentia_torch
def test_predict_ess_tf():
    _load_model()
    _get_features()


@pytest.mark.full_ml
def test_predict_full():
    _load_model()


@pytest.mark.all
def test_predict_all():
    _load_model()
    _get_features()
