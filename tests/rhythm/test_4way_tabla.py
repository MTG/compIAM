import os
import pytest

import numpy as np

from compiam import load_model
from compiam.data import TESTDIR


def _test_model():
    one_way_tabla = load_model("rhythm:1way-tabla")
    four_way_tabla = load_model("rhythm:4way-tabla")
    assert one_way_tabla.categories == ["D", "RT", "RB", "B"]
    assert four_way_tabla.categories == ["D", "RT", "RB", "B"]

    with pytest.raises(FileNotFoundError):
        four_way_tabla.predict(
            os.path.join(TESTDIR, "resources", "melody", "hola.wav")
        )

    onsets, labels = four_way_tabla.predict(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
    )
    assert list(onsets) == []
    assert list(labels) == []

    

@pytest.mark.torch
def test_predict_tf():
    _test_model()


@pytest.mark.essentia_torch
def test_predict_ess_tf():
    _test_model()


@pytest.mark.full_ml
def test_predict_full():
    _test_model()


@pytest.mark.all
def test_predict_all():
    _test_model()
