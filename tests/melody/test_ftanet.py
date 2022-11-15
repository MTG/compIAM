import os
import pytest

import numpy as np

from compiam import load_model
from compiam.melody.tonic_identification import TonicIndianMultiPitch
from compiam.data import WORKDIR


def _predict_pitch():
    ftanet = load_model("melody:ftanet-carnatic")
    with pytest.raises(ValueError):
        ftanet.predict(
            os.path.join(WORKDIR, "tests", "resources", "melody", "hola.wav")
        )
    pitch = ftanet.predict(os.path.join(WORKDIR, "tests", "resources", \
        "melody", "test.wav"))

    assert isinstance(pitch, np.array)
    assert np.shape(pitch) == (202, 2)
    assert pitch[:10, 0] == np.array([0., 0.01007774, 0.02015547, 0.03023321, 0.04031095,
       0.05038868, 0.06046642, 0.07054415, 0.08062189, 0.09069963])
    assert pitch[140:150, 1] == np.array([354., 354., 354., 354., 358., 358., 363., 367., 
        371., 375.])


def _predict_normalized_pitch():
    ftanet = load_model("melody:ftanet-carnatic")
    with pytest.raises(ValueError):
        ftanet.predict(
            os.path.join(WORKDIR, "tests", "resources", "melody", "hola.wav")
        )
    # pitch = ftanet.predict(os.path.join(WORKDIR, "tests", "resources", \
    #    "melody", "test.wav"))

    # assert pitch

    tonic_multipitch = TonicIndianMultiPitch()
    with pytest.raises(ValueError):
        tonic_multipitch.extract(
            os.path.join(WORKDIR, "tests", "resources", "melody", "hola.wav")
        )
    # tonic = tonic_multipitch.predict(os.path.join(WORKDIR, "tests", "resources", \
    #    "melody", "test.wav"))

    # assert tonic here

    # assert normalisation here


@pytest.mark.tensorflow
def test_predict_tf():
    _predict_pitch()


@pytest.mark.essentia_tensorflow
def test_predict_ess_tf():
    _predict_normalized_pitch()


@pytest.mark.full_ml
def test_predict_full():
    _predict_pitch()


@pytest.mark.all
def test_predict_full():
    _predict_pitch()
