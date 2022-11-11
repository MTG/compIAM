import os
import pytest

from compiam import load_model
from compiam.melody.tonic_identification import TonicIndianMultiPitch
from compiam.data import WORKDIR


def _predict_pitch():
    ftanet = load_model("melody:ftanet-carnatic")
    with pytest.raises(ValueError):
        ftanet.predict(
            os.path.join(WORKDIR, "tests", "resources", "melody", "hola.wav")
        )
    # pitch = ftanet.predict(os.path.join(WORKDIR, "tests", "resources", \
    #    "melody", "test.wav"))

    # assert pitch


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
