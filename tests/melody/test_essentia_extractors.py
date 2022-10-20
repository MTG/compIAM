import os
import pytest

from compiam.data import WORKDIR
from compiam import load_model


def _predict_normalized_pitch():
    melodia = load_model("melody:melodia")
    with pytest.raises(ValueError):
        melodia.extract(
            os.path.join(WORKDIR, "tests", "resources", "melody", "hola.wav")
        )
    # pitch = melodia.predict(os.path.join(WORKDIR, "tests", "resources", \
    #    "melody", "test.wav"))

    # assert pitch here

    tonic_multipitch = load_model("melody:tonic-multipitch")
    with pytest.raises(ValueError):
        tonic_multipitch.extract(
            os.path.join(WORKDIR, "tests", "resources", "melody", "hola.wav")
        )
    # tonic = tonic_multipitch.predict(os.path.join(WORKDIR, "tests", "resources", \
    #    "melody", "test.wav"))

    # assert tonic here

    # assert normalisation here


@pytest.mark.essentia
def test_ess_extractors_ess():
    _predict_normalized_pitch()


@pytest.mark.essentia
def test_ess_extractors_ess_tf():
    _predict_normalized_pitch()


@pytest.mark.essentia
def test_ess_extractors_ess_torch():
    _predict_normalized_pitch()


@pytest.mark.all
def test_ess_extractors_ess_torch():
    _predict_normalized_pitch()
