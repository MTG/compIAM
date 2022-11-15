import os
import pytest

import numpy as np

from compiam.melody.pitch_extraction import Melodia
from compiam.melody.tonic_identification import TonicIndianMultiPitch
from compiam.data import WORKDIR


def _predict_normalized_pitch():
    melodia = Melodia()
    with pytest.raises(ValueError):
        melodia.extract(
            os.path.join(WORKDIR, "tests", "resources", "melody", "hola.wav")
        )
    pitch = melodia.extract(os.path.join(WORKDIR, "tests", "resources", \
        "melody", "pitch_test.wav"))

    assert isinstance(pitch, np.array)
    assert np.shape(pitch) == (699, 2)
    assert pitch[:10, 0] == np.array([0., 0.00290249, 0.00580499, 0.00870748, 0.01160998,
       0.01451247, 0.01741497, 0.02031746, 0.02321995, 0.02612245])
    assert pitch[140:150, 1] == np.array([274.00152588, 270.85430908, 269.29431152, 267.74328613,
       266.20120239, 263.14358521, 261.62796021, 260.12109375, 258.6229248, 257.13336182])

    tonic_multipitch = TonicIndianMultiPitch()
    with pytest.raises(ValueError):
        tonic_multipitch.extract(
            os.path.join(WORKDIR, "tests", "resources", "melody", "hola.wav")
        )
    tonic = tonic_multipitch.extract(os.path.join(WORKDIR, "tests", "resources", \
        "melody", "pitch_test.wav"))

    assert isinstance(tonic, float)
    assert tonic == 157.64892578125

    normalised_pitch = melodia.normalise_pitch(pitch, tonic)
    assert isinstance(normalised_pitch, np.array)
    assert np.shape(normalised_pitch) == np.shape(pitch)
    assert normalised_pitch[:10, 0] == pitch[:10, 0]
    assert pitch[140:150, 1] == np.array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4.])


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
