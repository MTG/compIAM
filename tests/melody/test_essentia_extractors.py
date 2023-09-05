import os
import pytest

import numpy as np

from compiam.data import TESTDIR


def _predict_normalized_pitch():
    from compiam.melody.pitch_extraction import Melodia

    melodia = Melodia()
    with pytest.raises(FileNotFoundError):
        melodia.extract(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))
    pitch = melodia.extract(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
    )

    assert isinstance(pitch, np.ndarray)
    assert np.shape(pitch) == (699, 2)
    assert np.all(
        np.isclose(
            pitch[:10, 0],
            np.array(
                [
                    0.0,
                    0.00290249,
                    0.00580499,
                    0.00870748,
                    0.01160998,
                    0.01451247,
                    0.01741497,
                    0.02031746,
                    0.02321995,
                    0.02612245,
                ]
            ),
        )
    )
    assert np.all(
        np.isclose(
            pitch[140:150, 1],
            np.array(
                [
                    274.00152588,
                    270.85430908,
                    269.29431152,
                    267.74328613,
                    266.20120239,
                    263.14358521,
                    261.62796021,
                    260.12109375,
                    258.6229248,
                    257.13336182,
                ]
            ),
        )
    )

    from compiam.melody.tonic_identification import TonicIndianMultiPitch

    tonic_multipitch = TonicIndianMultiPitch()
    with pytest.raises(FileNotFoundError):
        tonic_multipitch.extract(
            os.path.join(TESTDIR, "resources", "melody", "hola.wav")
        )
    tonic = tonic_multipitch.extract(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
    )

    assert isinstance(tonic, float)
    assert tonic == 157.64892578125

    normalised_pitch = melodia.normalise_pitch(pitch, tonic)
    assert isinstance(normalised_pitch, np.ndarray)
    assert np.shape(normalised_pitch) == np.shape(pitch)
    assert np.all(
        np.isclose(
            normalised_pitch[140:150, 1],
            np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
        )
    )


@pytest.mark.essentia
def test_ess_extractors_ess():
    _predict_normalized_pitch()


@pytest.mark.essentia_tensorflow
def test_ess_extractors_ess_tf():
    _predict_normalized_pitch()


@pytest.mark.essentia_torch
def test_ess_extractors_ess_torch():
    _predict_normalized_pitch()


@pytest.mark.all
def test_ess_extractors_ess_all():
    _predict_normalized_pitch()
