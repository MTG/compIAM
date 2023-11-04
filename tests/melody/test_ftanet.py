import os
import pytest
import librosa

import numpy as np

from compiam.data import TESTDIR
from compiam.exceptions import ModelNotTrainedError


def _predict_pitch():
    from compiam.melody.pitch_extraction import FTANetCarnatic

    ftanet = FTANetCarnatic()
    with pytest.raises(ModelNotTrainedError):
        ftanet.predict(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))
    ftanet.trained = True
    with pytest.raises(FileNotFoundError):
        ftanet.predict(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))
    pitch = ftanet.predict(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
    )

    audio_in, sr = librosa.load(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
    )
    pitch_2 = ftanet.predict(audio_in, input_sr=sr)

    assert np.all(np.isclose(pitch, pitch_2))

    assert isinstance(pitch, np.ndarray)
    assert np.shape(pitch) == (202, 2)
    assert np.all(
        np.isclose(
            pitch[:10, 0],
            np.array(
                [
                    0.0,
                    0.01007774,
                    0.02015547,
                    0.03023321,
                    0.04031095,
                    0.05038868,
                    0.06046642,
                    0.07054415,
                    0.08062189,
                    0.09069963,
                ]
            ),
        )
    )
    assert np.all(
        np.isclose(
            pitch[140:150, 1],
            np.array(
                # [354.0, 350.0, 350.0, 354.0, 354.0, 358.0, 367.0, 371.0, 375.0, 375.0]
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
        )
    )

    pitch = ftanet.predict(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav"),
        out_step=0.001,
    )

    assert np.all(
        np.isclose(
            pitch[:10, 0],
            np.array(
                [
                    0.0,
                    0.0010008,
                    0.00200161,
                    0.00300241,
                    0.00400321,
                    0.00500401,
                    0.00600482,
                    0.00700562,
                    0.00800642,
                    0.00900723,
                ]
            ),
        )
    )
    assert np.all(
        np.isclose(
            pitch[1000:1010, 1],
            np.array(
                # [327.0, 327.0, 327.0, 327.0, 327.0, 327.0, 327.0, 327.0, 327.0, 327.0]
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
        )
    )


def _predict_normalized_pitch():
    from compiam.melody.pitch_extraction import FTANetCarnatic

    ftanet = FTANetCarnatic()
    ftanet.trained = True
    with pytest.raises(FileNotFoundError):
        ftanet.predict(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))
    pitch = ftanet.predict(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
    )

    from compiam.melody.tonic_identification import TonicIndianMultiPitch

    tonic_multipitch = TonicIndianMultiPitch()
    tonic = tonic_multipitch.extract(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
    )

    assert isinstance(tonic, float)
    assert tonic == 157.64892578125

    normalised_pitch = ftanet.normalise_pitch(pitch, tonic)
    assert isinstance(normalised_pitch, np.ndarray)
    assert np.shape(normalised_pitch) == np.shape(pitch)


@pytest.mark.tensorflow
def test_predict_tf():
    _predict_pitch()


@pytest.mark.essentia_tensorflow
def test_predict_ess_tf():
    _predict_pitch()
    _predict_normalized_pitch()


@pytest.mark.full_ml
def test_predict_full():
    _predict_pitch()


@pytest.mark.all
def test_predict_all():
    _predict_pitch()
    _predict_normalized_pitch()
