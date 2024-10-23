import os
import pytest
import librosa

import numpy as np

from compiam.data import TESTDIR
from compiam.exceptions import ModelNotTrainedError


def _predict_pitch():
    from compiam.melody.pitch_extraction import FTAResNetCarnatic

    ftaresnet = FTAResNetCarnatic()
    with pytest.raises(ModelNotTrainedError):
        ftaresnet.predict(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))
    ftaresnet.trained = True
    with pytest.raises(FileNotFoundError):
        ftaresnet.predict(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))
    pitch = ftaresnet.predict(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
    )

    audio_in, sr = librosa.load(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
    )
    pitch_2 = ftaresnet.predict(audio_in, input_sr=sr)

    assert isinstance(pitch, np.ndarray)
    assert isinstance(pitch_2, np.ndarray)
    assert np.shape(pitch) == (128, 2)
    assert np.shape(pitch_2) == (128, 2)

    assert np.all(
        np.isclose(
            pitch[:10, 0],
            np.array(
                [
                    0.0,
                    0.01007874,
                    0.02015748,
                    0.03023622,
                    0.04031496,
                    0.0503937,
                    0.06047244,
                    0.07055118,
                    0.08062992,
                    0.09070867,
                ]
            ),
        )
    )

    pitch = ftaresnet.predict(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav"),
        out_step=0.001,
    )

    print(pitch.shape)
    print(pitch[:10, 0].shape)

    assert np.all(
        np.isclose(
            pitch[:10, 0],
            np.array(
                [
                    0.0,
                    0.00063241,
                    0.00126482,
                    0.00189723,
                    0.00252964,
                    0.00316206,
                    0.00379447,
                    0.00442688,
                    0.00505929,
                    0.0056917,
                ]
            ),
        )
    )


def _predict_normalized_pitch():
    from compiam.melody.pitch_extraction import FTAResNetCarnatic

    ftaresnet = FTAResNetCarnatic()
    ftaresnet.trained = True
    with pytest.raises(FileNotFoundError):
        ftaresnet.predict(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))
    pitch = ftaresnet.predict(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
    )

    from compiam.melody.tonic_identification import TonicIndianMultiPitch

    tonic_multipitch = TonicIndianMultiPitch()
    tonic = tonic_multipitch.extract(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
    )

    assert isinstance(tonic, float)
    assert tonic == 157.64892578125

    normalised_pitch = ftaresnet.normalise_pitch(pitch, tonic)
    assert isinstance(normalised_pitch, np.ndarray)
    assert np.shape(normalised_pitch) == np.shape(pitch)


@pytest.mark.torch
def test_predict_torch():
    _predict_pitch()


@pytest.mark.essentia_torch
def test_predict_ess_torch():
    _predict_pitch()
    _predict_normalized_pitch()


@pytest.mark.full_ml
def test_predict_full():
    _predict_pitch()


@pytest.mark.all
def test_predict_all():
    _predict_pitch()
    _predict_normalized_pitch()
