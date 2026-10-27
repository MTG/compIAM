import os
import sys
import shutil
import pytest
import librosa

import numpy as np

import compiam
from compiam.data import TESTDIR
from compiam.exceptions import ModelNotTrainedError

BEAT_TEST = os.path.join(TESTDIR, "resources", "rhythm", "beat_test.wav")


def _predict(post_processor):
    from compiam.rhythm.meter.tcn_carnatic import TCNTracker

    tracker = TCNTracker(post_processor=post_processor)
    with pytest.raises(ModelNotTrainedError):
        tracker.predict(os.path.join(TESTDIR, "resources", "rhythm", "hola.wav"))
    tracker.trained = True
    with pytest.raises(FileNotFoundError):
        tracker.predict(os.path.join(TESTDIR, "resources", "rhythm", "hola.wav"))
    with pytest.raises(ValueError):
        tracker.predict(42)

    beats = tracker.predict(BEAT_TEST)

    audio_in, sr = librosa.load(BEAT_TEST)
    beats_2 = tracker.predict(audio_in, sr)
    beats_3 = tracker.predict(np.stack([audio_in, audio_in]), sr)

    for output in [beats, beats_2, beats_3]:
        assert isinstance(output, np.ndarray)
        assert output.ndim == 2 and output.shape[1] == 2

    # other sampling rates are resampled
    audio_48k, sr = librosa.load(
        os.path.join(TESTDIR, "resources", "rhythm", "48k.wav"), sr=48000
    )
    beats_4 = tracker.predict(audio_48k, sr)
    assert isinstance(beats_4, np.ndarray)
    assert beats_4.ndim == 2 and beats_4.shape[1] == 2

    # madmom is not needed
    assert "madmom" not in sys.modules


def _invalid_arguments():
    from compiam.rhythm.meter.tcn_carnatic import TCNTracker

    with pytest.raises(ValueError):
        TCNTracker(post_processor="hola")
    with pytest.raises(ValueError):
        TCNTracker(model_version=1)


def _predict_pretrained():
    from compiam.rhythm.meter.tcn_carnatic.post import joint_tracker, sequential_tracker

    tracker = compiam.load_model("rhythm:tcn-carnatic", data_home=TESTDIR)
    assert tracker.trained

    for post_processor, beats_per_bar in [
        (joint_tracker, (3, 5, 7, 8)),
        (sequential_tracker, (4,)),
    ]:
        tracker.post_processor = post_processor
        beats = tracker.predict(BEAT_TEST, beats_per_bar=beats_per_bar)
        assert beats.ndim == 2 and beats.shape[1] == 2
        assert len(beats) > 0
        assert np.all(np.diff(beats[:, 0]) > 0)
        assert np.all(beats[:, 1] >= 1)

    tracker.save_beats(beats, os.path.join(TESTDIR, "beats.csv"))
    assert os.path.exists(os.path.join(TESTDIR, "beats.csv"))
    os.remove(os.path.join(TESTDIR, "beats.csv"))
    shutil.rmtree(os.path.join(TESTDIR, "models"))


@pytest.mark.torch
def test_predict_torch():
    _predict("joint")
    _predict("sequential")
    _invalid_arguments()


@pytest.mark.full_ml
def test_predict_full():
    _predict("joint")
    _predict("sequential")
    _invalid_arguments()


@pytest.mark.all
def test_predict_all():
    _predict("joint")
    _predict("sequential")
    _invalid_arguments()
    _predict_pretrained()
