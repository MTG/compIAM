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
    from compiam.rhythm.meter.beat_this_carnatic import BeatThisCarnatic

    tracker = BeatThisCarnatic(post_processor=post_processor)
    with pytest.raises(ModelNotTrainedError):
        tracker.predict(os.path.join(TESTDIR, "resources", "rhythm", "hola.wav"))
    tracker.trained = True
    with pytest.raises(FileNotFoundError):
        tracker.predict(os.path.join(TESTDIR, "resources", "rhythm", "hola.wav"))
    with pytest.raises(ValueError):
        tracker.predict(42)
    with pytest.raises(ValueError):
        tracker.predict(np.ones((2, 2, 2)))

    beats = tracker.predict(BEAT_TEST)

    audio_in, sr = librosa.load(BEAT_TEST)
    beats_2 = tracker.predict(audio_in, sr)
    beats_3 = tracker.predict(np.stack([audio_in, audio_in]), sr)

    # the same audio, at a different sampling rate
    audio_48k, sr = librosa.load(BEAT_TEST, sr=48000)
    beats_4 = tracker.predict(audio_48k, sr)

    # audio longer than a chunk of the model
    beats_5 = tracker.predict(np.tile(audio_in, 40), sr=22050)

    for output in [beats, beats_2, beats_3, beats_4, beats_5]:
        assert isinstance(output, np.ndarray)
        assert output.ndim == 2 and output.shape[1] == 2

    # madmom is not needed
    assert "madmom" not in sys.modules


def _invalid_arguments():
    from compiam.rhythm.meter.beat_this_carnatic import BeatThisCarnatic

    with pytest.raises(ValueError):
        BeatThisCarnatic(post_processor="hola")


def _load_weights(tmp_path):
    import torch
    from compiam.rhythm.meter.beat_this_carnatic import BeatThisCarnatic

    tracker = BeatThisCarnatic()
    state_dict = tracker.model.state_dict()
    model_path = str(tmp_path)

    # weights saved from the training module have the "model." prefix
    torch.save(
        {"state_dict": {"model." + k: v for k, v in state_dict.items()}},
        os.path.join(model_path, tracker.model_version),
    )
    loaded = BeatThisCarnatic(model_path=model_path)
    assert loaded.trained
    for key, value in loaded.model.state_dict().items():
        assert torch.equal(value, state_dict[key])

    # plain state dicts
    torch.save(state_dict, os.path.join(model_path, tracker.model_version))
    assert BeatThisCarnatic(model_path=model_path).trained


def _predict_pretrained():
    tracker = compiam.load_model("rhythm:beat-this-carnatic", data_home=TESTDIR)
    assert tracker.trained

    beats = tracker.predict(BEAT_TEST)
    assert beats.ndim == 2 and beats.shape[1] == 2
    assert len(beats) > 0
    assert np.all(np.diff(beats[:, 0]) > 0)

    tracker.save_beats(beats, os.path.join(TESTDIR, "beats.csv"))
    assert os.path.exists(os.path.join(TESTDIR, "beats.csv"))
    os.remove(os.path.join(TESTDIR, "beats.csv"))
    shutil.rmtree(os.path.join(TESTDIR, "models"))


def _frames(peaks, length=100):
    import torch

    logits = torch.full((length,), -10.0)
    logits[list(peaks)] = 10.0
    return logits


def _post_processing():
    from compiam.rhythm.meter.beat_this_carnatic.post import (
        beat_positions,
        dbn_tracker,
        deduplicate_peaks,
        minimal_tracker,
    )

    assert deduplicate_peaks([]).tolist() == []
    assert deduplicate_peaks([7]).tolist() == [7]
    assert deduplicate_peaks([3, 4, 20]).tolist() == [3.5, 20]

    # 8 beats every 0.5 seconds, with a downbeat every 4 beats
    beats = _frames(range(10, 90, 10), 100)
    downbeats = _frames([10, 50], 100)
    output = minimal_tracker(beats, downbeats, fps=20)
    assert output[:, 0].tolist() == [0.5 * n + 0.5 for n in range(8)]
    assert output[:, 1].tolist() == [1, 2, 3, 4] * 2

    # downbeats are moved to the closest beat
    output = minimal_tracker(beats, _frames([12], 100), fps=20)
    assert output[:, 1].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]

    # no beats, or no downbeats
    assert minimal_tracker(_frames([]), _frames([])).shape == (0, 2)
    assert minimal_tracker(beats, _frames([])).shape == (8, 2)

    # positions in the bar
    times = np.arange(9) * 0.5
    # the beats before the first downbeat are counted backwards, using the length of the first bar
    assert beat_positions(times, times[[3, 7]]).tolist() == [2, 3, 4, 1, 2, 3, 4, 1, 2]
    assert beat_positions(times, times[[2, 6]]).tolist() == [3, 4, 1, 2, 3, 4, 1, 2, 3]
    # unknown length of the bars
    assert beat_positions(times, times[[4]]).tolist() == [0, 0, 0, 0, 1, 2, 3, 4, 5]
    assert beat_positions(times, np.array([])).tolist() == [0] * 9

    # dbn
    beats = _frames(range(25, 25 + 50 * 24, 50), 1200)
    downbeats = _frames(range(25, 25 + 50 * 24, 200), 1200)
    output = dbn_tracker(beats, downbeats, fps=100, beats_per_bar=(3, 4, 5))
    assert output.shape[1] == 2 and len(output) == 24
    assert set(output[:, 1]) == {1, 2, 3, 4}


def _chunking():
    import torch
    from compiam.rhythm.meter.beat_this_carnatic.utils import (
        split_spectrogram,
        merge_chunks,
    )

    chunk_size, border_size = 50, 6
    # frames shorter, similar and longer than a chunk
    for length in [10, 38, 44, 50, 51, 87, 88, 100, 333, 1000]:
        spect = torch.arange(length, dtype=torch.float32)[:, None].repeat(1, 3)
        chunks, starts = split_spectrogram(spect, chunk_size, border_size=border_size)
        assert all(len(chunk) <= chunk_size for chunk in chunks)
        # a model returning the frame index reconstructs the whole piece
        predictions = [chunk[:, 0] for chunk in chunks]
        merged = merge_chunks(predictions, starts, length, chunk_size, border_size)
        assert torch.equal(merged, spect[:, 0])


@pytest.mark.torch
def test_predict_torch(tmp_path):
    _predict("minimal")
    _predict("dbn")
    _invalid_arguments()
    _load_weights(tmp_path)
    _post_processing()
    _chunking()


@pytest.mark.full_ml
def test_predict_full(tmp_path):
    _predict("minimal")
    _predict("dbn")
    _invalid_arguments()
    _load_weights(tmp_path)
    _post_processing()
    _chunking()


@pytest.mark.all
def test_predict_all(tmp_path):
    _predict("minimal")
    _predict("dbn")
    _invalid_arguments()
    _load_weights(tmp_path)
    _post_processing()
    _chunking()
    _predict_pretrained()
