import itertools

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from compiam.utils.dbn import (
    BeatStateSpace,
    BarStateSpace,
    HiddenMarkovModel,
    beat_transitions,
    dbn_beat_tracking,
    dbn_downbeat_tracking,
    dbn_bar_tracking,
    exponential_transition,
)

FPS = 100


def _activations(
    num_beats, interval=50, beats_per_bar=None, num_frames=None, floor=0.02
):
    """Clean beat (and downbeat) activation functions, with a beat every `interval` frames."""
    num_frames = num_frames or num_beats * interval
    beats = np.full(num_frames, floor)
    downbeats = np.full(num_frames, floor)
    for n in range(num_beats):
        beats[n * interval + interval // 2] = 0.95
        if beats_per_bar and n % beats_per_bar == 0:
            downbeats[n * interval + interval // 2] = 0.95
    return beats, downbeats


def test_beat_state_space():
    space = BeatStateSpace(10, 12)
    assert space.intervals.tolist() == [10, 11, 12]
    assert space.num_states == 33
    assert space.first_states.tolist() == [0, 10, 21]
    assert space.last_states.tolist() == [9, 20, 32]
    assert space.state_positions[[0, 5, 10, 21]].tolist() == [0, 0.5, 0, 0]

    # log-spaced tempi
    space = BeatStateSpace(10, 100, num_intervals=5)
    assert space.num_intervals == 5
    assert space.intervals[0] == 10 and space.intervals[-1] == 100


def test_bar_state_space():
    space = BarStateSpace(3, 10, 12)
    assert space.num_beats == 3
    assert space.num_states == 3 * 33
    assert space.state_positions.max() < 3
    assert len(space.first_states) == len(space.last_states) == 3


def test_transitions():
    prob = exponential_transition(np.array([10, 20]), np.array([10, 20]), 100)
    assert np.allclose(prob.sum(axis=1), 1)
    assert prob[0, 0] > 0.99 and prob[1, 1] > 0.99
    assert np.array_equal(
        exponential_transition(np.array([10, 20]), np.array([10, 20]), None), np.eye(2)
    )

    space = BarStateSpace(2, 10, 12)
    transitions = beat_transitions(space, 100)
    # each destination state is reached from at least one state, and probabilities add up to one
    assert transitions.shape == (space.num_states,) * 2
    assert np.allclose(transitions.sum(axis=0), 1)


def test_viterbi_matches_brute_force():
    rng = np.random.default_rng(0)
    num_states, num_frames = 4, 6
    transitions = rng.random((num_states, num_states))
    transitions[0, 2] = transitions[1, 3] = transitions[3, 0] = 0  # sparse
    transitions /= transitions.sum(axis=0, keepdims=True)  # columns: origin states
    pointers = np.array([0, 1, 1, 0])
    densities = np.log(rng.random((num_frames, 2)))

    path, log_probability = HiddenMarkovModel(
        csr_matrix(transitions), pointers
    ).viterbi(densities)

    # the prior is uniform, and it is transitioned once before the first observation
    best = (-np.inf, None)
    for candidate in itertools.product(range(num_states), repeat=num_frames + 1):
        score = -np.log(num_states)
        for t in range(num_frames):
            with np.errstate(divide="ignore"):
                score += np.log(transitions[candidate[t + 1], candidate[t]])
            score += densities[t, pointers[candidate[t + 1]]]
        if score > best[0]:
            best = (score, candidate[1:])
    assert path.tolist() == list(best[1])
    assert np.isclose(log_probability, best[0])

    # no observations
    assert (
        len(
            HiddenMarkovModel(csr_matrix(transitions), pointers).viterbi(densities[:0])[
                0
            ]
        )
        == 0
    )


def test_hmm_requires_incoming_transitions():
    transitions = csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError):
        HiddenMarkovModel(transitions, [0, 0])


def test_dbn_beat_tracking():
    beats, _ = _activations(20, interval=50)  # 120 bpm
    times = dbn_beat_tracking(beats, fps=FPS)
    expected = (np.arange(20) * 50 + 25) / FPS
    assert len(times) == 20
    assert np.allclose(times, expected, atol=0.02)

    # without aligning to the activation peaks
    times = dbn_beat_tracking(beats, fps=FPS, correct=False)
    assert abs(len(times) - 20) <= 1

    # tempo out of the allowed range
    assert len(dbn_beat_tracking(beats, fps=FPS, min_bpm=200, max_bpm=230)) != 20

    assert len(dbn_beat_tracking(np.zeros(500), fps=FPS)) == 0
    assert len(dbn_beat_tracking(np.array([]), fps=FPS)) == 0


def test_dbn_downbeat_tracking():
    beats, downbeats = _activations(24, interval=50, beats_per_bar=4)
    combined = np.vstack(
        (np.maximum(beats - downbeats, 1e-3), downbeats)
    ).T  # in (0, 1)
    output = dbn_downbeat_tracking(combined, fps=FPS, beats_per_bar=[3, 4, 5])
    assert output.shape[1] == 2
    assert len(output) == 24
    assert set(output[:, 1]) == {1, 2, 3, 4}
    assert np.allclose(
        output[output[:, 1] == 1, 0] % 2.0, 0.25
    )  # downbeats every 4 beats

    assert dbn_downbeat_tracking(
        np.zeros((500, 2)), fps=FPS, beats_per_bar=[4]
    ).shape == (0, 2)
    assert dbn_downbeat_tracking(
        np.zeros((0, 2)), fps=FPS, beats_per_bar=[4]
    ).shape == (0, 2)


def test_dbn_bar_tracking():
    beats = np.arange(1, 41) * 0.5
    activations = np.full(len(beats), 0.05)
    activations[::4] = 0.9

    for beats_per_bar in ([4], [3, 4], [4, 5, 6]):
        output = dbn_bar_tracking(
            beats,
            activations,
            beats_per_bar=beats_per_bar,
            meter_change_prob=1e-3,
            observation_weight=4,
        )
        assert output.shape == (40, 2)
        assert np.array_equal(output[:, 0], beats)
        assert output[:, 1].tolist() == [1, 2, 3, 4] * 10

    # not enough beats
    assert dbn_bar_tracking(beats[:1], activations[:1], beats_per_bar=[4]).shape == (
        0,
        2,
    )
    assert dbn_bar_tracking([], [], beats_per_bar=[4]).shape == (0, 2)
