"""Dynamic Bayesian Networks (DBN) for beat, downbeat and bar tracking.

Self-contained numpy/scipy implementation of the DBN post-processing described in

    Böck, S., Krebs, F., & Widmer, G. (2016). Joint beat and downbeat tracking with recurrent
    neural networks. In Proceedings of the 17th International Society for Music Information
    Retrieval Conference (ISMIR 2016).

    Krebs, F., Böck, S., & Widmer, G. (2015). An efficient state-space model for joint tempo
    and meter tracking. In Proceedings of the 16th International Society for Music Information
    Retrieval Conference (ISMIR 2015).

Beat, downbeat and bar tracking are exposed via :func:`dbn_beat_tracking`,
:func:`dbn_downbeat_tracking` and :func:`dbn_bar_tracking`.
"""

import warnings

import numpy as np
from scipy.sparse import csr_matrix


class BeatStateSpace:
    """State space of a single beat period, for a range of tempi (beat intervals in frames)."""

    def __init__(self, min_interval, max_interval, num_intervals=None):
        """Beat state space init method.

        :param min_interval: minimum beat interval (in frames).
        :param max_interval: maximum beat interval (in frames).
        :param num_intervals: number of (log-spaced) intervals, if None all integer intervals are used.
        """
        intervals = np.arange(np.round(min_interval), np.round(max_interval) + 1)
        if num_intervals is not None and num_intervals < len(intervals):
            num_log_intervals = num_intervals
            intervals = []
            while len(intervals) < num_intervals:
                intervals = np.logspace(
                    np.log2(min_interval),
                    np.log2(max_interval),
                    num_log_intervals,
                    base=2,
                )
                intervals = np.unique(np.round(intervals))
                num_log_intervals += 1

        self.intervals = np.ascontiguousarray(intervals, dtype=np.int64)
        self.num_states = int(np.sum(intervals))
        self.num_intervals = len(intervals)
        self.first_states = np.cumsum(np.r_[0, self.intervals[:-1]]).astype(np.int64)
        self.last_states = np.cumsum(self.intervals) - 1

        self.state_positions = np.empty(self.num_states)
        self.state_intervals = np.empty(self.num_states, dtype=np.int64)
        idx = 0
        for interval in self.intervals:
            self.state_positions[idx : idx + interval] = np.linspace(
                0, 1, interval, endpoint=False
            )
            self.state_intervals[idx : idx + interval] = interval
            idx += interval


class BarStateSpace:
    """State space of a bar, i.e. `num_beats` consecutive beat state spaces."""

    def __init__(self, num_beats, min_interval, max_interval, num_intervals=None):
        """Bar state space init method.

        :param num_beats: number of beats in the bar.
        :param min_interval: minimum beat interval (in frames).
        :param max_interval: maximum beat interval (in frames).
        :param num_intervals: number of (log-spaced) intervals, if None all integer intervals are used.
        """
        self.num_beats = int(num_beats)
        self.num_states = 0
        self.first_states = []
        self.last_states = []
        state_positions = []
        state_intervals = []

        beat_space = BeatStateSpace(min_interval, max_interval, num_intervals)
        for beat in range(self.num_beats):
            state_positions.append(beat_space.state_positions + beat)
            state_intervals.append(beat_space.state_intervals)
            self.first_states.append(beat_space.first_states + self.num_states)
            self.last_states.append(beat_space.last_states + self.num_states)
            self.num_states += beat_space.num_states

        self.state_positions = np.hstack(state_positions)
        self.state_intervals = np.hstack(state_intervals)


def exponential_transition(from_intervals, to_intervals, transition_lambda):
    """Tempo transition probabilities between two sets of beat intervals.

    :param from_intervals: beat intervals of the origin states.
    :param to_intervals: beat intervals of the destination states.
    :param transition_lambda: rate of the exponential distribution, higher values favour a
        constant tempo. If None, only transitions between equal intervals are allowed.

    :returns: 2-D array (origin x destination) with the row-normalised transition probabilities.
    """
    if transition_lambda is None:
        return np.eye(len(from_intervals), len(to_intervals))
    ratio = (
        to_intervals.astype(np.float64)
        / from_intervals.astype(np.float64)[:, np.newaxis]
    )
    prob = np.exp(-transition_lambda * np.abs(ratio - 1.0))
    prob[prob <= np.spacing(1)] = 0
    return prob / np.sum(prob, axis=1)[:, np.newaxis]


def beat_transitions(state_space, transition_lambda):
    """Transition matrix of a beat or bar state space.

    States advance deterministically within a beat, and the tempo can change when entering a new beat.

    :param state_space: a BeatStateSpace or BarStateSpace.
    :param transition_lambda: rate of the exponential tempo transition distribution.

    :returns: sparse matrix with destination states in rows and origin states in columns.
    """
    if isinstance(state_space, BeatStateSpace):
        first_states, last_states = [state_space.first_states], [
            state_space.last_states
        ]
    else:
        first_states, last_states = state_space.first_states, state_space.last_states

    all_first = np.concatenate(first_states)
    states = np.setdiff1d(np.arange(state_space.num_states), all_first)
    to_list, from_list, prob_list = [states], [states - 1], [np.ones(len(states))]
    for beat, to_states in enumerate(first_states):
        from_states = last_states[beat - 1]
        prob = exponential_transition(
            state_space.state_intervals[from_states],
            state_space.state_intervals[to_states],
            transition_lambda,
        )
        from_idx, to_idx = np.nonzero(prob)
        to_list.append(to_states[to_idx])
        from_list.append(from_states[from_idx])
        prob_list.append(prob[prob != 0])
    return _sparse_transitions(
        np.hstack(to_list),
        np.hstack(from_list),
        np.hstack(prob_list),
        state_space.num_states,
    )


def _sparse_transitions(to_states, from_states, probabilities, num_states):
    return csr_matrix(
        (probabilities, (to_states, from_states)), shape=(num_states, num_states)
    )


class HiddenMarkovModel:
    """Hidden Markov Model with sparse transitions and shared observation densities.

    Only the Viterbi decoding is implemented. The prior is a uniform distribution over the states,
    that is transitioned once before the first observation.
    """

    def __init__(self, transitions, observation_pointers):
        """Hidden Markov Model init method.

        :param transitions: sparse matrix with the transition probabilities, destination states
            in rows and origin states in columns.
        :param observation_pointers: for every state, the column of the observation log-densities
            it uses.
        """
        transitions = csr_matrix(transitions)
        transitions.sort_indices()
        self.num_states = transitions.shape[0]
        self.observation_pointers = np.asarray(observation_pointers, dtype=np.int64)

        counts = np.diff(transitions.indptr)
        if np.any(counts == 0):
            raise ValueError("All states need at least one incoming transition.")
        log_probs = np.log(transitions.data)

        # most states have a single predecessor, those are just gathered while decoding
        single = counts == 1
        self._single = np.flatnonzero(single)
        single_idx = transitions.indptr[self._single]
        self._single_prev = transitions.indices[single_idx].astype(np.int64)
        self._single_log_prob = log_probs[single_idx]
        # states that follow their preceding state with probability 1 can be updated by shifting
        self._shift = bool(
            np.all(self._single_prev == self._single - 1)
            and np.all(self._single_log_prob == 0)
        )

        # states with several predecessors need an actual maximisation and a back-pointer,
        # their predecessors are stored in a padded array (missing ones have -inf log-probability)
        self._multi = np.flatnonzero(~single)
        self._multi_index = np.full(self.num_states, -1, dtype=np.int64)
        self._multi_index[self._multi] = np.arange(len(self._multi))
        multi_counts = counts[self._multi]
        width = int(multi_counts.max()) if len(multi_counts) else 0
        valid = np.arange(width) < multi_counts[:, np.newaxis]
        gather = np.concatenate(
            [
                np.arange(transitions.indptr[s], transitions.indptr[s + 1])
                for s in self._multi
            ]
            + [np.empty(0, dtype=np.int64)]
        ).astype(np.int64)
        self._multi_prev = np.zeros((len(self._multi), width), dtype=np.int64)
        self._multi_prev[valid] = transitions.indices[gather]
        self._multi_log_prob = np.full((len(self._multi), width), -np.inf)
        self._multi_log_prob[valid] = log_probs[gather]
        self._multi_rows = np.arange(len(self._multi))

    def viterbi(self, log_densities):
        """Most likely state sequence given the observation log-densities.

        :param log_densities: 2-D array (frames x densities) with the observation log-densities.

        :returns: tuple with the state path and its log-probability.
        """
        num_frames = len(log_densities)
        if num_frames == 0:
            return np.empty(0, dtype=np.int64), -np.inf

        densities = np.asarray(log_densities, dtype=np.float64)[
            :, self.observation_pointers
        ]
        previous = np.full(self.num_states, -np.log(self.num_states))
        current = np.empty(self.num_states)
        backtrack = np.empty((num_frames, len(self._multi)), dtype=np.int32)
        has_multi = len(self._multi) > 0

        for frame in range(num_frames):
            if self._shift:
                current[1:] = previous[:-1]
            else:
                current[self._single] = (
                    previous[self._single_prev] + self._single_log_prob
                )
            if has_multi:
                candidates = previous[self._multi_prev] + self._multi_log_prob
                # first predecessor reaching the maximum
                best = candidates.argmax(axis=1)
                current[self._multi] = candidates[self._multi_rows, best]
                backtrack[frame] = self._multi_prev[self._multi_rows, best]
            current += densities[frame]
            previous, current = current, previous

        state = int(np.argmax(previous))
        log_probability = previous[state]
        if np.isinf(log_probability):
            warnings.warn("The sequence has -inf probability, no path found.")
            return np.empty(0, dtype=np.int64), log_probability

        single_lookup = np.zeros(self.num_states, dtype=np.int64)
        single_lookup[self._single] = self._single_prev
        path = np.empty(num_frames, dtype=np.int64)
        for frame in range(num_frames - 1, -1, -1):
            path[frame] = state
            multi = self._multi_index[state]
            state = backtrack[frame, multi] if multi >= 0 else single_lookup[state]
        return path, log_probability


def _trim_activations(activations, threshold):
    """Remove leading and trailing frames of the activations below the threshold."""
    first = 0
    if threshold:
        idx = np.nonzero(activations >= threshold)[0]
        if idx.any():
            first = max(first, np.min(idx))
            last = min(len(activations), np.max(idx) + 1)
        else:
            last = first
        activations = activations[first:last]
    return activations, first


def _segments(mask):
    """Start and end (exclusive) indices of the runs of True in a boolean array."""
    idx = np.nonzero(np.diff(mask.astype(np.int64)))[0] + 1
    if mask[0]:
        idx = np.r_[0, idx]
    if mask[-1]:
        idx = np.r_[idx, mask.size]
    return idx.reshape((-1, 2))


def dbn_beat_tracking(
    activations,
    fps,
    min_bpm=55.0,
    max_bpm=215.0,
    num_tempi=None,
    transition_lambda=100,
    observation_lambda=16,
    threshold=0,
    correct=True,
):
    """Track beats in a beat activation function with a DBN.

    :param activations: 1-D array of beat probabilities, one per frame. They should be strictly
        between 0 and 1.
    :param fps: frames per second of the activations.
    :param min_bpm: minimum tempo (in beats per minute).
    :param max_bpm: maximum tempo (in beats per minute).
    :param num_tempi: number of tempi to model, if None all integer beat intervals are used.
    :param transition_lambda: higher values favour a constant tempo over a tempo change between beats.
    :param observation_lambda: split one beat period into N parts, the first representing beat states.
    :param threshold: threshold the activations before decoding.
    :param correct: align the beats to the maximum of the activations.

    :returns: 1-D array with the beat times (in seconds).
    """
    activations = np.asarray(activations, dtype=np.float64)
    activations, first = _trim_activations(activations, threshold)
    if not activations.any():
        return np.empty(0)

    state_space = BeatStateSpace(60.0 * fps / max_bpm, 60.0 * fps / min_bpm, num_tempi)
    pointers = np.zeros(state_space.num_states, dtype=np.int64)
    pointers[state_space.state_positions < 1.0 / observation_lambda] = 1
    hmm = HiddenMarkovModel(beat_transitions(state_space, transition_lambda), pointers)

    densities = np.empty((len(activations), 2))
    densities[:, 0] = np.log((1.0 - activations) / (observation_lambda - 1))
    densities[:, 1] = np.log(activations)
    path, _ = hmm.viterbi(densities)
    if len(path) == 0:
        return np.empty(0)

    if correct:
        beats = []
        beat_range = pointers[path]
        for left, right in _segments(beat_range.astype(bool)):
            beats.append(np.argmax(activations[left:right]) + left)
        beats = np.asarray(beats, dtype=np.int64)
    else:
        positions = state_space.state_positions[path]
        # local minima of the (circular) position, i.e. beginning of each beat
        beats = np.nonzero(
            (positions < np.roll(positions, 1)) & (positions <= np.roll(positions, -1))
        )[0]
        beats = beats[pointers[path[beats]] == 1]
    return (beats + first) / float(fps)


def dbn_downbeat_tracking(
    activations,
    fps,
    beats_per_bar,
    min_bpm=55.0,
    max_bpm=215.0,
    num_tempi=60,
    transition_lambda=100,
    observation_lambda=16,
    threshold=0.05,
    correct=True,
):
    """Jointly track beats and downbeats with a DBN.

    One DBN is decoded per candidate bar length, and the most likely one is kept.

    :param activations: 2-D array (frames x 2) with the probabilities of a non-downbeat beat and a
        downbeat. They should be strictly between 0 and 1, and add up to less than 1.
    :param fps: frames per second of the activations.
    :param beats_per_bar: list of candidate numbers of beats per bar.
    :param min_bpm: minimum tempo (in beats per minute).
    :param max_bpm: maximum tempo (in beats per minute).
    :param num_tempi: number of tempi to model, if None all integer beat intervals are used.
    :param transition_lambda: higher values favour a constant tempo over a tempo change between beats.
    :param observation_lambda: split one beat period into N parts, the first representing beat states.
    :param threshold: threshold the activations before decoding.
    :param correct: align the beats to the maximum of the activations.

    :returns: 2-D array with the beat times (in seconds) and their position in the bar (1 for downbeats).
    """
    activations = np.asarray(activations, dtype=np.float64)
    activations, first = _trim_activations(activations, threshold)
    if not activations.any():
        return np.empty((0, 2))

    densities = np.empty((len(activations), 3))
    densities[:, 0] = np.log(
        (1.0 - np.sum(activations, axis=1)) / (observation_lambda - 1)
    )
    densities[:, 1] = np.log(activations[:, 0])
    densities[:, 2] = np.log(activations[:, 1])

    results = []
    for num_beats in np.atleast_1d(beats_per_bar):
        state_space = BarStateSpace(
            num_beats, 60.0 * fps / max_bpm, 60.0 * fps / min_bpm, num_tempi
        )
        pointers = np.zeros(state_space.num_states, dtype=np.int64)
        border = 1.0 / observation_lambda
        pointers[state_space.state_positions % 1 < border] = 1
        pointers[state_space.state_positions < border] = 2
        hmm = HiddenMarkovModel(
            beat_transitions(state_space, transition_lambda), pointers
        )
        path, log_probability = hmm.viterbi(densities)
        results.append((path, log_probability, state_space, pointers))

    path, _, state_space, pointers = results[
        int(np.argmax([result[1] for result in results]))
    ]
    if len(path) == 0:
        return np.empty((0, 2))

    beat_numbers = state_space.state_positions[path].astype(int) + 1
    if correct:
        beats = []
        for left, right in _segments(pointers[path] >= 1):
            beats.append(np.argmax(activations[left:right]) // 2 + left)
        beats = np.asarray(beats, dtype=np.int64)
    else:
        beats = np.nonzero(np.diff(beat_numbers))[0] + 1
    return np.vstack(((beats + first) / float(fps), beat_numbers[beats])).T


def dbn_bar_tracking(
    beats,
    downbeat_activations,
    beats_per_bar=(3, 4),
    observation_weight=100,
    meter_change_prob=1e-7,
):
    """Track the position of given beats within the bar with a DBN.

    :param beats: 1-D array with the beat times (in seconds).
    :param downbeat_activations: 1-D array with the downbeat probability at each beat, strictly
        between 0 and 1.
    :param beats_per_bar: list of candidate numbers of beats per bar.
    :param observation_weight: weight of the downbeat activations.
    :param meter_change_prob: probability of a change of bar length at the end of a bar.

    :returns: 2-D array with the beat times (in seconds) and their position in the bar (1 for downbeats).
    """
    beats = np.asarray(beats, dtype=np.float64)
    beats_per_bar = list(np.atleast_1d(beats_per_bar))
    # the last beat has no next beat to be compared to
    activations = np.asarray(downbeat_activations, dtype=np.float64)[:-1]
    if len(activations) == 0:
        return np.empty((0, 2))

    # one state per beat in each of the bar lengths
    offsets = np.cumsum([0] + beats_per_bar)
    num_states = int(offsets[-1])
    to_states, from_states, probabilities = [], [], []
    num_patterns = len(beats_per_bar)
    for pattern, num_beats in enumerate(beats_per_bar):
        offset = offsets[pattern]
        to_states.extend(offset + np.arange(1, num_beats))
        from_states.extend(offset + np.arange(0, num_beats - 1))
        probabilities.extend([1.0] * (num_beats - 1))
        stay = 1.0 - meter_change_prob if num_patterns > 1 else 1.0
        to_states.append(offset)
        from_states.append(offsets[pattern + 1] - 1)
        probabilities.append(stay)
        if num_patterns > 1 and meter_change_prob:
            for other in range(num_patterns):
                if other != pattern:
                    to_states.append(offset)
                    from_states.append(offsets[other + 1] - 1)
                    probabilities.append(meter_change_prob / (num_patterns - 1))

    state_positions = np.hstack([np.arange(n) for n in beats_per_bar])
    state_patterns = np.repeat(np.arange(num_patterns), beats_per_bar)
    pointers = (state_positions < 1.0 / observation_weight).astype(np.int64)
    hmm = HiddenMarkovModel(
        _sparse_transitions(
            np.asarray(to_states),
            np.asarray(from_states),
            np.asarray(probabilities),
            num_states,
        ),
        pointers,
    )

    densities = np.empty((len(activations), 2))
    densities[:, 0] = np.log((1.0 - activations) / (observation_weight - 1))
    densities[:, 1] = np.log(activations)
    path, _ = hmm.viterbi(densities)
    if len(path) == 0:
        return np.empty((0, 2))

    beat_numbers = state_positions[path].astype(int) + 1
    meter = beats_per_bar[state_patterns[path[-1]]]
    beat_numbers = np.append(beat_numbers, np.mod(beat_numbers[-1], meter) + 1)
    return np.vstack((beats, beat_numbers)).T
