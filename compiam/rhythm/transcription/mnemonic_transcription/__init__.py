import csv
import pickle

import hmmlearn.hmm as hmm
import librosa
import numpy as np
from sklearn.exceptions import NotFittedError
import tqdm

from compiam.exceptions import ModelNotTrainedError
from compiam.rhythm.meter.akshara_pulse_tracker import AksharaPulseTracker
from compiam.utils import get_logger

logger = get_logger(__name__)


class MnemonicTranscription:
    """
    bōl or solkattu transcription from audio. Based on model presented in [1]

    [1] Gupta, S., Srinivasamurthy, A., Kumar, M., Murthy, H., & Serra, X.
    (2015, October). Discovery of Syllabic Percussion Patterns in Tabla
    Solo Recordings. In Proceedings of the 16th International Society
    for Music Information Retrieval Conference (ISMIR 2015) (pp. 385--391).
    Malaga, Spain.
    """

    def __init__(
        self,
        syllables,
        feature_kwargs={"n_mfcc": 13, "win_length": 1024, "hop_length": 256},
        model_kwargs={
            "n_components": 7,
            "n_mix": 3,
            "n_iter": 100,
            "algorithm": "viterbi",
            "params": "mcw",
        },
        sr=44100,
    ):
        """
        :param syllables: List of strings representing expected bōl/solkattu syllables OR dict of string:string mappings.
            If a dict is passed, any subsequent syllable labels passed at training time will be converted as per this mapping. The values of
            this dict will act as the bōl/solkattu "vocabulary" for this model.
        :type syllables: list or dict
        :param feature_kwargs: Dict of parameters to pass to librosa.feature.mfcc or librosa.feature.delta. Defaults are selected as per [1]
        :type feature_kwargs: dict
        :param model_kwargs: Dict of parameters to pass to hmmlearn.hmm.GMMHMM. Defaults are selected as per [1]
        :type model_kwargs: dict
        :param sr: sampling rate of audio to train on (default 44.1Hz)
        :type sr: int

        [1] Gupta, S., Srinivasamurthy, A., Kumar, M., Murthy, H., & Serra, X.
        (2015, October). Discovery of Syllabic Percussion Patterns in Tabla
        Solo Recordings. In Proceedings of the 16th International Society
        for Music Information Retrieval Conference (ISMIR 2015) (pp. 385-391).
        Malaga, Spain.
        """
        self.sr = sr

        if isinstance(syllables, dict):
            self.syllables = set(syllables.values())
            self.mapping = syllables
        else:
            self.syllables = set(syllables)
            self.mapping = {x: x for x in syllables}

        self.models = {}
        for s in self.syllables:
            self.models[s] = hmm.GMMHMM(**model_kwargs)

        self.feature_kwargs = feature_kwargs
        self.trained = False

    def train(self, file_paths_audio, file_paths_annotation, sr=None):
        """
        Train one gaussian mixture model hidden markov model for each syllables passed at initialisation
        on input audios and annotations passed via <file_paths_audio> and <file_paths_annotation>.
        Training hyperparameters are configured upon intialisation and can be accessed/changed
        via self.model_kwargs.

        :param file_paths_audio: List of file_paths to audios to train on
        :type file_paths_audio: list
        :param file_paths_annotation: List of file_paths to annotations to train on.
            annotations should be in csv format, with no header of (timestamp in seconds, <syllable>).
            Annotated syllables that do not correspond to syllables passed at initialisation will be ignored
            One annotations path should be passed for each audio path
        :type file_paths_annotation: list
        :param sr: sampling rate of audio to train on (default <self.sr>)
        :param sr: int
        """
        if not len(file_paths_audio) == len(file_paths_annotation):
            raise Exception(
                "file_paths_audio and file_paths_annotation must be the same length"
            )

        sr = self.sr if not sr else sr

        samples = {s: [] for s in self.syllables}
        # load and extract features
        for fau, fan in zip(file_paths_audio, file_paths_annotation):
            audio, _ = librosa.load(fau, sr=sr)
            annotations = self.load_annotations(fan)

            annotations = [(round(t * sr), self.map(a)) for t, a in annotations]
            ann_syls = set([x[1] for x in annotations])

            # Extract melodic features
            features = self.extract_features(audio, sr=sr)

            for syl in ann_syls:
                if not syl in self.syllables:
                    continue

                # get params
                hop_length = self.feature_kwargs["hop_length"]

                # get training samples relevant to syl
                samples_ix = self.get_sample_ix(annotations, audio, syl)
                samples_ix = [
                    (int(i / hop_length), int(j / hop_length)) for i, j in samples_ix
                ]

                samples[syl] = samples[syl] + [features[:, i:j] for i, j in samples_ix]

        # Train models
        for syl, samps in tqdm.tqdm(samples.items()):
            if samps:
                samps_concat = np.concatenate(samps, axis=1)
                lengths = np.array([s.shape[1] for s in samps])
                self.models[syl].fit(samps_concat.T, lengths)

        self.trained = True

    def predict(self, file_paths, onsets=None, sr=None):
        """
        Predict bōl/solkattu transcription for list of input audios at <file_paths>.

        :param file_paths: Either one file_path or list of file_paths to audios to predict on
        :type file_paths: list or string
        :param onsets: list representing onsets in audios. If None, compiam.rhythm.akshara_pulse_tracker is used
            to automatically identify bōl/solkattu onsets. If passed should be a list of onset annotations, each being
            a list of bōl/solkattu onsets in seconds. <onsets> should contain one set of onset annotations for each file_path
            in <file_paths>
        :type onsets: list or None
        :param sr: sampling rate of audio to train on (default <self.sr>)
        :param sr: int

        :returns: if <file_paths> is a list, then return a list of transcriptions, each
            transcription of the form [(timestamp in seconds, bōl/solkattu),...]. Or if <file_paths>
            is a single fiel path string, return a single transcription.
        :rtype: list
        """
        if not self.trained:
            raise ModelNotTrainedError(
                "Please train model before predicting using .train() method"
            )
        if onsets:
            if not len(onsets) == len(file_paths):
                raise Exception("One onset annotation required for each file path")

        sr = self.sr if not sr else sr
        if not isinstance(file_paths, list):
            file_paths = list(file_paths)
        results = []
        for i, fau in enumerate(file_paths):
            ot = onsets[i] if onsets else None
            results.append(self.predict_single(fau, onsets=ot, sr=sr))
        return results[0] if len(results) == 1 else results

    def predict_single(self, file_path, onsets=None, sr=None):
        """
        Predict bōl/solkattu transcription directly from audio time series
        (such as for example that loaded by librosa.load)

        :param file_path: File path to audio to analyze
        :type file_path: str
        :param onsets: If None, compiam.rhythm.akshara_pulse_tracker is used to automatically
            identify bōl/solkattu onsets. If passed <onsets> should be a list of bōl/solkattu
            onsets in seconds
        :type onsets: list or None
        :param sr: sampling rate of audio to train on (default <self.sr>)
        :param sr: int

        :returns: bōl/solkattu transcription of form [(time in seconds, syllable),... ]
        :rtype: list
        """
        sr = self.sr if not sr else sr
        hop_length = self.feature_kwargs["hop_length"]

        audio, _ = librosa.load(file_path, sr=sr)
        features = self.extract_features(audio, sr=sr)

        if not onsets:
            # if not onsets are passed, extract using
            pulse = AksharaPulseTracker()
            onsets = pulse.extract(file_path)["aksharaPulses"]
            onsets = np.append(onsets, len(audio) / sr)

        n_ons = len(onsets)
        samples_ix = [
            (
                int(onsets[i] * sr / hop_length),
                (int(onsets[i + 1] * sr / hop_length)) - 1,
            )
            for i in range(n_ons - 1)
        ]
        samples_ix.append((samples_ix[-1][1], features.shape[1] - 1))

        labels = []
        for i, j in samples_ix:
            samp = features[:, i:j]
            t = round(i * self.feature_kwargs["hop_length"] / sr, 2)
            label = self.predict_sample(samp)
            labels.append((t, label))

        return labels

    def predict_sample(self, sample):
        """
        Predict one sample using internal models. One sample should correspond to one
        bōl/solkattu

        :param sample: Numpy array features corresponding to <sample> (extracted using self.extract_features)
        :type sample: np.array

        :returns: bōl/solkattu label
        :rtype: str
        """
        names = []

        scores = []
        for syl in self.models.keys():
            try:
                scores.append(self.models[syl].score(sample.T))
                names.append(syl)
            except NotFittedError:
                logger.warning(
                    f"{syl} not fitted (no instance of {syl} in training) data. Will not be used for prediction."
                )
        label = scores.index(max(scores))
        return names[label]

    def map(self, a):
        """
        Map input bōl/solkattu, <a> to reduced bōl/solkattu vocabulary

        :param a: bōl/solkattu string (that must exist in self.mapping)
        :type a: np.array

        :returns: mapped bōl/solkattu label
        :rtype: str
        """
        if a in self.mapping:
            return self.mapping[a]
        else:
            logger.warning(
                f"Skipping unknown syllable, {a} for training instance. Model syllables: {self.mapping.keys()}"
            )
            return "!UKNOWN"

    def get_sample_ix(self, annotations, audio, syl):
        """
        Convert input onset annotations to list of in/out points for a
        specific bōl/solkattu syllable, <syl>

        :param annotations: onset annotations of the form [(timestamp in seconds, bōl/solkattu),... ]
        :type annotations: list/iterable
        :param audio: time series representation of audio
        :type audio: np.array
        :param syl: bōl/solkattu syllable to extract
        :type syl: str

        :returns: list or [(t1,t2),..] where t1 and t2 correspdong to in and out points of single
            bōls/solkattus
        :rtype: str
        """
        annotations_two = annotations + [(len(audio), "!END")]
        zipped = zip(annotations, annotations_two[1:])
        samples = [(t1, t2 - 1) for ((t1, a1), (t2, a2)) in zipped if a1 == syl]
        return samples

    def extract_features(self, audio, sr=None):
        """
        Convert input audio to features MFCC features

        :param audio: time series representation of audio
        :type audio: np.array
        :param sr: sampling rate of audio to train on (default <self.sr>)
        :param sr: int

        :returns: array of features
        :rtype: np.array
        """
        sr = self.sr if not sr else sr
        MFCC = librosa.feature.mfcc(y=audio, sr=sr, **self.feature_kwargs)
        MFCC_delta = librosa.feature.delta(MFCC)
        features = np.concatenate((MFCC, MFCC_delta))
        return features

    def load_annotations(self, annotation_path):
        """
        Load onset annotations from <annotation_path>

        :param annotation_path: path to onset annotations for one recording
            of the form (timestamp in seconds, bōl/solkattu syllable)
        :type annotation_path: str

        :returns: list of onset annotations (timestamp seconds, bōl/solkattu syllable)
        :rtype: list
        """
        annotations = []
        with open(annotation_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                annotations.append((float(row[0]), str(row[1]).strip()))

        return annotations

    def save(self, model_path):
        """
        Save model at path as .pkl

        :param model_path: Path to save model to
        :type model_path: strs
        """
        with open(model_path, "wb") as d:
            pickle.dump(self, d)
