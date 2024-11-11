import os

import numpy as np

from configobj import ConfigObj

from compiam.utils import get_logger, WORKDIR
from compiam.utils.download import download_remote_model

logger = get_logger(__name__)


class CAEWrapper:
    """
    Wrapper for the Complex Autoencoder found at https://github.com/SonyCSLParis/cae-invar#quick-start
    specifically for the task of embedding audio to learnt CAE features.
    This wrapper is used for inference and it is not trainable. Please initialize it using
    compiam.load_model()
    """

    def __init__(
        self,
        model_path,
        conf_path,
        spec_path,
        download_link,
        download_checksum,
        device="cpu",
    ):
        """Initialise wrapper with trained model from original CAE implementation

        :param model_path: Path to .save model trained using original CAE implementation.
        :param conf_path: Path to .ini conf used to train model at <model_path>.
        :param spec_path: Path to .cfg configuration spec.
        :param download_link: link to the remote pre-trained model.
        :param download_checksum: checksum of the model file.
        :param map_location: cpu or gpu [optional, defaults to cpu].
        """
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global torch
            import torch

            global to_cqt_repr, standardize
            from compiam.melody.pattern.sancara_search.complex_auto.cqt import (
                to_cqt_repr,
                standardize,
            )

            global Complex
            from compiam.melody.pattern.sancara_search.complex_auto.complex import (
                Complex,
            )

            global cuda_variable
            from compiam.melody.pattern.sancara_search.complex_auto.util import (
                cuda_variable,
            )

        except:
            raise ImportError(
                "In order to use this tool you need to have torch installed. "
                "Install compIAM with torch support: pip install 'compiam[torch]'"
            )
        ###

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.conf_path = conf_path
        self.model_path = model_path
        self.download_link = download_link
        self.download_checksum = download_checksum

        self.trained = False
        # To prevent CUDNN_STATUS_NOT_INITIALIZED error in case of incompatible GPU
        try:
            self.load_model(model_path, conf_path, spec_path)

        except:
            self.device = "cpu"
            self.load_model(model_path, conf_path, spec_path)

    def load_conf(self, path, spec):
        """
        Load .ini conf at <path>

        :param path: path to .ini configuration file
        :type path: str
        :param spec: path to .cfg configuration spec file
        :type spec: str


        :returns: dict of parameters
        :rtype: dict
        """
        configspec = ConfigObj(
            spec, interpolation=True, list_values=False, _inspec=True
        )

        conf = ConfigObj(path, unrepr=True, configspec=configspec)

        return dict(conf)

    def validate_conf(self, conf):
        """
        Ensure all relevant parameters for feature extraction
        are present in <conf>

        :param path: dict of parameters
        :type path: dict

        :returns: True/False, are relevant parameters present
        :rtype: bool
        """
        for param in [
            "n_bins",
            "length_ngram",
            "n_bases",
            "dropout",
            "sr",
            "bins_per_oct",
            "fmin",
            "hop_length",
        ]:
            if param not in conf:
                raise ValueError(f"{param} not present in conf at <self.conf_path>")

        if not isinstance(conf["n_bins"], int):
            raise ValueError("n_bins in conf at <conf_path> should be an integer")

        if not isinstance(conf["length_ngram"], int):
            raise ValueError("length_ngram in conf at <conf_path> should be an integer")

        if not isinstance(conf["n_bases"], int):
            raise ValueError("n_bases in conf at <conf_path> should be an integer")

        if not isinstance(conf["dropout"], float):
            raise ValueError("dropout in conf at <conf_path> should be a float")

        if not isinstance(conf["sr"], int):
            raise ValueError("sr in conf at <conf_path> should be an integer")

        if not isinstance(conf["bins_per_oct"], int):
            raise ValueError("bins_per_oct in conf at <conf_path> should be an integer")

        if not isinstance(conf["fmin"], (float, int)):
            raise ValueError("fmin in conf at <conf_path> should be an float/integer")

        if not isinstance(conf["hop_length"], int):
            raise ValueError("hop_length in conf at <conf_path> should be an integer")

    def _build_model(self):
        """
        Build de CAE model.

        :returns: loaded model
        :rtype: torch.nn.Module
        """
        in_size = self.n_bins * self.length_ngram
        return Complex(in_size, self.n_bases, dropout=self.dropout).to(self.device)

    def load_model(self, model_path, conf_path, spec_path):
        """
        Load model at <model_path>. Expects model parameters to correspond
        to those found in self.params (loaded from self.conf_path).

        :param model_path: path to model
        :type model_path: str
        """
        if not os.path.exists(model_path):
            self.download_model(model_path)

        print(conf_path, spec_path)
        self.params = self.load_conf(conf_path, spec_path)
        self.validate_conf(self.params)

        for tp, v in self.params.items():
            # unpack parameters to class attributes
            setattr(self, tp, v)

        self.model = self._build_model()
        self.model.load_state_dict(
            torch.load(model_path, weights_only=True), strict=False
        )
        self.trained = True

    def download_model(self, model_path=None, force_overwrite=False):
        """Download pre-trained model."""
        download_path = (
            os.sep + os.path.join(*model_path.split(os.sep)[:-2])
            if model_path is not None
            else os.path.join(WORKDIR, "models", "melody", "caecarnatic")
        )
        # Creating model folder to store the weights
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        download_remote_model(
            self.download_link,
            self.download_checksum,
            download_path,
            force_overwrite=force_overwrite,
        )

    def extract_features(self, file_path, sr=None):
        """
        Extract CAE features using self.model on audio at <file_path>

        :param file_path: path to audio
        :type file_path: str
        :param sr: sampling rate of audio at <file_path>, if None, use self.sr
        :type sr: int

        :returns: amplitude vector, phases vector
        :rtype: np.ndarray, np.ndarray
        """
        sr = sr if sr else self.sr

        cqt = self.get_cqt(file_path, sr=None)
        ampls, phases = self.to_amp_phase(cqt)
        return ampls, phases

    def get_cqt(self, file_path, sr=None):
        """
        Extract CQT representation from audio at <file_path> according
        to parameters specified in conf at self.conf_path

        :param file_path: path to audio
        :type file_path: str
        :param sr: sampling rate of audio at <file_path>, if None, use self.sr
        :type sr: int

        :returns: cqt representation
        :rtype: np.ndarray
        """
        sr = sr if sr else self.sr

        repres = to_cqt_repr(
            file_path,
            self.n_bins,
            self.bins_per_oct,
            self.fmin,
            self.hop_length,
            use_nr_samples=-1,
            sr=sr,
            standard=True,
            mult=1.0,
        )

        return repres.transpose()

    def to_amp_phase(self, cqt):
        """
        Extract amplitude and phase vector from model
        on <cqt> representation

        :param cqt: CQT representation of audio
        :type cqt: np.ndarray

        :returns: amplitude vector, phases vector
        :rtype: np.ndarray, np.ndarray
        """
        self.model.to(self.device)
        self.model.eval()

        ngrams = []
        for i in range(0, len(cqt) - self.length_ngram, 1):
            curr_ngram = cqt[i : i + self.length_ngram].reshape((-1,))
            curr_ngram = standardize(curr_ngram)
            ngrams.append(curr_ngram)

        x = cuda_variable(torch.FloatTensor(np.vstack(ngrams)))

        ampl, phase = self.model(x)

        return ampl, phase
