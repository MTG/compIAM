import numpy as np

from configobj import ConfigObj


class CAEWrapper:
    """
    Wrapper for the Complex Autoencoder found at https://github.com/SonyCSLParis/cae-invar#quick-start
    specifically for the task of embedding audio to learnt CAE features.
    This wrapper is used for inference and it is not trainable. Please initialize it using
    compiam.load_model()

    Example:
    ```
    model_path = "<model_output_folder>/model_complex_auto_cqt.save"
    conf_path  = "cae-invar/config_cqt_old.ini"
    spec_path  = "cae-invar/config_spec.cfg"
    file_path = "<path_to_audio>"

    model = CAEWrapper(model_path, conf_path, spec_path)
    ampls, phase = model.extract_features(file_path)
    ampls
    >> tensor([[1.2935, 4.0693, 1.0390,  ..., 0.4740, 1.3497, 0.5319],
        [1.5923, 2.4673, 3.4847,  ..., 0.4998, 1.3553, 1.4519],
        [1.2482, 1.0292, 2.1280,  ..., 0.7126, 1.4086, 1.9351],
        ...,
        [7.7372, 5.1458, 1.2539,  ..., 1.0162, 0.3583, 0.9603],
        [7.4559, 4.9839, 1.7646,  ..., 1.2773, 0.2818, 1.1203],
        [7.3733, 5.0220, 1.8748,  ..., 1.1992, 0.4380, 1.2692]],
       grad_fn=<PowBackward0>)
    phase
    >> tensor([[-0.3304, -2.3667, -2.6688,  ...,  1.2986, -0.7942,  1.8279],
        [ 0.1712, -2.7483,  0.8825,  ...,  1.6427,  2.7336,  0.6940],
        [ 1.5836, -0.6167,  0.4733,  ...,  2.7312,  2.2416,  0.3589],
        ...,
        [-2.6896, -1.4801,  0.6764,  ...,  1.9469,  2.1927,  0.2756],
        [-2.5489, -1.5051,  0.8178,  ...,  1.9912,  2.0662,  0.1495],
        [-2.3444, -1.6104,  0.6585,  ...,  1.9241,  0.7816,  0.0332]],
       grad_fn=<Atan2Backward>)
    ```
    """
    def __init__(self, model_path, conf_path, spec_path, device="cpu"):
        """
        Initialise wrapper with trained model from original CAE implementation

        :param model_path: Path to .save model trained using original CAE implementation
        :type model_path: str
        :param conf_path: Path to .ini conf used to train model at <model_path>
        :type conf_path: str
        :param spec_path: Path to .cfg configuration spec 
        :type spec_path: str
        :param map_location: cpu or gpu [optional, defaults to cpu]
        :type map_location: str
        """
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global torch
            import torch

            global to_cqt_repr, standardize
            from compiam.melody.pattern.sancara_search.complex_auto.cqt import to_cqt_repr, standardize

            global Complex
            from compiam.melody.pattern.sancara_search.complex_auto.complex import Complex

            global cuda_variable
            from compiam.melody.pattern.sancara_search.complex_auto.util import cuda_variable

        except:
            raise ImportError(
                "In order to use this tool you need to have torch installed. "
                "Please install torch using: pip install torch"
            )
        ###

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.conf_path = conf_path
        self.model_path = model_path

        self.params = self.load_conf(conf_path, spec_path)
        self.validate_conf(self.params)
        
        for tp,v in self.params.items():
            # unpack parameters to class attributes
            setattr(self, tp, v)

        self.trained = False
        # To prevent CUDNN_STATUS_NOT_INITIALIZED error in case of incompatible GPU
        try:
            self.load_model(model_path)
        except:
            self.device = "cpu"
            self.load_model(model_path)
        
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
        configspec = ConfigObj(spec, interpolation=True,
                               list_values=False, _inspec=True)

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
            "n_bins","length_ngram","n_bases","dropout",
            "sr","bins_per_oct","fmin","hop_length"]:
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

        if not isinstance(conf["fmin"], (float,int)):
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

    def load_model(self, model_path):
        """
        Load model at <model_path>. Expects model parameters to correspond
        to those found in self.params (loaded from self.conf_path).

        :param model_path: path to model
        :type model_path: str
        """
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.trained = True

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

        repres = to_cqt_repr(file_path, self.n_bins, self.bins_per_oct, self.fmin,
                  self.hop_length, use_nr_samples=-1, sr=sr, standard=True, mult=1.)

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
            curr_ngram = cqt[i:i + self.length_ngram].reshape((-1,))
            curr_ngram = standardize(curr_ngram)
            ngrams.append(curr_ngram)

        x = cuda_variable(torch.FloatTensor(np.vstack(ngrams)))

        ampl, phase = self.model(x)

        return ampl, phase
