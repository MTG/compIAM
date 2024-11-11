import os

import numpy as np

import compiam
from compiam.melody.raga_recognition.deepsrgm.raga_mapping import create_mapping
from compiam.exceptions import (
    ModelNotTrainedError,
    DatasetNotLoadedError,
)
from compiam.utils import get_logger, WORKDIR
from compiam.utils.download import download_remote_model

logger = get_logger(__name__)


class DEEPSRGM(object):
    """DEEPSRGM model for raga classification. This DEEPSGRM implementation has been
    kindly provided by Shubham Lohiya and Swarada Bharadwaj.
    """

    def __init__(
        self,
        model_path=None,
        download_link=None,
        download_checksum=None,
        rnn="lstm",
        mapping_path=None,
        sample_rate=44100,
        device=None,
    ):
        """DEEPSRGM init method.

        :param model_path: path to file to the model weights
        :param download_link: link to the remote pre-trained model.
        :param download_checksum: checksum of the model file.
        :param rnn: type of rnn used "lstm" or "gru"
        :param mapping_path: path to raga to id JSON mapping
        :param sample_rate: sampling rate which the model is trained
        :param device: torch CUDA config to route model to GPU
        """
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global torch
            import torch

            global deepsrgmModel
            from compiam.melody.raga_recognition.deepsrgm.model import deepsrgmModel
        except:
            raise ImportError(
                "In order to use this tool you need to have torch installed. "
                "Install compIAM with torch support: pip install 'compiam[torch]'"
            )
        ###

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rnn = rnn
        # To prevent CUDNN_STATUS_NOT_INITIALIZED error in case of incompatible GPU
        try:
            self.model = self._build_model(rnn=self.rnn)
        except:
            self.device = "cpu"
            self.model = self._build_model(rnn=self.rnn)

        self.model_path = model_path
        self.download_link = download_link
        self.download_checksum = download_checksum
        self.sample_rate = sample_rate
        self.trained = False

        ## Loading LSTM model by default
        if self.model_path is not None:
            self.load_model(model_path=self.model_path[self.rnn], rnn=self.rnn)

        self.mapping_path = mapping_path
        self.selected_ragas = [
            5,
            8,
            10,
            13,
            17,
            20,
            22,
            23,
            24,
            28,
        ]  # pre-defined for release 0.1.0

        if (mapping_path is not None) and (self.selected_ragas is not None):
            self.load_mapping(self.selected_ragas)
        self.dataset = None

    def _build_model(self, rnn="lstm"):
        """Building DEEPSRM

        :param rnn: lstm (default) or gru.
        """
        return deepsrgmModel(rnn=rnn).to(self.device)

    def load_mapping(self, selection=None):
        """Loading raga mapping for DEEPSRGM

        :param selection: Selection of ragas for the DEEPSRGM model. A default selection
            is initialized by default in compiam v1.0. Flexible selection and training of this
            model is under development at this moment and will be available in the next release.
        """
        selected_ragas = self.selected_ragas if selection is None else selection
        self.mapping = create_mapping(self.mapping_path, selected_ragas)

    def load_model(self, model_path, rnn="lstm"):
        """Loading weights for DEEPSRGM

        :param model_path: path to model.
        :param rnn: lstm (default) or gru.
        """
        if not os.path.exists(model_path):
            self.download_model(model_path)

        if rnn == "gru":
            self.model = self._build_model(rnn="gru")

        self.model_path = model_path
        weights = torch.load(model_path, weights_only=True, map_location=self.device)
        new_weights = weights.copy()
        keys_to_fix = [
            ".weight_ih_l0",
            ".weight_hh_l0",
            ".bias_ih_l0",
            ".bias_hh_l0",
        ]
        keys_to_fix = [rnn + x for x in keys_to_fix]
        for i in keys_to_fix:
            new_weights[i.replace(rnn, "rnn")] = weights[i]
            del new_weights[i]
        self.model.load_state_dict(new_weights)
        self.trained = True

    def download_model(self, model_path=None, force_overwrite=False):
        """Download pre-trained model."""
        download_path = (
            os.sep + os.path.join(*model_path.split(os.sep)[:-2])
            if model_path is not None
            else os.path.join(WORKDIR, "models", "melody", "deepsrgm")
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

    def load_raga_dataset(self, data_home=None, download=False):
        """Load an instance of the Compmusic raga dataset to assist the tool

        :param data_home: path where to store the dataset data
        :param download:
        """
        self.dataset = compiam.load_dataset(
            "compmusic_raga", data_home=data_home, version="default"
        )
        if download:
            self.dataset.download()  # Downloads index and features
            logger.warning(
                f"""
                The features are downloaded, but the audio of this dataset is private. 
                Please request it in this link: https://zenodo.org/records/7278511, 
                download it, and unzip it in {data_home} following the instructions.
            """
            )

    def get_features(
        self,
        input_data=None,
        input_sr=44100,
        pitch_path=None,
        tonic_path=None,
        from_mirdata=False,
        track_id=None,
        k=5,
    ):
        """Computing features for prediction of DEEPSRM

        :param input_data: path to audio file or numpy array like audio signal.
        :param input_sr: sampling rate of the input array of data (if any). This variable is only
            relevant if the input is an array of data instead of a filepath.
        :param pitch_path: path to pre-computed pitch file (if available).
        :param tonic_path: path to pre-computed tonic file (if available).
        :param from_mirdata: boolean to indicate if the features are parsed from the mirdata loader of
            Indian Art Music Raga Recognition Dataset (must be specifically this one).
        :param track_id: track id for the Indian Art Music Raga Recognition Dataset if from_mirdata is
            set to True.
        :param k: k indicating the precision of the pitch feature.
        """
        if (pitch_path is not None) and (tonic_path is not None):
            freqs = open(pitch_path).read().strip().split("\n")
            tonic = eval(open(tonic_path).read().strip())

        elif from_mirdata:
            if self.dataset is None:
                raise DatasetNotLoadedError(
                    "Dataloader is not initialized. Have you run .load_raga_dataset()?"
                )
            if track_id is None:
                raise ValueError(
                    "To load a track we need a track id. See mirdata instructions \
                    to know how to list the available ids."
                )
            track = self.dataset.track(track_id)
            pitch_path = track.pitch_post_processed_path
            tonic_path = track.tonic_fine_tuned_path
            freqs = open(pitch_path).read().strip().split("\n")
            tonic = eval(open(tonic_path).read().strip())

        else:
            try:
                import essentia.standard as estd

                melodia = compiam.melody.pitch_extraction.Melodia
                melodia = melodia(sample_rate=self.sample_rate)
                tonic_extraction = (
                    compiam.melody.tonic_identification.TonicIndianMultiPitch
                )
                tonic_extraction = tonic_extraction(sample_rate=self.sample_rate)
            except:
                raise ImportError(
                    "In order to use these tools to extract the features you need to have essentia installed."
                    "Install compIAM with essentia support: pip install 'compiam[essentia]'"
                )

            if isinstance(input_data, str):
                if not os.path.exists(input_data):
                    raise FileNotFoundError("Target audio not found.")
                audio = estd.MonoLoader(
                    filename=input_data, sampleRate=self.sample_rate
                )()
            elif isinstance(input_data, np.ndarray):
                logger.warning(
                    "Resampling... (input sampling rate is {input_sr}Hz, make sure this is correct)"
                )
                resampling = estd.Resample(
                    inputSampleRate=input_sr, outputSampleRate=self.sample_rate
                )
                audio = resampling(input_data)
            else:
                raise ValueError("Input must be path to audio signal or an audio array")

            logger.info("Extracting pitch track using melodia...")
            freqs = melodia.extract(audio)[:, 1]

            logger.info("Extracting tonic using multi-pitch approach...")
            tonic = tonic_extraction.extract(audio)

        # Normalise pitch
        feature = np.round(1200 * np.log2(freqs / tonic) * (k / 100)).clip(0)
        N = 200
        a = []
        if len(feature) <= 5000:
            raise ValueError(
                """
                Audio signal is not longer enough for a proper estimation. Please provide a larger audio.
            """
            )
        for i in range(N):
            c = np.random.randint(0, len(feature) - 5000)
            a.append(feature[c : c + 5000])
        return np.array(a)

    def predict(self, features, threshold=0.6, gpu="-1"):
        """Predict raga for recording

        :param features: all subsequences for a certain music recording
        :param threshold: majority voting threshold
        :param gpu: Id of the available GPU to use (-1 by default, to run on CPU)
        :return: recognition result
        """
        ## Setting up GPU if any
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        if isinstance(features, str):
            raise ValueError(
                "Please first extract features using .get_features() and use \
                these as input for this predict function."
            )

        # Make sure model is loaded
        if self.trained is False:
            raise ModelNotTrainedError(
                """
                Model is not trained. Please load model before running inference!
                You can load the pre-trained instance with the load_model wrapper.
            """
            )

        # Make sure mapping is loaded
        if self.mapping is None:
            self.load_mapping(self.selected_ragas)
        list_of_ragas = list(self.mapping.values())

        # Predict
        logger.info(
            "Performing prediction for the following {} ragas: {}".format(
                len(list_of_ragas), list_of_ragas
            )
        )
        with torch.no_grad():
            out = self.model.forward(torch.from_numpy(features).to(self.device).long())
        preds = torch.argmax(out, axis=-1)
        majority, _ = torch.mode(preds)
        majority = int(majority)
        votes = float(torch.sum(preds == majority)) / features.shape[0]
        if votes >= threshold:
            logger.info(
                "Input music sample belongs to the {} raga".format(
                    self.mapping[majority]
                )
            )
        logger.info(
            "CONFUSED - Closest raga predicted is {} with {} votes".format(
                self.mapping[majority], (votes * 100)
            )
        )
        return self.mapping[majority]
