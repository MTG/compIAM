import os
import mirdata

import numpy as np

import compiam
from compiam.melody.deepsrgm.raga_mapping import create_mapping


class DEEPSRGM(object):
    """DEEPSRGM model for raga classification. This DEEPSGRM implementation has been
    kindly provided by Shubham Lohiya and Swarada Bharadwaj.
    """

    def __init__(self, model_path, mapping_path, dataset_home=None, device=None):
        """DEEPSRGM init method.

        :param model_path: path to file to the model weights.
        :param mapping_path: path to raga to id JSON mapping
        :param dataset_home: path to find the mirdata dataset
        :param device: torch CUDA config to route model to GPU
        """
        ###
        try:
            global torch
            import torch

            global deepsrgmModel
            from compiam.melody.deepsrgm.model import deepsrgmModel
        except:
            raise ImportError(
                "In order to use this tool you need to have torch installed. "
                "Please reinstall compiam using `pip install 'compiam[torch]'"
            )
        ###

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_path = model_path

        ## Loading LSTM model by default
        if self.model_path is not None:
            self.model = deepsrgmModel(rnn="lstm").to(self.device)
            self.load_model(rnn="lstm")

        self.mapping_path = mapping_path
        self.mapping = None
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
        ]  # pre-defined for release 0.1
        # self.dataset = mirdata.initialize("compmusic_raga_dataset", data_home=dataset_home)
        self.dataset = (
            None  # To update when CompMusic Raga dataset is integrated mirdata
        )

    def load_mapping(self, selection=None):
        """TODO"""
        selected_ragas = self.selected_ragas if selection is None else selection
        self.mapping = create_mapping(self.mapping_path, selected_ragas)

    def load_model(self, rnn="lstm"):
        """TODO"""
        if rnn == "gru":
            self.model = deepsrgmModel(rnn="gru").to(self.device)
            weights_path = os.path.join(self.model_path, "gru_30_checkpoint.pth")
        else:
            weights_path = os.path.join(self.model_path, "lstm_25_checkpoint.pth")
        if not os.path.exists(weights_path):
            raise ValueError(
                """
                Given path to model weights not found. Make sure you enter the path correctly.
                A training process for DEEPSRGM is under development right now and will be added 
                to the library soon. Meanwhile, we provide the weights in the latest repository 
                version (https://github.com/MTG/compIAM) so make sure you have these available before 
                loading the DEEPSRGM.
            """
            )

        else:
            weights = torch.load(weights_path, map_location=self.device)
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

    def get_features(
        self,
        audio_file=None,
        pitch_file=None,
        tonic_file=None,
        from_mirdata=False,
        track_id=None,
        k=5,
    ):
        """TODO"""
        if (pitch_file is not None) and (tonic_file is not None):
            freqs = open(pitch_file).read().strip().split("\n")
            tonic = eval(open(tonic_file).read().strip())

        elif from_mirdata:
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
                melodia = compiam.load_model("melody:melodia")
                tonic_extraction = compiam.load_model("melody:tonic-multipitch")
            except:
                raise ImportError(
                    "In order to use this tool in this context you need to have essentia "
                    " and torch installed. "
                    "Please reinstall compiam using `pip install 'compiam[essentia-torch]'`"
                )
            if not os.path.exists(audio_file):
                raise FileNotFoundError("Input audio not found.")
            freqs = melodia.extract(audio_file)[:, 1]

            if not os.path.exists(audio_file):
                raise FileNotFoundError("Input audio not found.")
            tonic = tonic_extraction.extract(audio_file)

        # Normalise pitch
        feature = np.round(1200 * np.log2(freqs / tonic) * (k / 100)).clip(0)
        N = 200
        a = []
        for i in range(N):
            c = np.random.randint(0, len(feature) - 5000)
            a.append(feature[c : c + 5000])
        return np.array(a)

    def predict(self, features, threshold=0.6):
        """Predict raga for recording

        :param features: all subsequences for a certain music recording
        :param threshold: majority voting threshold
        :return: recognition result
        """
        if isinstance(features, str):
            raise ValueError(
                "Please first extract features using .get_features() and use \
                these as input for this predict function."
            )

        # Make sure mapping is loaded
        if self.mapping is None:
            self.load_mapping(self.selected_ragas)

        # Make sure model is loaded
        self.load_model()
        self.model.eval()

        # Predict
        with torch.no_grad():
            out = self.model.forward(torch.from_numpy(features).to(self.device).long())
        preds = torch.argmax(out, axis=-1)
        majority, _ = torch.mode(preds)
        majority = int(majority)
        votes = float(torch.sum(preds == majority)) / features.shape[0]
        if votes >= threshold:
            return f"Input music sample belongs to the {self.mapping[majority]} raga"
        return f"CONFUSED - Closest raga predicted is {self.mapping[majority]} with {(votes*100):.2f}% votes"
