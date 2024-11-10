import os
import glob
import librosa

import numpy as np
import matplotlib.pyplot as plt
from compiam.exceptions import ModelNotTrainedError

from compiam.data import WORKDIR
from compiam.utils import get_logger
from compiam.utils.download import download_remote_model

logger = get_logger(__name__)


class DhrupadBandishSegmentation:
    """Dhrupad Bandish Segmentation"""

    def __init__(
        self,
        mode="net",
        fold=0,
        model_path=None,
        splits_path=None,
        annotations_path=None,
        features_path=None,
        original_audios_path=None,
        processed_audios_path=None,
        download_link=None,
        download_checksum=None,
        device=None,
    ):
        """Dhrupad Bandish Segmentation init method.

        :param mode: net, voc, or pakh. That indicates the source for s.t.m. estimation. Use the net
            mode if audio is a mixture signal, else use voc or pakh for clean/source-separated vocals or
            pakhawaj tracks.
        :param fold: 0, 1 or 2, it is the validation fold to use during training.
        :param model_path: path to file to the model weights.
        :param splits_path: path to file to audio splits.
        :param annotations_path: path to file to the annotations.
        :param features_path: path to file to the computed features.
        :param original_audios_path: path to file to the original audios from the dataset (see README.md in
            compIAM/models/structure/dhrupad_bandish_segmentation/audio_original)
        :param processed_audios_path: path to file to the processed audio files.
        :param download_link: link to the remote pre-trained model.
        :param download_checksum: checksum of the model file.
        :param device: indicate whether the model will run on the GPU.
        """
        ### IMPORTING OPTIONAL DEPENDENCIES
        try:
            global torch
            import torch

            global split_audios
            from compiam.structure.segmentation.dhrupad_bandish_segmentation.audio_processing import (
                split_audios,
            )

            global extract_features, makechunks
            from compiam.structure.segmentation.dhrupad_bandish_segmentation.feature_extraction import (
                extract_features,
                makechunks,
            )

            global class_to_categorical, categorical_to_class, build_model, smooth_boundaries
            from compiam.structure.segmentation.dhrupad_bandish_segmentation.model_utils import (
                class_to_categorical,
                categorical_to_class,
                build_model,
                smooth_boundaries,
            )

            global pars
            import compiam.structure.segmentation.dhrupad_bandish_segmentation.params as pars
        except:
            raise ImportError(
                "In order to use this tool you need to have torch installed. "
                "Install compIAM with torch support: pip install 'compiam[torch]'"
            )
        ###

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load mode by default: update with self.update_mode()
        self.mode = mode
        # Load fold by default: update with self.update_fold()
        self.fold = fold
        self.classes = pars.classes_dict[self.mode]

        # To prevent CUDNN_STATUS_NOT_INITIALIZED error in case of incompatible GPU
        try:
            self.model = self._build_model()
        except:
            self.device = "cpu"
            self.model = self._build_model()
        self.model_path = model_path
        self.download_link = download_link
        self.download_checksum = download_checksum
        self.loaded_model_path = None
        self.trained = False

        if self.model_path is not None:
            path_to_model = os.path.join(
                self.model_path[self.mode], "saved_model_fold_" + str(self.fold) + ".pt"
            )
            self.load_model(path_to_model)  # Loading pre-trained model for given mode

        self.splits_path = (
            splits_path
            if splits_path is not None
            else os.path.join(
                WORKDIR, "models", "structure", "dhrupad_bandish_segmentation", "splits"
            )
        )
        self.annotations_path = (
            annotations_path
            if annotations_path is not None
            else os.path.join(
                WORKDIR,
                "models",
                "structure",
                "dhrupad_bandish_segmentation",
                "annotations",
            )
        )
        self.features_path = (
            features_path
            if features_path is not None
            else os.path.join(
                WORKDIR,
                "models",
                "structure",
                "dhrupad_bandish_segmentation",
                "features",
            )
        )
        self.original_audios_path = (
            original_audios_path
            if original_audios_path is not None
            else os.path.join(
                WORKDIR,
                "models",
                "structure",
                "dhrupad_bandish_segmentation",
                "audio_original",
            )
        )
        self.processed_audios_path = (
            processed_audios_path
            if processed_audios_path is not None
            else os.path.join(
                WORKDIR,
                "models",
                "structure",
                "dhrupad_bandish_segmentation",
                "audio_sections",
            )
        )

    def _build_model(self):
        """Building non-trained model"""
        return (
            build_model(pars.input_height, pars.input_len, len(self.classes))
            .float()
            .to(self.device)
        )

    def load_model(self, model_path):
        """Loading weights for model, given self.mode and self.fold

        :param model_path: path to model weights
        """
        if not os.path.exists(model_path):
            self.download_model(model_path)

        self.model = self._build_model()
        self.model.load_state_dict(
            torch.load(model_path, weights_only=True, map_location=self.device)
        )
        self.model.eval()
        self.loaded_model_path = model_path
        self.trained = True

    def download_model(self, model_path=None, force_overwrite=False):
        """Download pre-trained model."""
        print("modelpathhh", model_path)
        download_path = (
            os.sep + os.path.join(*model_path.split(os.sep)[:-4])
            if model_path is not None
            else os.path.join(
                WORKDIR, "models", "structure", "dhrupad_bandish_segmentation"
            )
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

    def update_mode(self, mode):
        """Update mode for the training and sampling. Mode is one of net, voc,
        pakh, indicating the source for s.t.m. estimation. Use the net mode if
        audio is a mixture signal, else use voc or pakh for clean/source-separated
        vocals or pakhawaj tracks.

        :param mode: new mode to use
        """
        self.mode = mode
        self.classes = pars.classes_dict[mode]
        path_to_model = os.path.join(
            self.model_path[self.mode], "saved_model_fold_" + str(self.fold) + ".pt"
        )
        self.load_model(path_to_model)

    def update_fold(self, fold):
        """Update data fold for the training and sampling

        :param fold: new fold to use
        """
        self.fold = fold
        path_to_model = os.path.join(
            self.model_path[self.mode], "saved_model_fold_" + str(self.fold) + ".pt"
        )
        self.load_model(path_to_model)

    def train(self, verbose=True):
        """Train the Dhrupad Bandish Segmentation model

        :param verbose: showing details of the model
        """
        logger.info("Splitting audios...")
        split_audios(
            save_dir=self.processed_audios_path,
            annotations_path=self.annotations_path,
            audios_path=self.original_audios_path,
        )
        logger.info("Extracting features...")
        extract_features(
            self.processed_audios_path,
            self.annotations_path,
            self.features_path,
            self.mode,
        )

        # generate cross-validation folds for training
        songlist = os.listdir(self.features_path)
        labels_stm = np.load(
            os.path.join(self.features_path, "labels_stm.npy"), allow_pickle=True
        ).item()

        partition = {"train": [], "validation": []}
        n_folds = 3
        all_folds = []
        for i_fold in range(n_folds):
            all_folds.append(
                np.loadtxt(
                    os.path.join(
                        self.splits_path, self.mode, "fold_" + i_fold + ".csv"
                    ),
                    delimiter=",",
                    dtype=str,
                )
            )

        val_fold = all_folds[self.fold]
        train_fold = np.array([])
        for i_fold in np.delete(np.arange(0, n_folds), self.fold):
            if len(train_fold) == 0:
                train_fold = all_folds[i_fold]
            else:
                train_fold = np.append(train_fold, all_folds[i_fold])

        for song in songlist:
            try:
                ids = glob.glob(os.path.join(self.features_path + song, "*.pt"))
            except:
                continue
            section_name = "_".join(song.split("_")[0:4])

            if section_name in val_fold:
                partition["validation"].extend(ids)
            elif section_name in train_fold:
                partition["train"].extend(ids)

        # generators
        training_set = torch.utils.data.Dataset(
            self.features_path, partition["train"], labels_stm
        )
        training_generator = torch.utils.data.data.DataLoader(training_set, **pars)

        validation_set = torch.utils.data.Dataset(
            self.features_path, partition["validation"], labels_stm
        )
        validation_generator = torch.utils.data.DataLoader(validation_set, **pars)

        # model definition and training
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        if verbose:
            logger.info(self.model)
            n_params = 0
            for param in self.model.parameters():
                n_params += torch.prod(torch.tensor(param.shape))
            logger.info("Num of trainable params: %d\n" % n_params)

        ##training epochs loop
        train_loss_epoch = []
        train_acc_epoch = []
        val_loss_epoch = []
        val_acc_epoch = []
        n_idle = 0

        if not os.path.exists(os.path.join(self.model_path, self.mode)):
            os.mkdir(os.path.join(self.model_path, self.mode))

        for epoch in range(pars.max_epochs):
            if n_idle == 50:
                break
            train_loss_epoch += [0]
            train_acc_epoch += [0]
            val_loss_epoch += [0]
            val_acc_epoch += [0]

            n_iter = 0
            ##training
            self.model.train()
            for local_batch, local_labels, _ in training_generator:
                # map labels to class ids
                local_labels = class_to_categorical(local_labels, self.classes)

                # add channel dimension
                if len(local_batch.shape) == 3:
                    local_batch = local_batch.unsqueeze(1)

                # transfer to GPU
                local_batch, local_labels = local_batch.float().to(
                    self.device
                ), local_labels.to(self.device)

                # update weights
                optimizer.zero_grad()
                outs = self.model(local_batch).squeeze()
                loss = criterion(outs, local_labels.long())
                loss.backward()
                optimizer.step()

                # append loss and acc to arrays
                train_loss_epoch[-1] += loss.item()
                acc = (
                    np.sum(
                        (
                            np.argmax(outs.detach().cpu().numpy(), 1)
                            == local_labels.detach().cpu().numpy()
                        )
                    )
                    / pars.batch_size
                )
                train_acc_epoch[-1] += acc
                n_iter += 1

            train_loss_epoch[-1] /= n_iter
            train_acc_epoch[-1] /= n_iter

            n_iter = 0
            ##validation
            self.model.eval()
            with torch.set_grad_enabled(False):
                for local_batch, local_labels, _ in validation_generator:
                    # map labels to class ids
                    local_labels = pars.class_to_categorical(local_labels, self.classes)

                    # add channel dimension
                    if len(local_batch.shape) == 3:
                        local_batch = local_batch.unsqueeze(1)

                    # transfer to GPU
                    local_batch, local_labels = local_batch.float().to(
                        self.device
                    ), local_labels.to(self.device)

                    # evaluate model
                    outs = self.model(local_batch).squeeze()
                    loss = criterion(outs, local_labels.long())

                    # append loss and acc to arrays
                    val_loss_epoch[-1] += loss.item()
                    acc = (
                        np.sum(
                            (
                                np.argmax(outs.detach().cpu().numpy(), 1)
                                == local_labels.detach().cpu().numpy()
                            )
                        )
                        / pars.batch_size
                    )
                    val_acc_epoch[-1] += acc

                    n_iter += 1
                val_loss_epoch[-1] /= n_iter
                val_acc_epoch[-1] /= n_iter

            # save if val_loss reduced
            if val_loss_epoch[-1] == min(val_loss_epoch):
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.model_path, self.mode, "saved_model_fold_%d.pt" % self.fold
                    ),
                )
                n_idle = 0
            else:
                n_idle += 1

            # log loss in current epoch
            logger.info(
                "Epoch no: %d/%d\tTrain loss: %f\tTrain acc: %f\tVal loss: %f\tVal acc: %f"
                % (
                    epoch,
                    pars.max_epochs,
                    train_loss_epoch[-1],
                    train_acc_epoch[-1],
                    val_loss_epoch[-1],
                    val_acc_epoch[-1],
                )
            )
        self.trained = True

    def predict_stm(
        self, input_data, input_sr=44100, save_output=False, output_path=None
    ):
        """Predict Dhrupad Bandish Segmentation

        :param input_data: path to audio file or numpy array like audio signal.
        :param input_sr: sampling rate of the input array of data (if any). This variable is only
            relevant if the input is an array of data instead of a filepath.
        :param save_output: boolean indicating whether the output figure for the estimation is
            stored.
        :param output_path: if the input is an array, and the user wants to save the estimation,
            the output_path must be provided, path/to/picture.png.
        """
        if not isinstance(save_output, bool):
            raise ValueError("save_output must be a boolean")
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError("Target audio not found.")
            audio, sr = librosa.load(input_data, sr=pars.fs)
            if output_path is None:
                output_path = os.path.basename(input_data).replace(
                    input_data.split(".")[-1], "png"
                )
        elif isinstance(input_data, np.ndarray):
            logger.warning(
                f"Resampling... (input sampling rate is {input_sr}Hz, make sure this is correct)"
            )
            audio = librosa.resample(input_data, orig_sr=input_sr, target_sr=pars.fs)
            if (save_output is True) and (output_path is None):
                raise ValueError(
                    "Please provide an output_path in order to save the estimation"
                )
        else:
            raise ValueError("Input must be path to audio signal or an audio array")

        if save_output is True:
            if not os.path.exists(os.path.basename(output_path)):
                os.mkdir(os.path.basename(output_path))
        if self.trained is False:
            raise ModelNotTrainedError(
                """
                Model is not trained. Please load model before running inference!
                You can load the pre-trained instance with the load_model wrapper.
            """
            )

        # convert to mel-spectrogram
        melgram = librosa.feature.melspectrogram(
            y=audio,
            sr=pars.fs,
            n_fft=pars.nfft,
            hop_length=pars.hopsize,
            win_length=pars.winsize,
            n_mels=pars.input_height,
            fmin=20,
            fmax=8000,
        )
        melgram = 10 * np.log10(1e-10 + melgram)
        melgram_chunks = makechunks(melgram, pars.input_len, pars.input_hop)

        # predict s.t.m. versus time
        stm_vs_time = []
        for chunk in melgram_chunks:
            model_in = (
                (torch.tensor(chunk).unsqueeze(0)).unsqueeze(1).float().to(self.device)
            )
            self.model.to(self.device)
            model_out = self.model.forward(model_in)
            model_out = torch.nn.Softmax(1)(model_out).detach().numpy()
            stm_vs_time.append(np.argmax(model_out))

        # smooth predictions with a minimum section duration of 5s
        stm_vs_time = smooth_boundaries(stm_vs_time, pars.min_sec_dur)

        # plot
        plt.plot(np.arange(len(stm_vs_time)) * 0.5, stm_vs_time)
        plt.yticks(np.arange(-1, 6), [""] + ["1", "2", "4", "8", "16"] + [""])
        plt.grid("on", linestyle="--", axis="y")
        plt.xlabel("Time (s)", fontsize=12)
        plt.ylabel("Surface tempo multiple", fontsize=12)
        if save_output is True:
            plt.savefig(output_path)
        else:
            plt.show()

        return stm_vs_time
