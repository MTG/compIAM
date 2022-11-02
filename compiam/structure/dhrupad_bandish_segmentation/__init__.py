import os
import glob
import librosa

import numpy as np
import matplotlib.pyplot as plt
from compiam.exceptions import ModelNotFoundError

from compiam.utils import get_logger

logger = get_logger(__name__)


class DhrupadBandishSegmentation:
    """Dhrupad Bandish Segmentation"""

    def __init__(
        self,
        mode="net",
        model_path=None,
        splits_path=None,
        annotations_path=None,
        features_path=None,
        original_audios_path=None,
        processed_audios_path=None,
        device=None,
    ):
        """Dhrupad Bandish Segmentation init method.

        :param model_path: path to file to the model weights.
        :param device: indicate whether the model will run on the GPU.
        """
        ###
        try:
            global torch
            import torch

            global split_audios
            from compiam.structure.dhrupad_bandish_segmentation.audio_processing import (
                split_audios,
            )

            global extract_features, makechunks
            from compiam.structure.dhrupad_bandish_segmentation.feature_extraction import (
                extract_features,
                makechunks,
            )

            global class_to_categorical, categorical_to_class, build_model, smooth_boundaries
            from compiam.structure.dhrupad_bandish_segmentation.model_utils import (
                class_to_categorical,
                categorical_to_class,
                build_model,
                smooth_boundaries,
            )

            global pars
            import compiam.structure.dhrupad_bandish_segmentation.params as pars
        except:
            raise ImportError(
                "In order to use this tool you need to have torch installed. "
                "Please reinstall compiam using `pip install compiam[torch]`"
            )
        ###

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load mode by default: update with self.update mode()
        self.mode = mode
        self.classes = pars.classes_dict[self.mode]

        self.model = None
        self.model_path = model_path

        if self.model_path is not None:
            if not os.path.exists(
                os.path.join(self.model_path, self.mode, "saved_model_fold_0.pt")
            ):
                raise ModelNotFoundError("""
                    Given path to model weights not found. Make sure you enter the path correctly.
                    A training process for the FTA-Net tuned to Carnatic is under development right
                    now and will be added to the library soon. Meanwhile, we provide the weights in the
                    latest repository version (https://github.com/MTG/compIAM) so make sure you have these
                    available before loading the Carnatic FTA-Net.
                """)
            self.load_model()  # Loading pre-trained model for given mode


        self.splits_path = splits_path
        self.annotations_path = annotations_path
        self.features_path = features_path
        self.original_audios_path = original_audios_path
        self.processed_audios_path = processed_audios_path


    def load_model(self):
        """TODO
        """
        self.model = (
            build_model(pars.input_height, pars.input_len, len(self.classes))
            .float()
            .to(self.device)
        )
        self.model.load_state_dict(
            torch.load(os.path.join(self.model_path), map_location=self.device)
        )
        self.model.eval()


    def update_mode(self, mode):
        """TODO
        """
        self.mode = mode
        self.classes = pars.classes_dict[mode]
        self.load_model()


    def train(self, fold=0, verbose=0):
        """Train the Dhrupad Bandish Segmentation model

        :param data_dir: path to extracted features and labels.
        :param mode: model mode: "voc", "pakh" or "net"
        :param fold: fold id 0, 1 or 2
        :param verbose: showing details of the model
        """
        print("Splitting audios...")
        split_audios(
            self.processed_audios_path, self.annotations_path, self.original_audios_path
        )
        print("Extracting features...")
        extract_features(
            self.processed_audios_path, self.annotations_path, self.features_path, self.mode
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
                        self.splits_path,
                        self.mode,
                        "fold_" + i_fold + ".csv",
                        delimiter=",",
                        dtype="str",
                    )
                )
            )

        val_fold = all_folds[fold]
        train_fold = np.array([])
        for i_fold in np.delete(np.arange(0, n_folds), fold):
            if len(train_fold) == 0:
                train_fold = all_folds[i_fold]
            else:
                train_fold = np.append(train_fold, all_folds[i_fold])

        for song in songlist:
            try:
                ids = glob.glob(self.features_path + song + "/*.pt")
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
        self.model = (
            build_model(pars.input_height, pars.input_len, len(self.classes))
            .float()
            .to(self.device)
        )
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        if verbose == 1:
            print(self.model)
            n_params = 0
            for param in self.model.parameters():
                n_params += torch.prod(torch.tensor(param.shape))
            print("Num of trainable params: %d\n" % n_params)

        ##training epochs loop
        train_loss_epoch = []
        train_acc_epoch = []
        val_loss_epoch = []
        val_acc_epoch = []
        n_idle = 0

        if not os.path.exists(os.path.join(self.model_path, self.mode)):
            os.makedir(os.path.join(self.model_path, self.mode))

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
                        self.model_path, self.mode, "saved_model_fold_%d.pt" % fold
                    ),
                )
                n_idle = 0
            else:
                n_idle += 1

            # print loss in current epoch
            print(
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

    def predict_stm(self, path_to_file, output_dir=None):
        """Dhrupad Bandish Segmentation init method.

        :param path_to_file: path of the input file
        :param mode: model mode: "voc", "pakh" or "net"
        :param output_dir: directory to store printed outputs
        """
        if not os.path.exists(path_to_file):
            raise ValueError("Input file not found")
        if not os.path.exists(output_dir):
            raise FileNotFoundError(
                """
                Folder to store output does not exists or it is not specified. Please enter a valid folder
                to store the outputs."""
            )

        # load input audio
        audio, fs = librosa.load(path_to_file, sr=fs)

        # convert to mel-spectrogram
        melgram = librosa.feature.melspectrogram(
            audio,
            sr=fs,
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
        plt.savefig(
            os.path.join(output_dir, path_to_file.split("/")[-1].split(".wav")[0])
            + ".png"
        )
