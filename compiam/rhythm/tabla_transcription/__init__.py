import os

import numpy as np

from compiam.exceptions import ModelNotTrainedError
from compiam.utils import get_logger
from compiam.visualisation.training import plot_losses

logger = get_logger(__name__)


class FourWayTabla:
    """TODO"""

    def __init__(
        self, model_path=None, n_folds=3, seq_length=15, hop_dur=10e-3, device=None
    ):
        """TODO"""
        ###
        try:
            global torch
            import torch

            global onsetCNN_D, onsetCNN_RT, onsetCNN
            from compiam.rhythm.tabla_transcription.model import (
                onsetCNN_D,
                onsetCNN_RT,
                onsetCNN,
            )

            global TablaDataset, gen_melgrams, peakPicker, make_train_val_split, load_mel_data
            from compiam.rhythm.tabla_transcription.utils import (
                TablaDataset,
                gen_melgrams,
                peakPicker,
                make_train_val_split,
                load_mel_data,
            )

        except:
            raise ImportError(
                "In order to use this tool you need to have torch installed. "
                "Please reinstall compiam using pip install compiam[torch] or "
                "install torch with pip install torch."
            )
        ###

        if not device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.categories = ["D", "RT", "RB", "B"]
        self.model_names = {
            "D": onsetCNN_D,
            "RT": onsetCNN_RT,
            "RB": onsetCNN,
            "B": onsetCNN,
        }
        self.models = {}
        self.stats = None
        self.n_folds = n_folds
        self.seq_length = seq_length
        self.hop_dur = hop_dur

        # Load model if passed
        self.model_path = model_path
        if self.model_path:
            self.load_models(model_path)

    def load_models(self, model_path):
        """TODO"""
        stats_path = os.path.join(model_path, "means_stds.npy")
        self.stats = np.load(stats_path)
        for cat in self.categories:
            self.models[cat] = {}
            for fold in range(self.n_folds):
                saved_model_path = os.path.join(
                    model_path, cat, "saved_model_%d.pt" % fold
                )
                model = self.model_names[cat]().double().to(self.device)
                model.load_state_dict(
                    torch.load(saved_model_path, map_location=self.device)
                )
                model.eval()

                self.models[cat][fold] = model

    def train_epoch(self, model, training_generator, optimizer, criterion=None):
        # Torch is imported inside __init__
        if criterion == None:
            criterion = torch.nn.BCELoss(reduction="none")
        model.train()
        n_batch = 0
        loss_epoch = 0
        for local_batch, local_labels, local_weights in training_generator:
            n_batch += 1

            # transfer to GPU
            local_batch, local_labels, local_weights = (
                local_batch.double().to(self.device),
                local_labels.double().to(self.device),
                local_weights.double().to(self.device),
            )

            # model forward pass
            optimizer.zero_grad()
            outs = model(local_batch).squeeze()
            outs = outs.double()

            # compute loss
            loss = criterion(outs, local_labels)
            loss = loss.double()
            loss = torch.dot(loss, local_weights)
            loss /= local_batch.size()[0]
            loss_epoch += loss.item()

            # update weights
            loss.backward()
            optimizer.step()

        return model, loss_epoch / n_batch

    def validate(self, model, criterion, validation_generator):
        """TODO"""
        model.eval()
        n_batch = 0
        loss_epoch = 0

        with torch.set_grad_enabled(False):
            for local_batch, local_labels, local_weights in validation_generator:
                n_batch += 1

                # transfer to GPU
                local_batch, local_labels = local_batch.double().to(
                    self.device
                ), local_labels.double().to(self.device)

                # model forward pass
                outs = model(local_batch).squeeze()
                outs = outs.double()

                # compute loss
                loss = criterion(outs, local_labels).mean()
                loss_epoch += loss.item()

        return model, loss_epoch / n_batch

    def train(
        self,
        model_path,
        categories=["D", "RT", "RB", "B"],
        model_kwargs={},
        criterion=None,
        max_epochs=50,
        early_stop_patience=10,
        lr=1e-4,
        val_fold=0,
        batch_size=256,
        num_workers=4,
        seq_length=15,
        n_channels=3,
        gpu_id=0,
        aug_method=None,
        model_save_dir=None,
        plot_save_dir=None,
    ):
        """Train 4-way model on input data. By default, four separate models will be trained on
        the input data. Adam optimizer
        """
        if not isinstance(categories, (np.ndarray, list)):
            categories = [categories]

        if not isinstance(model_kwargs, (np.ndarray, list)):
            model_kwargs = [model_kwargs]

        if len(model_kwargs) == 1:
            # one model kwargs dict for each category model
            model_kwargs = model_kwargs * len(categories)
        elif len(model_kwargs) != len(categories):
            raise Exception(
                "<model_kwargs> should be either a single dict or a list of dicts equal in length to <categories>"
            )

        # Torch is imported in __init__
        if criterion == None:
            criterion = torch.nn.BCELoss(reduction="none")

        use_cuda = torch.cuda.is_available()
        device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")

        for ci, c in enumerate(categories):
            logger.info(f"Training category {c} ({ci}/{len(categories)}")

            mk = model_kwargs[ci]
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            if c in self.models:
                logger.warning(f"Category {c} model already exists, retraining...")

            model = self.model_names[c](**mk).double().to(device)

            logger.info("Getting train and val generators")
            training_generator, validation_generator = self.get_generators(
                model_path,
                aug_method,
                val_fold,
                c,
                seq_length,
                n_channels,
                batch_size,
                num_workers,
            )

            # train-val loop
            train_loss_epoch = []
            val_loss_epoch = []
            for epoch in range(max_epochs):
                ##training
                model, loss_epoch = self.train_epoch(
                    model, training_generator, optimizer, criterion
                )
                train_loss_epoch.append(loss_epoch)

                ##validation
                model, loss_epoch = self.validate(
                    model, criterion, validation_generator
                )
                val_loss_epoch.append(loss_epoch)

                # print loss in current epoch
                logger.info(
                    f"Epoch no: {epoch}/{max_epochs}\tTrain loss: {train_loss_epoch[-1]}\tVal loss: {val_loss_epoch[-1]}"
                )

                # save model if val loss is minimum so far
                if val_loss_epoch[-1] == min(val_loss_epoch):
                    if model_save_dir:
                        torch.save(
                            model.state_dict(),
                            os.path.join(model_save_dir, f"saved_model_{val_fold}.pt"),
                        )
                    # reset early stop count to 0 since new minimum loss reached
                    early_stop_count = 0
                else:
                    early_stop_count += 1

                if early_stop_count == early_stop_patience:
                    logger.info(f"Early stopping at {epoch}")
                    break

            if plot_save_dir:
                # plot losses vs epoch
                plot_save_filepath = os.path.join(
                    plot_save_dir, f"loss_curves_{val_fold}.png"
                )
                plot_losses(train_loss_epoch, val_loss_epoch, plot_save_filepath)

    def predict(self, path_to_audio, predict_thresh=0.3):
        """TODO"""
        if not self.models:
            raise ModelNotTrainedError("Please load or train model before predicting")

        # get log-mel-spectrogram of audio
        melgrams = gen_melgrams(path_to_audio, stats=self.stats)

        # get frame-wise onset predictions
        n_frames = melgrams.shape[-1] - self.seq_length
        odf = dict(zip(self.categories, [np.zeros(n_frames)] * 4))
        for i_frame in np.arange(0, n_frames):
            x = (
                torch.tensor(melgrams[:, :, i_frame : i_frame + self.seq_length])
                .double()
                .to(self.device)
            )
            x = x.unsqueeze(0)

            for cat in self.categories:
                y = 0
                for fold in range(self.n_folds):
                    model = self.models[cat][fold]

                    y += model(x).squeeze().cpu().detach().numpy()
                odf[cat][i_frame] = y / self.n_folds

        # pick peaks in predicted activations
        odf_peaks = dict(zip(self.categories, [] * 4))
        for cat in self.categories:
            odf_peaks[cat] = peakPicker(odf[cat], predict_thresh)

        onsets = np.concatenate([odf_peaks[cat] for cat in odf_peaks])
        onsets = np.array(onsets * self.hop_dur, dtype=float)
        labels = np.concatenate([[cat] * len(odf_peaks[cat]) for cat in odf_peaks])

        sorted_order = onsets.argsort()
        onsets = onsets[sorted_order]
        labels = labels[sorted_order]

        return onsets, labels

    def get_generators(
        self,
        model_path,
        aug_method,
        fold,
        category,
        seq_length,
        n_channels,
        batch_size,
        num_workers,
    ):
        # make train-val splits
        logger.info("Making train-val splits")
        # these are temp files that will be created and overwritten during every
        # training run; they contain the frame-wise onset labels for a given category
        # and weights to be applied during loss computation
        train_val_data_filepaths = {
            "train": os.path.join(
                model_path, f"labels_weights_train_{aug_method}.hdf5"
            ),
            "validation": os.path.join(
                model_path, f"labels_weights_val_{aug_method}.hdf5"
            ),
        }

        # get list of audios in each CV fold
        split_dir = os.path.join(model_path, "cv_folds", "")
        folds = {"val": fold, "train": np.delete([0, 1, 2], fold)}
        splits = dict(
            zip(
                [0, 1, 2],
                [
                    np.loadtxt(
                        os.path.join(split_dir, f"3fold_cv_{fold}.fold"), dtype=str
                    )
                    for fold in range(3)
                ],
            )
        )

        # create training and validation splits of data and save them to disk as temporary files
        labels_weights_orig_filepath = os.path.join(
            model_path, f"labels_weights_orig_{category}.hdf5"
        )
        labels_weights_aug_filepath = os.path.join(
            model_path, f"labels_weights_{aug_method}_{category}.hdf5"
        )
        make_train_val_split(
            folds,
            labels_weights_orig_filepath,
            labels_weights_aug_filepath,
            train_val_data_filepaths,
        )

        # load all melgram-label pairs as a dict to memory for faster training (ensure sufficient RAM size apriori)
        songlist_orig = np.loadtxt(
            os.path.join(model_path, "songlists", "songlist_orig.txt"), dtype=str
        )
        songlist_aug = np.loadtxt(
            os.path.join(
                model_path, "songlists", "songlist_" + str(aug_method) + ".txt"
            ),
        )
        mel_data = load_mel_data(model_path, folds, splits, songlist_orig, songlist_aug)

        # data loaders
        params = {"batch_size": batch_size, "shuffle": True, "num_workers": num_workers}
        training_set = TablaDataset(
            train_val_data_filepaths["train"],
            seq_length=seq_length,
            n_channels=n_channels,
            mel_data=mel_data,
        )
        training_generator = torch.utils.data.th_data.DataLoader(training_set, **params)

        validation_set = TablaDataset(
            train_val_data_filepaths["validation"],
            seq_length=seq_length,
            n_channels=n_channels,
            mel_data=mel_data,
        )
        validation_generator = torch.utils.data.th_data.DataLoader(
            validation_set, **params
        )

        return training_generator, validation_generator
