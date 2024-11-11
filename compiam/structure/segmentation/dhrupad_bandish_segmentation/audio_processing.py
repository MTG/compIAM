import os
import librosa

import numpy as np
import soundfile as sf

from compiam.structure.segmentation.dhrupad_bandish_segmentation.params import fs
from compiam.utils import get_logger

logger = get_logger(__name__)



def split_audios(save_dir=None, annotations_path=None, audios_path=None):
    """Split audio of Dhrupad dataset

    :param save_dir: path where to save the splits
    :param annotations_path: path where to find the annotations
    :param audios_path: path where to find the original audios
    """
    if not os.path.exists(save_dir):
        logger.warning(
            """Save directory not found. Creating it...
        """
        )
        os.mkdir(save_dir)

    if not os.path.exists(annotations_path):
        raise ValueError(
            """
            Path to annotations not found."""
        )

    if not os.path.exists(audios_path):
        raise ValueError(
            """
            Path to original audios not found."""
        )

    annotations = np.loadtxt(
        os.path.join(annotations_path, "section_boundaries_labels.csv"),
        delimiter=",",
        dtype=str,
    )

    song = ""  # please leave this line as it is
    for item in annotations:
        if "_".join(item[0].split("_")[:-1]) != song:
            song = "_".join(item[0].split("_")[:-1])
            try:
                x, _ = librosa.load(os.path.join(audios_path, song + ".wav"), sr=fs)
            except FileNotFoundError:
                logger.error(
                    f"""
                    Audio for {song} not found. Please make sure you check:
                    models/structure/dhrupad_bandish_segmentation/original_audio/README.md
                """
                )
                continue

        start = int(float(item[1]) * fs)
        end = int(float(item[2]) * fs)
        y = x[start:end]
        sf.write(os.path.join(save_dir, item[0] + ".wav"), y, fs)
