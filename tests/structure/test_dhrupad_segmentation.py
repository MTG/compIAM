import os
import pytest
import subprocess

import numpy as np

from compiam import load_model
from compiam.data import WORKDIR, TESTDIR
from compiam.exceptions import ModelNotTrainedError


def _test_model():
    dbs = load_model("structure:dhrupad-bandish-segmentation")

    assert dbs.mode == "net"
    assert dbs.fold == 0

    dbs.update_fold(fold=1)
    assert dbs.fold == 1

    dbs.update_mode(mode="voc")
    assert dbs.mode == "voc"
    assert isinstance(dbs.model_path, dict)
    assert isinstance(dbs.loaded_model_path, str)

    with pytest.raises(FileNotFoundError):
        dbs.predict_stm(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))

    # dbs.predict_stm(
    #    file_path=os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav"),
    #    output_dir=os.path.join(TESTDIR, "resources", "melody")
    # )

    # subprocess.run(
    #    ["rm", os.path.join(TESTDIR, "resources", "melody", "pitch_test.png")]
    # )

    from compiam.structure.segmentation import DhrupadBandishSegmentation

    dbs = DhrupadBandishSegmentation()

    with pytest.raises(ModelNotTrainedError):
        dbs.predict_stm(os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav"))


@pytest.mark.torch
def test_predict_tf():
    _test_model()


@pytest.mark.essentia_torch
def test_predict_ess_tf():
    _test_model()


@pytest.mark.full_ml
def test_predict_full():
    _test_model()


@pytest.mark.all
def test_predict_all():
    _test_model()
