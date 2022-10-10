import os
import pytest

from compiam.data import WORKDIR
from compiam import load_model

def _predict():
    ftanet = load_model("melody:ftanet-carnatic")
    with pytest.raises(ValueError):
        ftanet.predict(os.path.join(WORKDIR, "tests", "resources", \
            "melody", "hola.wav"))
    #pitch = ftanet.predict(os.path.join(WORKDIR, "tests", "resources", \
    #    "melody", "test.wav"))

    # assert pitch

@pytest.mark.tensorflow
def test_predict_tf():
    _predict()

@pytest.mark.essentia_tensorflow
def test_predict_ess_tf():
    _predict()

@pytest.mark.full_ml
def test_predict_full():
    _predict()
