import os
import pytest

from compiam.data import WORKDIR
from compiam import load_model

def _run_prediction():
    ftanet = load_model("melody:ftanet-carnatic")
    with pytest.raises(ValueError):
        ftanet.predict(os.path.join(WORKDIR, "tests", "resources", \
            "melody", "hola.wav"))
    #pitch = ftanet.predict(os.path.join(WORKDIR, "tests", "resources", \
    #    "melody", "test.wav"))

    #assert pitch here

@pytest.mark.tensorflow
def test_predict_tf():
    _run_prediction()

@pytest.mark.essentia
def test_predict_ess():
    _run_prediction()