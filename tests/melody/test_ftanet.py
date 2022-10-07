import os
import pytest

from compiam.data import WORKDIR
from compiam import load_model

@pytest.mark.tensorflow
def test_predict():
    import tensorflow as tf
    ftanet = load_model("melody:ftanet-carnatic")
    with pytest.raises(ValueError):
        ftanet.predict(os.path.join(WORKDIR, "tests", "resources", \
            "melody", "hola.wav"))
    #pitch = ftanet.predict(os.path.join(WORKDIR, "tests", "resources", \
    #    "melody", "test.wav"))

    #assert pitch here