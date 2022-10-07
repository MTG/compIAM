import os
import pytest

from compiam.data import WORKDIR
from compiam import load_model

@pytest.mark.essentia
def test_essentia_extractors():
    melodia = load_model("melody:melodia")
    with pytest.raises(ValueError):
        melodia.extract(os.path.join(WORKDIR, "tests", "resources", \
            "melody", "hola.wav"))
    #pitch = melodia.predict(os.path.join(WORKDIR, "tests", "resources", \
    #    "melody", "test.wav"))
        
    # assert pitch here

    tonic_multipitch = load_model("melody:tonic-multipitch")
    with pytest.raises(ValueError):
        tonic_multipitch.extract(os.path.join(WORKDIR, "tests", "resources", \
            "melody", "hola.wav"))
    #tonic = tonic_multipitch.predict(os.path.join(WORKDIR, "tests", "resources", \
    #    "melody", "test.wav"))

    # assert tonic here

    # assert normalisation here