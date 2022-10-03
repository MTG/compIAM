import pytest
from compiam import load_model
from compiam.melody import ftanetCarnatic

def test_load_model():
    melodia = load_model("melody:melodia")
    ftanet = load_model("melody:ftanet-carnatic")
    assert type(ftanet) == ftanetCarnatic