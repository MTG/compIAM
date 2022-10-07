import pytest
import mirdata
from compiam.dunya import Corpora
from compiam import list_models, load_dataset, load_corpora, \
    list_datasets, list_corpora

def test_load_dataset():
    dataset = load_dataset("mridangam_stroke")
    dataset_mirdata = mirdata.initialize("mridangam_stroke")
    assert type(dataset) == type(dataset_mirdata)
    with pytest.raises(ValueError):
        load_dataset("hola")

def test_load_corpora():
    with pytest.raises(ValueError):
        load_corpora("hola")
    with pytest.raises(ValueError):
        load_corpora("hola", cc="hola")
    corpora = load_corpora("carnatic", token="hola")
    assert type(corpora) == Corpora

def test_lists():
    assert type(list_models()) is list
    assert type(list_datasets()) is list
    assert type(list_corpora()) is list
    assert "melody:melodia" in list_models()
    assert "saraga_carnatic" in list_datasets()
    assert "hindustani" in list_corpora()

@pytest.mark.tensorflow
def test_load_tensorflow_models():
    from compiam import load_model
    from compiam.melody import FTANetCarnatic
    melodia = load_model("melody:melodia")
    ftanet = load_model("melody:ftanet-carnatic")
    assert type(ftanet) == FTANetCarnatic

@pytest.mark.torch
def test_load_torch_models():
    from compiam import load_model
    from compiam.rhythm import FourWayTabla
    tabla_class = load_model("rhythm:4way-tabla")
    assert type(tabla_class) == FourWayTabla


