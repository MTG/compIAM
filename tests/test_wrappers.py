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

def _load_tf_models():
    from compiam import load_model
    from compiam.melody.ftanet_carnatic import FTANetCarnatic
    ftanet = load_model("melody:ftanet-carnatic")
    assert type(ftanet) == FTANetCarnatic

def _load_torch_models():
    from compiam import load_model
    from compiam.rhythm.tabla_transcription import FourWayTabla
    tabla_class = load_model("rhythm:4way-tabla")
    assert type(tabla_class) == FourWayTabla

def _load_ess_models():
    from compiam import load_model
    from compiam.melody.melodia import Melodia
    melodia = load_model("melody:melodia")
    assert type(melodia) == Melodia

@pytest.mark.tensorflow
def test_load_tf_models_tf():
    _load_tf_models()

@pytest.mark.essentia_tensorflow
def test_load_tf_models_ess_tf():
    _load_tf_models()

@pytest.mark.full_ml
def test_load_tf_models_full():
    _load_tf_models()

@pytest.mark.torch
def test_no_tf():
    with pytest.raises(ImportError):
        from compiam.melody.ftanet_carnatic import FTANetCarnatic

@pytest.mark.torch
def test_load_torch_models_torch():
    _load_torch_models()

@pytest.mark.essentia_torch
def test_load_torch_models_ess_torch():
    _load_torch_models()

@pytest.mark.full_ml
def test_load_torch_models_full():
    _load_torch_models()

@pytest.mark.torch
def test_no_torch():
    with pytest.raises(ImportError):
        from compiam.rhythm.tabla_transcription import FourWayTabla

@pytest.mark.essentia
def test_load_ess_models_ess():
    _load_ess_models()

@pytest.mark.tensorflow
def test_load_ess_models_ess_tf():
    _load_ess_models()

@pytest.mark.essentia_torch
def test_load_torch_models_ess_torch():
    _load_ess_models()

@pytest.mark.full_ml
def test_no_ess():
    with pytest.raises(ImportError):
        from compiam.melody.melodia import Melodia



