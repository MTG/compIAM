import pytest
import mirdata
from compiam.dunya import Corpora
from compiam import list_models, load_dataset, load_corpora, \
    list_datasets, list_corpora


######################
# Test base operations
######################

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


########################
# Defining wrapper utils
########################

def _load_tf_models():
    from compiam import load_model
    from compiam.melody.ftanet_carnatic import FTANetCarnatic
    ftanet = load_model("melody:ftanet-carnatic")
    assert type(ftanet) == FTANetCarnatic

def _load_torch_models():
    from compiam import load_model
    from compiam.rhythm.tabla_transcription import FourWayTabla
    from compiam.structure.dhrupad_bandish_segmentation import DhrupadBandishSegmentation
    tabla_class = load_model("rhythm:4way-tabla")
    dhrupad_segmentation = load_model("structure:dhrupad-bandish-segmentation")
    assert type(tabla_class) == FourWayTabla
    assert type(dhrupad_segmentation) == DhrupadBandishSegmentation

def _load_ess_models():
    from compiam import load_model
    from compiam.melody.melodia import Melodia
    from compiam.melody.tonic_multipitch import TonicIndianMultiPitch
    melodia = load_model("melody:melodia")
    tonic_multipitch = load_model("melody:tonic-multipitch")
    assert type(melodia) == Melodia
    assert type(tonic_multipitch) == TonicIndianMultiPitch


####################
# Tensorflow testing
####################

@pytest.mark.tensorflow
def test_load_tf_models_tf():
    _load_tf_models()

@pytest.mark.essentia_tensorflow
def test_load_tf_models_ess_tf():
    _load_tf_models()

@pytest.mark.full_ml
def test_load_tf_models_full():
    _load_tf_models()

@pytest.mark.all
def test_load_tf_models_all():
    _load_tf_models()

@pytest.mark.torch
def test_no_tf():
    with pytest.raises(ImportError):
        from compiam.melody.ftanet_carnatic import FTANetCarnatic


###############
# Torch testing
###############

@pytest.mark.torch
def test_load_torch_models_torch():
    _load_torch_models()

@pytest.mark.essentia_torch
def test_load_torch_models_ess_torch():
    _load_torch_models()

@pytest.mark.full_ml
def test_load_torch_models_full():
    _load_torch_models()

@pytest.mark.all
def test_load_torch_models_all():
    _load_torch_models()

@pytest.mark.tensorflow
def test_no_torch():
    with pytest.raises(ImportError):
        from compiam.rhythm.tabla_transcription import FourWayTabla
        from compiam.structure.dhrupad_bandish_segmentation import DhrupadBandishSegmentation


##################
# Essentia testing
##################

@pytest.mark.essentia
def test_load_ess_models_ess():
    _load_ess_models()

@pytest.mark.essentia_tensorflow
def test_load_ess_models_ess_tf():
    _load_ess_models()

@pytest.mark.essentia_torch
def test_load_torch_models_ess_torch():
    _load_ess_models()

@pytest.mark.all
def test_load_torch_models_all():
    _load_ess_models()

@pytest.mark.full_ml
def test_no_ess():
    with pytest.raises(ImportError):
        from compiam.melody.melodia import Melodia



