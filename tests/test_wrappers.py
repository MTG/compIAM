import pytest
import mirdata
from compiam import list_models, load_dataset, load_corpora, list_datasets


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
        load_corpora(tradition="hola", token="test")
    with pytest.raises(ImportError):
        load_corpora("carnatic", token=None)


def test_lists():
    assert type(list_models()) is list
    assert type(list_datasets()) is list
    assert "melody:ftanet-carnatic" in list_models()
    assert "saraga_carnatic" in list_datasets()


########################
# Defining wrapper utils
########################


def _load_torch_models():
    from compiam import load_model
    from compiam.structure.segmentation.dhrupad_bandish_segmentation import (
        DhrupadBandishSegmentation,
    )

    dhrupad_segmentation = load_model("structure:dhrupad-bandish-segmentation")
    assert type(dhrupad_segmentation) == DhrupadBandishSegmentation


#############################################
# Load model with optional dependency testing
#############################################


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
