import os
import pytest
import subprocess

from compiam.data import TESTDIR
from compiam.exceptions import DatasetNotLoadedError, ModelNotTrainedError

test_files = [
    os.path.join(TESTDIR, "resources", "timbre", "224030__akshaylaya__bheem-b-001.wav"),
    os.path.join(TESTDIR, "resources", "timbre", "225359__akshaylaya__cha-c-001.wav"),
]


def _predict_strokes():
    if not os.path.exists(os.path.join(TESTDIR, "resources", "mir_datasets")):
        subprocess.run(["mkdir", os.path.join(TESTDIR, "resources", "mir_datasets")])
    from compiam.timbre.stroke_classification import MridangamStrokeClassification

    mridangam_stroke_class = MridangamStrokeClassification()
    with pytest.raises(DatasetNotLoadedError):
        mridangam_stroke_class.train_model()
    with pytest.raises(ValueError):
        mridangam_stroke_class.train_model(load_computed=True)
    with pytest.raises(ModelNotTrainedError):
        mridangam_stroke_class.predict(test_files)
    mridangam_stroke_class.load_mridangam_dataset(
        data_home=os.path.join(TESTDIR, "resources", "mir_datasets"),
        download=True,
    )
    assert mridangam_stroke_class.list_strokes() == [
        "bheem",
        "cha",
        "dheem",
        "dhin",
        "num",
        "ta",
        "tha",
        "tham",
        "thi",
        "thom",
    ]
    assert mridangam_stroke_class.dict_strokes() == {
        0: "bheem",
        1: "cha",
        2: "dheem",
        3: "dhin",
        4: "num",
        5: "ta",
        6: "tha",
        7: "tham",
        8: "thi",
        9: "thom",
    }
    acc = mridangam_stroke_class.train_model()
    assert acc > 90
    preds = mridangam_stroke_class.predict(test_files)
    assert isinstance(preds, dict)
    assert len(list(preds.keys())) == 2
    subprocess.run(["rm", "-r", os.path.join(TESTDIR, "resources", "mir_datasets")])


@pytest.mark.essentia
def test_strokes_ess():
    _predict_strokes()


@pytest.mark.all
def test_strokes_ess_all():
    _predict_strokes()
