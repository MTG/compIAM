import pytest
from compiam import load_model
from compiam.rhythm import fourWayTabla

def test_load_model():
    tabla_class = load_model("rhythm:4way-tabla")
    assert type(tabla_class) == fourWayTabla