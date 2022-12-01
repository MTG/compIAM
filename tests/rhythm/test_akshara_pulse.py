import os
import pytest

import numpy as np

from compiam.data import TESTDIR


def test_tool():
    from compiam.rhythm.meter import AksharaPulseTracker
    apt = AksharaPulseTracker()

    with pytest.raises(FileNotFoundError):
        apt.extract(
            os.path.join(TESTDIR, "resources", "melody", "hola.wav")
        )

    pulses = apt.extract(
        os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav")
    )
    assert np.all(
        np.isclose(
            pulses,
            np.array([0.75403026, 1.70514967, 1.97136237])
        )
    )
