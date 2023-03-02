import os
import pytest

import numpy as np

from compiam.data import TESTDIR


def test_tool():
    from compiam.rhythm.meter import AksharaPulseTracker

    apt = AksharaPulseTracker()

    with pytest.raises(FileNotFoundError):
        apt.extract(os.path.join(TESTDIR, "resources", "melody", "hola.wav"))

    pulses = apt.extract(os.path.join(TESTDIR, "resources", "melody", "pitch_test.wav"))
    assert pulses["aksharaPeriod"] == 0.316
    assert pulses["sections"] == {
        "startTime": np.array([0.0]),
        "endTime": np.array([2.009]),
        "label": "Kriti",
    }
    assert np.all(
        np.isclose(
            pulses["aksharaPulses"], np.array([0.75403026, 1.70514967, 1.97136237])
        )
    )
    assert np.all(
        np.isclose(
            pulses["APcurve"],
            np.array(
                [
                    [0.0, 0.31496062992125984],
                    [0.4992290249433111, 0.31496062992125984],
                    [0.9984580498866213, 0.31496062992125984],
                    [1.4976870748299325, 0.31496062992125984],
                ]
            ),
        )
    )
