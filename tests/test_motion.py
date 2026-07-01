"""Known-answer tests for Module 7 (motion FWD/DVARS)."""

import numpy as np

from osipy_qc.checks.motion import dvars, framewise_displacement, motion_check
from osipy_qc.core import Verdict


def test_fwd_single_pair_hand_computed():
    # trans 0.10+0.05+0.20 = 0.35 ; rot 50*(0.005+0.002+0.003) = 0.50 ; FWD = 0.85
    mp = np.array([[0, 0, 0, 0, 0, 0],
                   [0.10, 0.05, 0.20, 0.005, 0.002, 0.003]])
    fwd = framewise_displacement(mp)
    assert fwd.shape == (1,)
    assert abs(fwd[0] - 0.85) < 1e-9


def test_fwd_rotation_scaling():
    # pure 0.01 rad rotation about one axis -> 50 * 0.01 = 0.5 mm
    mp = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0.01, 0, 0]])
    assert abs(framewise_displacement(mp)[0] - 0.5) < 1e-9


def test_motion_verdict_bands():
    calm = np.cumsum(np.full((20, 6), 0.001), axis=0)      # tiny steady drift -> low FWD
    assert motion_check(motion_params=calm).verdict == Verdict.PASS

    jerky = np.array([[0, 0, 0, 0, 0, 0],
                      [0.10, 0.05, 0.20, 0.005, 0.002, 0.003]])  # FWD 0.85 -> FAIL
    assert motion_check(motion_params=jerky).verdict == Verdict.FAIL


def test_dvars_from_series():
    rng = np.random.default_rng(0)
    series = 100.0 + rng.normal(0, 5.0, (4, 4, 4, 10))
    dv = dvars(series)
    assert dv.shape == (9,)
    assert np.all(dv > 0)


def test_motion_missing_input_unknown():
    assert motion_check().verdict == Verdict.UNKNOWN
