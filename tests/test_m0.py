"""Known-answer tests for Module 6 (M0 calibration)."""

import numpy as np

from osipy_qc.checks.m0 import (
    m0_geometry_check,
    m0_no_bs_check,
    m0_present_check,
    m0_tr_check,
)
from osipy_qc.core import Verdict


def test_m0_present():
    assert m0_present_check(m0_type="separate").verdict == Verdict.PASS
    assert m0_present_check(m0_type="absent").verdict == Verdict.WARN


def test_m0_tr_pass_and_correction():
    r5 = m0_tr_check(m0_tr_s=5.0)
    assert r5.verdict == Verdict.PASS and r5.metric["correction_factor"] == 1.0

    r4 = m0_tr_check(m0_tr_s=4.0)        # 1/(1-exp(-4/1.4)) = 1.0609
    assert r4.verdict == Verdict.WARN
    expected = 1.0 / (1.0 - np.exp(-4.0 / 1.4))
    assert abs(r4.metric["correction_factor"] - round(expected, 4)) < 1e-4


def test_m0_no_bs():
    assert m0_no_bs_check(m0_background_suppression=False).verdict == Verdict.PASS
    assert m0_no_bs_check(m0_background_suppression=True).verdict == Verdict.FAIL
    assert m0_no_bs_check(m0_background_suppression=None).verdict == Verdict.UNKNOWN


def test_m0_geometry():
    # GE: M0 128x128x40 == ASL -> PASS
    assert m0_geometry_check(m0_shape=(128, 128, 40), asl_shape=(128, 128, 40)).verdict == Verdict.PASS
    # mismatch -> WARN
    assert m0_geometry_check(m0_shape=(64, 64, 30), asl_shape=(80, 80, 20)).verdict == Verdict.WARN
