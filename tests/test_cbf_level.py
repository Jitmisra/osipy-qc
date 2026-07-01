"""Known-answer tests for Module 3 (CBF level, ratio, negative-GM)."""

import numpy as np

from osipy_qc.checks.cbf_level import cbf_level_check, gm_wm_ratio_check, negative_gm_check
from osipy_qc.core import Verdict


def _masks_from_values(gm_values, wm_values):
    """Pack GM and WM voxel values into a 1-D CBF array + boolean prob maps."""
    n = len(gm_values) + len(wm_values)
    cbf = np.zeros((n, 1, 1))
    gm = np.zeros((n, 1, 1))
    wm = np.zeros((n, 1, 1))
    for i, v in enumerate(gm_values):
        cbf[i, 0, 0] = v
        gm[i, 0, 0] = 1.0
    for j, v in enumerate(wm_values):
        k = len(gm_values) + j
        cbf[k, 0, 0] = v
        wm[k, 0, 0] = 1.0
    return cbf, gm, wm


def test_cbf_level_good_pass():
    cbf, gm, wm = _masks_from_values([52, 58, 50, 60, 55, 55], [20, 24, 19, 23, 22, 24])
    r = cbf_level_check(cbf=cbf, gm=gm, wm=wm)
    assert r.metric["mean_gm_cbf"] == 55.0
    assert r.metric["mean_wm_cbf"] == 22.0
    assert r.metric["median_wm_cbf"] == 22.5
    assert r.verdict == Verdict.PASS


def test_cbf_level_inflated_fail():
    cbf, gm, wm = _masks_from_values([175, 188, 170, 200, 182, 165], [60, 68, 62, 72, 65, 65])
    r = cbf_level_check(cbf=cbf, gm=gm, wm=wm)
    assert r.metric["mean_gm_cbf"] == 180.0     # > 150 -> FAIL
    assert r.verdict == Verdict.FAIL


def test_ratio_good_pass():
    cbf, gm, wm = _masks_from_values([52, 58, 50, 60, 55, 55], [20, 24, 19, 23, 22, 24])
    r = gm_wm_ratio_check(cbf=cbf, gm=gm, wm=wm)
    assert abs(r.metric["gm_wm_ratio"] - 2.5) < 1e-9   # 55/22
    assert r.verdict == Verdict.PASS


def test_ratio_weak_warn():
    cbf, gm, wm = _masks_from_values([34, 36, 35, 35], [29, 31, 30, 30])
    r = gm_wm_ratio_check(cbf=cbf, gm=gm, wm=wm)
    assert r.metric["gm_wm_ratio"] == round(35.0 / 30.0, 3)   # 1.167 (metric is 3-dp rounded)
    assert r.verdict == Verdict.WARN


def test_negative_gm_bands():
    # 3 of 100 negative -> PASS
    gm_vals = [-1.0, -2.0, -0.5] + [50.0] * 97
    cbf, gm, _ = _masks_from_values(gm_vals, [])
    r = negative_gm_check(cbf=cbf, gm=gm)
    assert r.metric["negGM_fraction"] == 0.03 and r.verdict == Verdict.PASS

    # 15 of 100 negative -> WARN
    gm_vals = [-1.0] * 15 + [50.0] * 85
    cbf, gm, _ = _masks_from_values(gm_vals, [])
    r = negative_gm_check(cbf=cbf, gm=gm)
    assert r.metric["negGM_fraction"] == 0.15 and r.verdict == Verdict.WARN

    # 25 of 100 negative -> FAIL
    gm_vals = [-1.0] * 25 + [50.0] * 75
    cbf, gm, _ = _masks_from_values(gm_vals, [])
    r = negative_gm_check(cbf=cbf, gm=gm)
    assert r.verdict == Verdict.FAIL


def test_negative_gm_na_on_raw_deltam():
    cbf, gm, _ = _masks_from_values([1.0, -1.0], [])
    r = negative_gm_check(cbf=cbf, gm=gm, is_raw_deltam=True)
    assert r.verdict == Verdict.NA
