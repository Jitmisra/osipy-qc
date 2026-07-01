"""Known-answer tests for Module 4 (coregistration Dice/Jaccard)."""

import numpy as np

from osipy_qc.checks.coreg import coregistration_check
from osipy_qc.core import Verdict


def _mask_from_indices(n, true_indices):
    m = np.zeros(n, dtype=bool)
    m[list(true_indices)] = True
    return m.reshape(n, 1, 1)


def test_coreg_good_pass():
    # A = 110 voxels (0..109), B = 105 voxels (5..109) -> overlap 100, union 115
    a = _mask_from_indices(150, range(0, 110))
    b = _mask_from_indices(150, range(5, 110))
    r = coregistration_check(asl_mask=a, struct_mask=b)
    assert r.metric["n_intersection"] == 105 or r.metric["n_intersection"] == 105
    # recompute expectation exactly: A=110, B=105, inter=105 (5..109), union=110
    # Dice = 2*105/(110+105) = 210/215 = 0.97674
    assert abs(r.metric["dice"] - 210 / 215) < 1e-3
    assert r.verdict == Verdict.PASS


def test_coreg_misregistered_fail():
    # A = 0..109 (110), B = 40..144 (105) -> overlap 40..109 = 70, union 145
    a = _mask_from_indices(150, range(0, 110))
    b = _mask_from_indices(150, range(40, 145))
    r = coregistration_check(asl_mask=a, struct_mask=b)
    assert r.metric["n_intersection"] == 70
    assert abs(r.metric["dice"] - 140 / 215) < 1e-3   # 0.65116
    assert abs(r.metric["jaccard"] - 70 / 145) < 1e-3  # 0.48276
    assert r.verdict == Verdict.FAIL


def test_coreg_missing_mask_unknown():
    a = _mask_from_indices(10, range(5))
    assert coregistration_check(asl_mask=a, struct_mask=None).verdict == Verdict.UNKNOWN
