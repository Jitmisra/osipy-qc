"""
Regression tests for the bugs the adversarial audit found. Each test FAILS on
the pre-fix code and PASSES after the fix — so they lock the bugs shut.
"""

import numpy as np

from osipy_qc.checks.cbf_level import cbf_level_check, gm_wm_ratio_check
from osipy_qc.checks.coreg import coregistration_check
from osipy_qc.checks.noise import histogram_check
from osipy_qc.checks.schema import classify_role, guess_vendor, swap_check
from osipy_qc.core import Verdict
from osipy_qc.utils.mathops import geometric_mean


def _gm(values):
    n = len(values)
    return np.array(values, float).reshape(n, 1, 1), np.ones((n, 1, 1))


def _gm_wm(gm_values, wm_values):
    n = len(gm_values) + len(wm_values)
    cbf = np.zeros((n, 1, 1)); gm = np.zeros((n, 1, 1)); wm = np.zeros((n, 1, 1))
    for i, v in enumerate(gm_values):
        cbf[i, 0, 0] = v; gm[i, 0, 0] = 1.0
    for j, v in enumerate(wm_values):
        cbf[len(gm_values) + j, 0, 0] = v; wm[len(gm_values) + j, 0, 0] = 1.0
    return cbf, gm, wm


# --- bug 1: WM fail bound was 10, design says 5 ---
def test_wm_mean_seven_is_warn_not_fail():
    cbf, gm, wm = _gm_wm([55, 55, 55, 55], [7, 7, 7, 7])   # WM 7: between 5 and 15
    r = cbf_level_check(cbf=cbf, gm=gm, wm=wm)
    assert r.metric["wm_verdict"] == Verdict.WARN.value     # not FAIL


# --- bug 2: a single NaN voxel turned the plain mean into NaN -> spurious FAIL ---
def test_swap_single_nan_voxel_not_spurious_fail():
    # control (even) slightly brighter than label (odd); one stray NaN voxel
    series = np.full((4, 4, 4, 8), 100.0)
    series[..., 0::2] += 2.0                                 # even = control, brighter
    series[0, 0, 0, 0] = np.nan                              # one corrupt voxel
    assert swap_check(asl_4d=series).verdict == Verdict.PASS   # NaN-robust, not FAIL


def test_swap_all_nan_is_unknown():
    series = np.full((4, 4, 4, 8), np.nan)
    assert swap_check(asl_4d=series).verdict == Verdict.UNKNOWN


# --- bug 3: 't1' substring mis-tagged an ASL file; vendor substring false positives ---
def test_classify_role_asl_beats_stray_t1():
    assert classify_role("PCASL_T1corrected.nii.gz") == "asl"
    assert classify_role("M0_scan.nii.gz") == "m0"
    assert classify_role("T1.nii.gz") == "t1"


def test_vendor_no_substring_false_positive():
    assert guess_vendor("brain_image_storage_montage") == "unknown"   # contains 'ge' substrings
    assert guess_vendor("GE_PCASL_Product_Sequence") == "GE"


def test_vendor_concatenated_name_still_detected():
    # real folder name has no separator after 'Siemens' — must still detect it
    assert guess_vendor("Siemens2DPCASL_No_M0") == "Siemens"
    assert guess_vendor("Siemens_BS3DPCASL") == "Siemens"


# --- bug 4: healthy right-skew was wrongly WARNed (OR logic) ---
# The skewness VERDICT has since been retracted entirely: its -0.5 cutoff was a
# generic statistics textbook heuristic (Bulmer 1979) with no ASL/CBF basis, and
# the "ExploreASL-style histogram QC" justification was untrue — neither
# ExploreASL nor ASLPrep computes CBF skewness at all. The check now reports the
# shape as INFO. This test is kept to pin the original bug's intent: a healthy
# right-skewed distribution must never be graded as a problem.
def test_histogram_right_skew_is_not_penalised():
    cbf, gm = _gm([40, 45, 48, 50, 52, 55, 90])   # positive (right) skew, no negatives
    r = histogram_check(cbf=cbf, gm=gm)
    assert r.metric["skewness"] > 0
    assert r.verdict == Verdict.INFO               # reported, never graded


# --- hardening: ratio of two negatives must not be a false PASS ---
def test_ratio_both_negative_is_fail():
    cbf, gm, wm = _gm_wm([-50, -50], [-20, -20])   # both negative -> ratio +2.5 if unguarded
    r = gm_wm_ratio_check(cbf=cbf, gm=gm, wm=wm)
    assert r.verdict == Verdict.FAIL


# --- hardening: empty masks -> UNKNOWN, not FAIL ---
def test_coreg_both_empty_is_unknown():
    empty = np.zeros((4, 4, 4), dtype=bool)
    assert coregistration_check(asl_mask=empty, struct_mask=empty).verdict == Verdict.UNKNOWN


# --- hardening: geometric_mean never leaks NaN on negative input ---
def test_geometric_mean_negative_collapses_to_zero():
    assert geometric_mean([-8.0, 1.0, 1.0]) == 0.0
    assert geometric_mean([-1.0, -1.0, 8.0]) == 0.0
    assert geometric_mean([0.0, 0.5, 0.5]) == 0.0
