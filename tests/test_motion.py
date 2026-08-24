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


def test_motion_calm_passes():
    calm = np.cumsum(np.full((20, 6), 0.001), axis=0)      # tiny steady drift -> low FWD
    r = motion_check(motion_params=calm)
    assert r.verdict == Verdict.PASS
    assert r.metric["n_frames_over_censor"] == 0


def test_motion_fails_on_mean_fwd_above_aslprep_rule():
    """The ONE citable motion verdict: Adebimpe 2022 excludes participants with
    MEAN framewise displacement > 1 mm. So the FAIL must key off the mean."""
    # each step: trans 0.4*3 = 1.2 mm -> mean FWD 1.2 > 1.0
    jerky = np.cumsum(np.tile([0.4, 0.4, 0.4, 0, 0, 0], (10, 1)), axis=0)
    r = motion_check(motion_params=jerky)
    assert r.metric["mean_fwd_mm"] > 1.0
    assert r.verdict == Verdict.FAIL


def test_motion_below_mean_rule_but_over_censor_line_warns():
    """mean FWD 0.85 mm is BELOW ASLPrep's 1 mm exclusion rule, so it must not
    FAIL — but the frame exceeds Power 2012's 0.5 mm per-frame censoring line,
    which is worth a WARN. (An earlier version wrongly FAILed this.)"""
    jerky = np.array([[0, 0, 0, 0, 0, 0],
                      [0.10, 0.05, 0.20, 0.005, 0.002, 0.003]])  # FWD 0.85
    r = motion_check(motion_params=jerky)
    assert abs(r.metric["mean_fwd_mm"] - 0.85) < 1e-6
    assert r.metric["n_frames_over_censor"] == 1
    assert r.verdict == Verdict.WARN


def test_motion_non_strict_demotes_fail():
    from osipy_qc.core.config import QCConfig

    jerky = np.cumsum(np.tile([0.4, 0.4, 0.4, 0, 0, 0], (10, 1)), axis=0)
    assert motion_check(motion_params=jerky, cfg=QCConfig(strict=False)).verdict == Verdict.WARN


def test_dvars_from_series():
    rng = np.random.default_rng(0)
    series = 100.0 + rng.normal(0, 5.0, (4, 4, 4, 10))
    dv = dvars(series)
    assert dv.shape == (9,)
    assert np.all(dv > 0)


def test_motion_missing_input_unknown():
    assert motion_check().verdict == Verdict.UNKNOWN


# --------------------------------------------------------------------------- #
# NaN-robust DVARS (OSIPI Challenge DROs mask their background with NaN;
# 5.5% NaN voxels used to poison the RMS of every frame pair -> UNKNOWN
# "insufficient data" on 59 measurable volumes)
# --------------------------------------------------------------------------- #
def test_dvars_ignores_nan_background():
    rng = np.random.default_rng(1)
    series = 100.0 + rng.normal(0, 5.0, (6, 6, 6, 8))
    nan_mask = np.zeros((6, 6, 6), dtype=bool)
    nan_mask[0, :, :] = True                      # a NaN "background" slab
    masked = series.copy()
    masked[nan_mask, :] = np.nan
    dv = dvars(masked)
    assert dv.shape == (7,) and np.all(np.isfinite(dv))
    # must equal DVARS computed over only the finite voxels
    expected = dvars(series, mask=~nan_mask)
    assert np.allclose(dv, expected)


def test_dvars_mask_path_is_also_nan_robust():
    rng = np.random.default_rng(2)
    series = 100.0 + rng.normal(0, 5.0, (5, 5, 5, 6))
    series[0, 0, 0, :] = np.nan                   # one NaN voxel INSIDE the mask
    dv = dvars(series, mask=np.ones((5, 5, 5), dtype=bool))
    assert np.all(np.isfinite(dv))


def test_motion_info_notes_nan_exclusion():
    rng = np.random.default_rng(3)
    series = 100.0 + rng.normal(0, 5.0, (6, 6, 6, 8))
    series[0, :, :, :] = np.nan
    r = motion_check(asl_4d=series)
    assert r.verdict == Verdict.INFO
    assert "non-finite voxels excluded" in r.reason
    assert r.metric["nonfinite_voxel_fraction"] > 0


def test_motion_all_nan_series_honest_unknown():
    r = motion_check(asl_4d=np.full((4, 4, 4, 6), np.nan))
    assert r.verdict == Verdict.UNKNOWN
    # "insufficient data" would be false - the data is present but unusable
    assert "non-finite" in r.reason
