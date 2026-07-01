"""Known-answer tests for Module 2 (sCoV, SNR, histogram)."""

import numpy as np

from osipy_qc.checks.noise import histogram_check, snr_check, spatial_cov_check
from osipy_qc.core import Verdict


def _gm_cbf(values):
    n = len(values)
    cbf = np.array(values, dtype=float).reshape(n, 1, 1)
    gm = np.ones((n, 1, 1))
    return cbf, gm


def test_scov_clean_pass():
    cbf, gm = _gm_cbf([50, 55, 60, 52, 58])      # mean 55, std sqrt(13.6)=3.6878
    r = spatial_cov_check(cbf=cbf, gm=gm)
    assert abs(r.metric["sCoV"] - 3.687817782917155 / 55.0) < 1e-3
    assert r.metric["tier"] == "CBF-contrast"
    assert r.verdict == Verdict.PASS


def test_scov_macrovascular_fail():
    cbf, gm = _gm_cbf([120, 15, 110, 10, 45])    # mean 60, std 46.5833 -> sCoV 0.7764
    r = spatial_cov_check(cbf=cbf, gm=gm)
    assert abs(r.metric["sCoV"] - 0.7763876) < 1e-3
    assert r.verdict == Verdict.FAIL
    assert r.metric["tier"] == "macrovascular-dominant"


def test_snr_is_reciprocal_of_scov():
    cbf, gm = _gm_cbf([33, 77])                  # mean 55, std 22 -> SNR 2.5
    r = snr_check(cbf=cbf, gm=gm)
    assert abs(r.metric["spatial_snr"] - 2.5) < 1e-6
    assert r.verdict == Verdict.PASS


def test_tsnr_from_4d():
    # constant-in-time signal -> infinite tSNR; add tiny noise so std>0 and tSNR high
    rng = np.random.default_rng(0)
    series = 100.0 + rng.normal(0, 1.0, (4, 4, 4, 20))
    r = snr_check(asl_4d=series)
    assert "tsnr" in r.metric and r.metric["tsnr"] > 10


def test_histogram_healthy_pass():
    cbf, gm = _gm_cbf([48, 52, 55, 56, 58, 60, 71])
    r = histogram_check(cbf=cbf, gm=gm)
    assert abs(r.metric["skewness"] - 0.831776) < 1e-2
    assert r.metric["neg_fraction"] == 0.0
    assert r.verdict == Verdict.PASS


def test_histogram_left_skew_warn():
    cbf, gm = _gm_cbf([-30, 1, 2, 3, 4])         # strong left skew + 20% negatives
    r = histogram_check(cbf=cbf, gm=gm)
    assert r.metric["skewness"] < -1.0
    assert r.metric["neg_fraction"] == 0.2
    assert r.verdict == Verdict.WARN
