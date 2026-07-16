"""
Module 2 — Noise & distribution.

2.1 Spatial CoV in GM   — std/mean over POSITIVE GM CBF; ExploreASL's 3 tiers.
2.2 SNR (spatial/tSNR)  — mean/std of GM CBF (== 1/sCoV); tSNR for a 4D series.
2.3 Histogram           — skewness + negative fraction, reported as INFO (see below).

Pure NumPy. Population statistics (ddof=0) throughout, matching the literature.

Two corrections worth knowing about
-----------------------------------
1. sCoV tiers were previously inverted. ExploreASL's own bins are
   CBF-contrast < 0.67 < vascular-contrast < 1.0 < artifactual-contrast, and it
   KEEPS the 0.67-1.0 band (excluding it only from CBF statistics). The old code
   FAILed at 0.67, i.e. it failed scans ExploreASL keeps. The exclusion line is 1.0.

2. The histogram check no longer issues a verdict. Its old -0.5 skewness cutoff
   was a generic statistics textbook heuristic (Bulmer 1979: |skew|<0.5 is
   "approximately symmetric") with no ASL/CBF basis whatsoever, and the
   justification once given for it - "ExploreASL-style histogram QC" - was simply
   untrue: neither ExploreASL nor ASLPrep computes CBF skewness at all
   (`grep -i skew` over ExploreASL returns nothing). Rather than re-word an
   unsupported rule, the check now reports the shape as INFO. Negative voxels
   already have their own check (3.3), which at least rests on a physical
   argument (negative perfusion is impossible).
"""

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict
from ..utils.mathops import skewness
from ..utils.masks import clean_nonfinite, covered_tissue_mask


def _positive_gm(cbf, gm, cfg):
    """Positive CBF values inside the covered GM mask."""
    vals = clean_nonfinite(cbf)[covered_tissue_mask(cbf, gm, cfg.tissue_thresh)]
    return vals[vals > 0]


@register_qc_check("2.1.spatial_cov", stream="B", required=True)
def spatial_cov_check(cbf=None, gm=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Spatial coefficient of variation of GM CBF, binned into ExploreASL's tiers.

    Read this as a *transit-time* metric, not a pure image-quality one: sCoV
    correlates with GM arterial transit time at r=0.85 (Mutsaerts 2017), and it
    rises progressively with cognitive decline across control -> MCI -> AD
    (Sci Rep 2021;11:23325). A high sCoV in a patient with cerebrovascular
    disease is therefore often a TRUE FINDING, not a bad scan. Hence
    `cfg.strict=False` demotes the FAIL to WARN for clinical cohorts.
    """
    if cbf is None or gm is None:
        return CheckResult("2.1.spatial_cov", Verdict.UNKNOWN, reason="needs CBF + GM mask")
    vals = _positive_gm(cbf, gm, cfg)
    if vals.size < 2:
        return CheckResult("2.1.spatial_cov", Verdict.UNKNOWN, reason="too few positive GM voxels")
    mean = float(np.mean(vals))
    if mean == 0:
        return CheckResult("2.1.spatial_cov", Verdict.UNKNOWN, reason="mean GM CBF is zero")
    scov = float(np.std(vals)) / mean

    if scov < cfg.scov_vascular:
        v, tier = Verdict.PASS, "CBF-contrast"
    elif scov <= cfg.scov_artifact:
        # ExploreASL keeps this band; it is excluded only from CBF statistics.
        v, tier = Verdict.WARN, "vascular-contrast"
    else:
        v, tier = Verdict.FAIL, "artifactual-contrast"
        if not cfg.strict:
            v = Verdict.WARN

    metric = {
        "sCoV": round(scov, 4),
        "sCoV_pct": round(scov * 100, 2),
        "tier": tier,
        "interpretation": "sCoV tracks arterial transit time (r=0.85, Mutsaerts 2017); "
                          "elevated values may reflect vascular pathology rather than a bad scan",
    }
    return CheckResult("2.1.spatial_cov", v, metric=metric,
                       reason=f"sCoV {scov*100:.1f}% ({tier})")


@register_qc_check("2.2.snr", stream="B", required=True)
def snr_check(cbf=None, gm=None, asl_4d=None, brain=None,
              cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Spatial SNR = mean/std of GM CBF (== 1/sCoV). If a 4D series is given,
    also report temporal SNR (mean-over-time / std-over-time, averaged over the brain)."""
    metric: dict = {}
    if cbf is not None and gm is not None:
        vals = _positive_gm(cbf, gm, cfg)
        if vals.size >= 2 and np.std(vals) > 0:
            metric["spatial_snr"] = round(float(np.mean(vals)) / float(np.std(vals)), 4)

    if asl_4d is not None:
        series = clean_nonfinite(asl_4d)
        if series.ndim == 4 and series.shape[3] > 1:
            tmean = series.mean(axis=3)
            tstd = series.std(axis=3)
            mask = brain if brain is not None else (tmean != 0)
            mask = np.asarray(mask, dtype=bool) & (tstd > 0)
            if np.any(mask):
                metric["tsnr"] = round(float(np.mean(tmean[mask] / tstd[mask])), 4)

    if not metric:
        return CheckResult("2.2.snr", Verdict.UNKNOWN, reason="needs GM CBF map or a 4D series")

    snr = metric.get("spatial_snr")
    if snr is None:
        return CheckResult("2.2.snr", Verdict.INFO, metric=metric, reason="tSNR reported (informational)")
    # spatial SNR == 1/sCoV, so it carries no information the sCoV check does not
    # already carry, and there is no validated SNR cutoff in the literature. It
    # therefore never FAILs on its own - sCoV owns that verdict.
    if snr > 1.0 / cfg.scov_vascular:
        v, why = Verdict.PASS, f"spatial SNR {snr:.2f}"
    else:
        v, why = Verdict.WARN, f"spatial SNR {snr:.2f} (low - see sCoV for the verdict)"
    return CheckResult("2.2.snr", v, metric=metric, reason=why)


@register_qc_check("2.3.histogram", stream="B", required=False)
def histogram_check(cbf=None, gm=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Shape of the GM CBF distribution — reported, never graded.

    INFO by design. No published skewness cutoff exists for CBF (see module
    docstring), so issuing PASS/FAIL here would be inventing a standard. The
    numbers are still useful to a human and to later ROC calibration, so they
    are reported; `3.3.negative_gm` carries the negative-voxel verdict.
    """
    if cbf is None or gm is None:
        return CheckResult("2.3.histogram", Verdict.UNKNOWN, reason="needs CBF + GM mask")
    vals = clean_nonfinite(cbf)[covered_tissue_mask(cbf, gm, cfg.tissue_thresh)]
    if vals.size < 2:
        return CheckResult("2.3.histogram", Verdict.UNKNOWN, reason="too few GM voxels")
    sk = skewness(vals)
    neg_frac = float(np.count_nonzero(vals < 0)) / vals.size

    metric = {
        "skewness": round(sk, 4),
        "neg_fraction": round(neg_frac, 4),
        "mean": round(float(np.mean(vals)), 2),
        "median": round(float(np.median(vals)), 2),
        "provenance": "uncalibrated - no published skewness cutoff exists for CBF; "
                      "reported for inspection and future ROC calibration only",
    }
    return CheckResult("2.3.histogram", Verdict.INFO, metric=metric,
                       reason=f"skew {sk:.2f}, {neg_frac*100:.1f}% negative (informational)")
