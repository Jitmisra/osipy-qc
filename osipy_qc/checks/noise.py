"""
Module 2 — Noise & distribution.

2.1 Spatial CoV in GM   — std/mean over POSITIVE GM CBF; ExploreASL 3-tier classifier.
2.2 SNR (spatial/tSNR)  — mean/std of GM CBF (== 1/sCoV); tSNR for a 4D series.
2.3 Histogram plausibility — skewness + negative fraction of the GM CBF distribution.

Pure NumPy. Population statistics (ddof=0) throughout, matching the literature.
"""

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict
from ..utils.mathops import skewness
from ..utils.masks import clean_nonfinite, threshold_prob


def _positive_gm(cbf, gm, cfg):
    vals = clean_nonfinite(cbf)[threshold_prob(gm, cfg.tissue_thresh)]
    return vals[vals > 0]


@register_qc_check("2.1.spatial_cov", stream="B", required=True)
def spatial_cov_check(cbf=None, gm=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Spatial coefficient of variation of GM CBF (ExploreASL 3-tier)."""
    if cbf is None or gm is None:
        return CheckResult("2.1.spatial_cov", Verdict.UNKNOWN, reason="needs CBF + GM mask")
    vals = _positive_gm(cbf, gm, cfg)
    if vals.size < 2:
        return CheckResult("2.1.spatial_cov", Verdict.UNKNOWN, reason="too few positive GM voxels")
    mean = float(np.mean(vals))
    if mean == 0:
        return CheckResult("2.1.spatial_cov", Verdict.UNKNOWN, reason="mean GM CBF is zero")
    scov = float(np.std(vals)) / mean

    if scov < cfg.scov_warn:
        v, tier = Verdict.PASS, "CBF-contrast"
    elif scov <= cfg.scov_macro:
        v, tier = Verdict.WARN, "vascular-contrast"
    else:
        v, tier = Verdict.FAIL, "macrovascular-dominant"
    return CheckResult("2.1.spatial_cov", v,
                       metric={"sCoV": round(scov, 4), "sCoV_pct": round(scov * 100, 2), "tier": tier},
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
    # spatial SNR == 1/sCoV. There is NO validated literature cutoff for SNR (unlike
    # sCoV 0.67), so SNR never hard-FAILs on its own — it PASSes or WARNs, and the
    # sCoV check carries the FAIL. Mirrors the sCoV tier boundaries.
    if snr > 1.0 / cfg.scov_warn:
        v, why = Verdict.PASS, f"spatial SNR {snr:.2f}"
    else:
        v, why = Verdict.WARN, f"spatial SNR {snr:.2f} (low - see sCoV for the verdict)"
    return CheckResult("2.2.snr", v, metric=metric, reason=why)


@register_qc_check("2.3.histogram", stream="B", required=True)
def histogram_check(cbf=None, gm=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Shape of the GM CBF histogram: skewness + negative fraction. A supporting
    check (no standalone FAIL) — strong LEFT skew + many negatives -> WARN."""
    if cbf is None or gm is None:
        return CheckResult("2.3.histogram", Verdict.UNKNOWN, reason="needs CBF + GM mask")
    vals = clean_nonfinite(cbf)[threshold_prob(gm, cfg.tissue_thresh)]
    if vals.size < 2:
        return CheckResult("2.3.histogram", Verdict.UNKNOWN, reason="too few GM voxels")
    sk = skewness(vals)
    neg_frac = float(np.count_nonzero(vals < 0)) / vals.size

    # QC_DESIGN.md L447: the danger shape is a strong LEFT skew AND many negatives
    # (noise / failed labeling). A healthy distribution is mildly RIGHT-skewed, so a
    # positive skew on its own must NOT be flagged.
    danger = (sk < cfg.skew_lo) and (neg_frac > cfg.hist_neg_warn)
    if danger:
        v, why = Verdict.WARN, f"left-skew {sk:.2f} + {neg_frac*100:.1f}% negative (noise/labeling hint)"
    else:
        v, why = Verdict.PASS, f"skew {sk:.2f}, {neg_frac*100:.1f}% negative (plausible)"
    metric = {
        "skewness": round(sk, 4),
        "neg_fraction": round(neg_frac, 4),
        "mean": round(float(np.mean(vals)), 2),
        "median": round(float(np.median(vals)), 2),
    }
    return CheckResult("2.3.histogram", v, metric=metric, reason=why)
