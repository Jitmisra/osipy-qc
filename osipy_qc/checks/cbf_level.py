"""
Module 3 — CBF level & contrast.

3.1 mean/median GM & WM CBF   — are the absolute numbers physiological?
3.2 GM/WM ratio               — is GM brighter than WM (scale-free)?
3.3 negative-GM fraction      — same `p` the QEI consumes, reported on its own.

All pure NumPy. Masks come from probability maps thresholded at cfg.tissue_thresh.
"""

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict
from ..utils.masks import clean_nonfinite, threshold_prob


def _gm_wm_values(cbf, gm, wm, cfg):
    cbf = clean_nonfinite(cbf)
    gm_vals = cbf[threshold_prob(gm, cfg.tissue_thresh)]
    wm_vals = cbf[threshold_prob(wm, cfg.tissue_thresh)]
    return gm_vals, wm_vals


@register_qc_check("3.1.cbf_level", stream="B", required=True)
def cbf_level_check(cbf=None, gm=None, wm=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Absolute GM & WM CBF level. Overall = worst of the two tissues."""
    if cbf is None or gm is None or wm is None:
        return CheckResult("3.1.cbf_level", Verdict.UNKNOWN, reason="needs CBF + GM/WM masks")
    gm_vals, wm_vals = _gm_wm_values(cbf, gm, wm, cfg)
    if gm_vals.size == 0 or wm_vals.size == 0:
        return CheckResult("3.1.cbf_level", Verdict.UNKNOWN, reason="empty GM or WM mask")

    mean_gm, mean_wm = float(np.mean(gm_vals)), float(np.mean(wm_vals))

    def score(mean, lo, hi, flo, fhi):
        if mean <= flo or mean > fhi:
            return Verdict.FAIL
        if lo <= mean <= hi:
            return Verdict.PASS
        return Verdict.WARN

    v_gm = score(mean_gm, cfg.gm_cbf_lo, cfg.gm_cbf_hi, cfg.gm_cbf_fail_lo, cfg.gm_cbf_fail_hi)
    v_wm = score(mean_wm, cfg.wm_cbf_lo, cfg.wm_cbf_hi, cfg.wm_cbf_fail_lo, cfg.wm_cbf_fail_hi)
    overall = max([v_gm, v_wm], key=lambda v: [Verdict.PASS, Verdict.WARN, Verdict.FAIL].index(v))

    metric = {
        "mean_gm_cbf": round(mean_gm, 2), "median_gm_cbf": round(float(np.median(gm_vals)), 2),
        "mean_wm_cbf": round(mean_wm, 2), "median_wm_cbf": round(float(np.median(wm_vals)), 2),
        "gm_verdict": v_gm.value, "wm_verdict": v_wm.value,
    }
    return CheckResult("3.1.cbf_level", overall, metric=metric,
                       reason=f"GM {mean_gm:.1f} ({v_gm.value}), WM {mean_wm:.1f} ({v_wm.value})")


@register_qc_check("3.2.gm_wm_ratio", stream="B", required=True)
def gm_wm_ratio_check(cbf=None, gm=None, wm=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """GM/WM ratio — scale-free contrast (a bad M0 cancels out)."""
    if cbf is None or gm is None or wm is None:
        return CheckResult("3.2.gm_wm_ratio", Verdict.UNKNOWN, reason="needs CBF + GM/WM masks")
    gm_vals, wm_vals = _gm_wm_values(cbf, gm, wm, cfg)
    if gm_vals.size == 0 or wm_vals.size == 0:
        return CheckResult("3.2.gm_wm_ratio", Verdict.UNKNOWN, reason="empty GM or WM mask")
    mean_gm = float(np.mean(gm_vals))
    mean_wm = float(np.mean(wm_vals))
    # a meaningful ratio needs positive tissue means; two negatives would otherwise
    # divide to a spurious positive ratio and a false PASS.
    if mean_wm <= 0 or mean_gm <= 0:
        return CheckResult("3.2.gm_wm_ratio", Verdict.FAIL,
                           metric={"mean_gm_cbf": round(mean_gm, 2), "mean_wm_cbf": round(mean_wm, 2)},
                           reason=f"non-positive tissue CBF (GM {mean_gm:.1f}, WM {mean_wm:.1f}) - ratio invalid")
    ratio = mean_gm / mean_wm

    if ratio > cfg.ratio_pass:
        v, why = Verdict.PASS, f"GM/WM ratio {ratio:.2f} (healthy)"
    elif ratio > cfg.ratio_min:
        v, why = Verdict.WARN, f"GM/WM ratio {ratio:.2f} (weak contrast)"
    else:
        v, why = Verdict.FAIL, f"GM/WM ratio {ratio:.2f} (no contrast / inverted)"
    return CheckResult("3.2.gm_wm_ratio", v, metric={"gm_wm_ratio": round(ratio, 3)}, reason=why)


@register_qc_check("3.3.negative_gm", stream="B", required=True)
def negative_gm_check(cbf=None, gm=None, cfg: QCConfig = QCConfig(),
                      is_raw_deltam: bool = False, **_) -> CheckResult:
    """Fraction of GM voxels with physically-impossible negative CBF.
    N/A on a raw pre-subtracted deltaM (negatives are legitimate there)."""
    if is_raw_deltam:
        return CheckResult("3.3.negative_gm", Verdict.NA,
                           reason="raw pre-subtracted deltaM, not quantified CBF")
    if cbf is None or gm is None:
        return CheckResult("3.3.negative_gm", Verdict.UNKNOWN, reason="needs CBF + GM mask")
    gm_vals = clean_nonfinite(cbf)[threshold_prob(gm, cfg.tissue_thresh)]
    n_gm = int(gm_vals.size)
    if n_gm == 0:
        return CheckResult("3.3.negative_gm", Verdict.UNKNOWN, reason="empty GM mask")
    n_neg = int(np.count_nonzero(gm_vals < 0))
    p = n_neg / n_gm
    if p <= cfg.neg_gm_warn:
        v, why = Verdict.PASS, f"{p*100:.1f}% negative GM voxels"
    elif p <= cfg.neg_gm_fail:
        v, why = Verdict.WARN, f"{p*100:.1f}% negative GM voxels (elevated)"
    else:
        v, why = Verdict.FAIL, f"{p*100:.1f}% negative GM voxels (swap/failure)"
    return CheckResult("3.3.negative_gm", v,
                       metric={"negGM_fraction": round(p, 4), "n_negative": n_neg, "n_gm": n_gm},
                       reason=why)
