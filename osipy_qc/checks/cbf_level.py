"""
Module 3 — CBF level & contrast.

3.1 mean/median GM & WM CBF   — are the absolute numbers physiological?
3.2 GM/WM ratio               — is GM brighter than WM (scale-free)?
3.3 negative-GM fraction      — same `p` the QEI consumes, reported on its own.
3.4 deep-GM / cortical-GM     — neonatal only; deep GM should EXCEED cortical GM.

All pure NumPy. Masks are coverage-aware (see utils.masks.covered_tissue_mask):
tissue voxels the CBF map does not actually cover are excluded rather than
averaged in as zeros.

Bands are population-dependent — a child's normal GM CBF (~97) would FAIL an
adult band (40-100 is borderline) and an adult's would FAIL a neonatal one
(~16). Use `config.for_population(...)`; see config.POPULATIONS.
"""

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict
from ..utils.masks import clean_nonfinite, covered_tissue_mask, coverage_fraction


def _gm_wm_values(cbf, gm, wm, cfg):
    cbf_clean = clean_nonfinite(cbf)
    gm_vals = cbf_clean[covered_tissue_mask(cbf, gm, cfg.tissue_thresh)]
    wm_vals = cbf_clean[covered_tissue_mask(cbf, wm, cfg.tissue_thresh)]
    return gm_vals, wm_vals


@register_qc_check("3.1.cbf_level", stream="B", required=True)
def cbf_level_check(cbf=None, gm=None, wm=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Absolute GM & WM CBF level. Overall = worst of the two tissues.

    Only the GM PASS band (40-100, White Paper p.17) and the WM band
    (15.8-27.5, Wu 2013) are published; every FAIL bound here is an uncalibrated
    engineering default, so with `cfg.strict=False` they soften to WARN.
    """
    if cbf is None or gm is None or wm is None:
        return CheckResult("3.1.cbf_level", Verdict.UNKNOWN, reason="needs CBF + GM/WM masks")
    gm_vals, wm_vals = _gm_wm_values(cbf, gm, wm, cfg)
    if gm_vals.size == 0 or wm_vals.size == 0:
        return CheckResult("3.1.cbf_level", Verdict.UNKNOWN,
                           reason="empty GM or WM mask after excluding uncovered voxels")

    mean_gm, mean_wm = float(np.mean(gm_vals)), float(np.mean(wm_vals))

    def score(mean, lo, hi, flo, fhi):
        if mean <= flo or mean > fhi:
            return Verdict.FAIL if cfg.strict else Verdict.WARN
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
        "population": cfg.population,
        "gm_band": [cfg.gm_cbf_lo, cfg.gm_cbf_hi],
        "wm_band": [cfg.wm_cbf_lo, cfg.wm_cbf_hi],
    }
    return CheckResult("3.1.cbf_level", overall, metric=metric,
                       reason=f"GM {mean_gm:.1f} ({v_gm.value}), WM {mean_wm:.1f} ({v_wm.value}) "
                              f"[{cfg.population} bands]")


@register_qc_check("3.2.gm_wm_ratio", stream="B", required=True)
def gm_wm_ratio_check(cbf=None, gm=None, wm=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """GM/WM ratio — scale-free contrast (a bad M0 cancels out).

    Deliberately soft. A low ratio has at least four innocent explanations, so
    this check informs rather than condemns:

      * Pathology. Altered WM CBF is a real finding in small-vessel disease and
        WM hyperintensities, not an image failure.
      * Smoothing. The ratio is smoothing-dependent (Wu 2013: 2.3 unsmoothed ->
        1.8 at 8 mm).
      * Vendor. Smooth GE 3D spiral scans legitimately land at ~1.2-1.3.
      * Age. It declines ~0.79%/year (Parkes 2004).

    Only `ratio_min` = 1 is published (ASLPrep excludes ratio < 1); `ratio_pass`
    is uncalibrated, so it can only ever produce a WARN.
    """
    if cbf is None or gm is None or wm is None:
        return CheckResult("3.2.gm_wm_ratio", Verdict.UNKNOWN, reason="needs CBF + GM/WM masks")
    gm_vals, wm_vals = _gm_wm_values(cbf, gm, wm, cfg)
    if gm_vals.size == 0 or wm_vals.size == 0:
        return CheckResult("3.2.gm_wm_ratio", Verdict.UNKNOWN,
                           reason="empty GM or WM mask after excluding uncovered voxels")
    mean_gm = float(np.mean(gm_vals))
    mean_wm = float(np.mean(wm_vals))
    # A meaningful ratio needs positive tissue means; two negatives would otherwise
    # divide to a spurious positive ratio and a false PASS.
    if mean_wm <= 0 or mean_gm <= 0:
        return CheckResult("3.2.gm_wm_ratio", Verdict.FAIL,
                           metric={"mean_gm_cbf": round(mean_gm, 2), "mean_wm_cbf": round(mean_wm, 2)},
                           reason=f"non-positive tissue CBF (GM {mean_gm:.1f}, WM {mean_wm:.1f}) - ratio invalid")
    ratio = mean_gm / mean_wm

    note = ("normal GM/WM is ~2-4x (Clement 2022, Wu 2013). A low ratio may be pathology "
            "(altered WM CBF), heavy smoothing, a smooth GE 3D spiral readout, or age - "
            "not necessarily a bad scan")
    if ratio > cfg.ratio_pass:
        v, why = Verdict.PASS, f"GM/WM ratio {ratio:.2f} (GM brighter than WM)"
    elif ratio > cfg.ratio_min:
        v, why = Verdict.WARN, f"GM/WM ratio {ratio:.2f} (weak contrast - inspect, may be benign)"
    else:
        # ratio <= 1 is the one published rule (ASLPrep excludes it).
        v = Verdict.FAIL if cfg.strict else Verdict.WARN
        why = f"GM/WM ratio {ratio:.2f} (<= 1; ASLPrep excludes these)"
    return CheckResult("3.2.gm_wm_ratio", v,
                       metric={"gm_wm_ratio": round(ratio, 3), "interpretation": note},
                       reason=why)


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
    gm_vals = clean_nonfinite(cbf)[covered_tissue_mask(cbf, gm, cfg.tissue_thresh)]
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
        v = Verdict.FAIL if cfg.strict else Verdict.WARN
        why = f"{p*100:.1f}% negative GM voxels (swap/failure)"
    return CheckResult("3.3.negative_gm", v,
                       metric={"negGM_fraction": round(p, 4), "n_negative": n_neg, "n_gm": n_gm},
                       reason=why)


@register_qc_check("3.4.deep_gm_ratio", stream="B", required=False)
def deep_gm_ratio_check(cbf=None, deep_gm=None, cortical_gm=None,
                        cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Deep-GM / cortical-GM CBF ratio — a neonatal-specific sanity check.

    In newborns perfusion is HIGHER in deep grey matter (basal ganglia,
    thalami) than in cortical grey matter — the reverse of the adult pattern.
    Miranda 2006 (Pediatr Res 60(3):359-363) measured basal ganglia + thalami at
    39 vs cortical GM 19 mL/100g/min in preterm-at-term-equivalent infants, and
    30 vs 16 in term infants (both p<0.0001) — ratios of 2.05 and 1.88. A
    neonatal map WITHOUT that gradient is suspect.

    N/A outside neonatal populations, where the expectation does not hold.
    Requires separate deep-GM and cortical-GM masks; UNKNOWN without them.

    The band itself is uncalibrated — extrapolated from Miranda's two data
    points, not a published cutoff — so this check never FAILs.
    """
    if not cfg.population.startswith("neonate"):
        return CheckResult("3.4.deep_gm_ratio", Verdict.NA,
                           reason=f"deep>cortical GM is a neonatal pattern; population is "
                                  f"{cfg.population!r}")
    if cbf is None or deep_gm is None or cortical_gm is None:
        return CheckResult("3.4.deep_gm_ratio", Verdict.UNKNOWN,
                           reason="needs CBF + separate deep-GM and cortical-GM masks")

    cbf_clean = clean_nonfinite(cbf)
    deep_vals = cbf_clean[covered_tissue_mask(cbf, deep_gm, cfg.tissue_thresh)]
    cort_vals = cbf_clean[covered_tissue_mask(cbf, cortical_gm, cfg.tissue_thresh)]
    if deep_vals.size == 0 or cort_vals.size == 0:
        return CheckResult("3.4.deep_gm_ratio", Verdict.UNKNOWN,
                           reason="empty deep-GM or cortical-GM mask after coverage masking")

    mean_deep, mean_cort = float(np.mean(deep_vals)), float(np.mean(cort_vals))
    if mean_cort <= 0 or mean_deep <= 0:
        return CheckResult("3.4.deep_gm_ratio", Verdict.WARN,
                           metric={"mean_deep_gm_cbf": round(mean_deep, 2),
                                   "mean_cortical_gm_cbf": round(mean_cort, 2)},
                           reason="non-positive deep or cortical GM CBF - ratio invalid")
    ratio = mean_deep / mean_cort

    metric = {
        "deep_cortical_ratio": round(ratio, 3),
        "mean_deep_gm_cbf": round(mean_deep, 2),
        "mean_cortical_gm_cbf": round(mean_cort, 2),
        "expected_band": [cfg.deep_gm_ratio_lo, cfg.deep_gm_ratio_hi],
        "reference": "Miranda 2006 measured ratios 1.88 (term) and 2.05 (preterm at TEA)",
    }
    if cfg.deep_gm_ratio_lo <= ratio <= cfg.deep_gm_ratio_hi:
        return CheckResult("3.4.deep_gm_ratio", Verdict.PASS, metric=metric,
                           reason=f"deep/cortical GM {ratio:.2f} (expected neonatal gradient)")
    if ratio < 1.0:
        return CheckResult("3.4.deep_gm_ratio", Verdict.WARN, metric=metric,
                           reason=f"deep/cortical GM {ratio:.2f} - cortical EXCEEDS deep GM, "
                                  "which inverts the expected neonatal pattern")
    return CheckResult("3.4.deep_gm_ratio", Verdict.WARN, metric=metric,
                       reason=f"deep/cortical GM {ratio:.2f} outside the expected "
                              f"{cfg.deep_gm_ratio_lo}-{cfg.deep_gm_ratio_hi} band")
