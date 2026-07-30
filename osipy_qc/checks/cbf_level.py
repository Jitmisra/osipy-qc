"""
Module 3 — CBF level & contrast.

3.1 mean/median GM & WM CBF   — are the absolute numbers physiological?
3.2 GM/WM ratio               — is GM brighter than WM (scale-free)?
3.3 negative-GM fraction      — same `p` the QEI consumes, reported on its own.
3.4 deep-GM / cortical-GM     — neonatal only; deep GM should EXCEED cortical GM.

All pure NumPy. Masks are coverage-aware (see utils.masks.covered_tissue_mask):
tissue voxels the CBF map does not actually cover are excluded rather than
averaged in as zeros.

Bands are population-dependent — a newborn's normal GM CBF (~16) would FAIL the
adult 40-100 band, and an adult's would FAIL the neonatal one. v1.0 ships two
profiles (adult, neonate); use `config.for_population(...)`, see config.POPULATIONS.
"""

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict
from ..utils.masks import (brain_mask_fallback, clean_nonfinite, coverage_fraction,
                           covered_tissue_mask)


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
    # A FAIL here is only ever produced by the uncalibrated fail bounds
    # (the 40-100 / 15.8-27.5 PASS bands are published, but crossing them is a
    # WARN); so a FAIL is provisional, a published-band WARN is not.
    return CheckResult("3.1.cbf_level", overall, metric=metric,
                       reason=f"GM {mean_gm:.1f} ({v_gm.value}), WM {mean_wm:.1f} ({v_wm.value}) "
                              f"[{cfg.population} bands]",
                       provisional=overall is Verdict.FAIL)


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
    # neg_gm_warn / neg_gm_fail are both uncalibrated (Dolui uses negative-GM as a
    # continuous QEI term, never a cutoff), so any non-PASS here is provisional.
    return CheckResult("3.3.negative_gm", v,
                       metric={"negGM_fraction": round(p, 4), "n_negative": n_neg, "n_gm": n_gm},
                       reason=why, provisional=v is not Verdict.PASS)


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


@register_qc_check("3.5.brain_cbf", stream="B", required=False)
def brain_cbf_check(cbf=None, gm=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Whole-brain CBF over a self-derived mask. The only magnitude check that
    runs when no tissue maps were supplied.

    WHY THIS DOES NOT GRADE AGAINST A NORMAL RANGE
    ----------------------------------------------
    A mentor review asked for "an overall check of CBF brain values", because a
    CBF map uploaded without tissue maps had nothing gradeable at all. The
    obvious implementation - compare the brain mean against a normal band, the
    way 3.1 compares GM - is not defensible, for two independent reasons.

    1. No published bound exists. Every magnitude bound in the ASL literature is
       stated for GREY MATTER (White Paper p.17: "gray matter CBF values from
       40-100 ml/min/100ml can be normal") or for a RATIO (ASLPrep excludes
       GM/WM < 1). A literature sweep found no paper, guideline or pipeline
       stating an accept/reject band for whole-brain mean CBF. The published
       "global" figures are also not this quantity: they are computed inside
       parenchyma masks with CSF excluded, so a brain-mask mean that includes
       ventricles reads systematically lower and cannot be compared to them.

    2. The quantity is not stable. `brain_mask_fallback` thresholds at a
       percentile, and on synthetic data the resulting mean moves 41 -> 60
       mL/100g/min as that percentile goes 25 -> 75 - a spread of ~18, or nearly
       a third of the width of any plausible band. A threshold on a number that
       an internal knob moves that far is not measuring the scan.

    So this check grades only what survives both objections: whether the map can
    be a quantified CBF map AT ALL. On the same synthetic sweep the garbage case
    is negative at every percentile (-32 to -15) while the clean case is positive
    at every percentile, so the sign of the brain mean is robust to the mask in a
    way its value is not. Gross mis-scaling is exactly the failure the White Paper
    attributes to "a global reduction in labeling efficiency, or that the PD scan
    used for normalization was incorrectly acquired or scaled" (p.17).

    The mean itself is always reported, so a reader can judge it against their own
    population - which is the honest division of labour when the field has not
    published a bound.

    Skipped (N/A) when tissue maps are present: 3.1 then grades GM and WM against
    published bands, which is strictly better evidence, and two magnitude verdicts
    on the same data would double-count one problem in the overall verdict.

    KNOWN LIMITATION - it cannot tell CBF from arbitrary units
    ----------------------------------------------------------
    Nothing in a NIfTI states its units, so a raw deltaM or an un-quantified
    perfusion-weighted map in scanner units is indistinguishable from CBF here.
    Measured on the three real test datasets, raw ASL series give brain means of
    191, 482 and 905 - the last two FAIL as implausible, which is the right answer
    for the wrong reason, and the first PASSES at 191 while being no more a CBF map
    than the others.

    So a PASS from this check means "nothing about the magnitude rules out CBF",
    NOT "this is quantified CBF". The check that establishes what a file actually
    is, is 8.2.data_type, and the intended pipeline is that quantification happens
    upstream. Do not tighten the upper bound to catch this: the overlap between
    plausible CBF and plausible arbitrary units is real, and a tighter bound would
    start failing genuine high-perfusion scans without reliably catching deltaM.
    """
    if cbf is None:
        return CheckResult("3.5.brain_cbf", Verdict.UNKNOWN, reason="needs a CBF map")
    if gm is not None:
        return CheckResult("3.5.brain_cbf", Verdict.NA,
                           reason="tissue maps supplied - 3.1 grades GM/WM against "
                                  "published bands instead")

    cbf_clean = clean_nonfinite(cbf)
    brain = brain_mask_fallback(cbf_clean, cfg.brain_mask_percentile)
    vals = cbf_clean[brain]
    if vals.size == 0:
        return CheckResult("3.5.brain_cbf", Verdict.UNKNOWN,
                           reason="no non-zero voxels - cannot derive a brain mask")

    mean = float(np.mean(vals))
    median = float(np.median(vals))
    neg_frac = float(np.mean(vals < 0))
    metric = {
        "mean_brain_cbf": round(mean, 2),
        "median_brain_cbf": round(median, 2),
        "negative_fraction": round(neg_frac, 4),
        "n_voxels": int(vals.size),
        # Named so nobody mistakes this for a tissue-segmented measurement, and
        # so the percentile that produced it travels with the number.
        "mask": f"self-derived (magnitude > p{cfg.brain_mask_percentile:g} of non-zero)",
        "implausible_outside": [cfg.brain_cbf_absurd_lo, cfg.brain_cbf_absurd_hi],
        "graded_against": "gross implausibility only - no published whole-brain band exists",
    }

    if mean <= cfg.brain_cbf_absurd_lo:
        return CheckResult("3.5.brain_cbf", Verdict.FAIL, metric=metric, provisional=True,
                           reason=f"whole-brain mean CBF {mean:.1f} is not positive - the map "
                                  "cannot be quantified CBF (check labelling efficiency and "
                                  "the M0/PD scaling)")
    if mean >= cfg.brain_cbf_absurd_hi:
        return CheckResult("3.5.brain_cbf", Verdict.FAIL, metric=metric, provisional=True,
                           reason=f"whole-brain mean CBF {mean:.1f} exceeds {cfg.brain_cbf_absurd_hi:g} "
                                  "- implies a scaling or unit error, not perfusion")
    return CheckResult("3.5.brain_cbf", Verdict.PASS, metric=metric,
                       reason=f"whole-brain mean CBF {mean:.1f} mL/100g/min over a self-derived "
                              "mask - plausible as CBF; supply tissue maps for a graded level check")
