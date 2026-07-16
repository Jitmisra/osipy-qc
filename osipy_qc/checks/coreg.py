"""
Module 4 — Co-registration quality.

Dice and Jaccard *evaluate* an existing registration (they are not registration
methods): how well does the CBF/ASL brain mask overlap the structural (T1) mask,
on the same voxel grid? Pure NumPy set-overlap.

Note: ASLPrep's live code computes Dice + Pearson-on-masks + Szymkiewicz-Simpson
overlap (no Jaccard); we report Dice + Jaccard as scipy-free standards
(J = D/(2-D)).
"""

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict
from ..utils.mathops import dice, jaccard
from ..utils.masks import coverage_fraction


@register_qc_check("4.1.coregistration", stream="B", required=True)
def coregistration_check(asl_mask=None, struct_mask=None,
                         cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Grade alignment of the ASL brain mask vs the structural brain mask."""
    if asl_mask is None or struct_mask is None:
        return CheckResult("4.1.coregistration", Verdict.UNKNOWN,
                           reason="needs both an ASL mask and a structural (T1) mask")
    a = np.asarray(asl_mask, dtype=bool)
    b = np.asarray(struct_mask, dtype=bool)
    if a.shape != b.shape:
        return CheckResult("4.1.coregistration", Verdict.UNKNOWN,
                           reason=f"masks on different grids {a.shape} vs {b.shape}")
    if not a.any() or not b.any():
        return CheckResult("4.1.coregistration", Verdict.UNKNOWN,
                           reason="one or both masks are empty - cannot grade alignment")
    d = dice(a, b)
    j = jaccard(a, b)
    inter = int(np.count_nonzero(a & b))

    # Both Dice cutoffs are uncalibrated: no published Dice threshold exists for
    # registration QC (0.7 comes from Zou 2004 on SEGMENTATION), and Birn 2023
    # shows Dice is not even monotonic with registration quality. So this check
    # never hard-FAILs unless cfg.strict, and says so in the metric.
    if d >= cfg.dice_pass:
        v, why = Verdict.PASS, f"Dice {d:.3f} (well aligned)"
    elif d >= cfg.dice_warn:
        v, why = Verdict.WARN, f"Dice {d:.3f} (mild misalignment)"
    else:
        v = Verdict.FAIL if cfg.strict else Verdict.WARN
        why = f"Dice {d:.3f} (misregistration)"

    metric = {
        "dice": round(d, 4), "jaccard": round(j, 4),
        "n_asl_voxels": int(np.count_nonzero(a)),
        "n_struct_voxels": int(np.count_nonzero(b)),
        "n_intersection": inter,
        "provenance": "uncalibrated - no published Dice cutoff exists for registration QC",
    }
    return CheckResult("4.1.coregistration", v, metric=metric, reason=why)


@register_qc_check("4.2.coverage", stream="B", required=True)
def coverage_check(cbf=None, gm=None, wm=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """How much of the tissue ROI the CBF map actually covers.

    Guards a silent, high-impact failure: tissue masks come from the T1, whose
    field of view is larger than the ASL's. Where the ASL does not reach —
    classically the cerebellum — the ROI still contains voxels, but the CBF map
    holds 0 there. Averaging those structural zeros into mean GM CBF drags the
    level down and corrupts the GM/WM ratio, so a scan can look hypoperfused
    purely because of a cropped FOV.

    The other checks already exclude those voxels (utils.masks.covered_tissue_mask).
    This check exists so the exclusion is *visible* rather than silent: if a
    large share of the ROI had to be dropped, that is a finding in its own right
    and the level numbers should be read with care.

    Both thresholds are uncalibrated engineering defaults.
    """
    if cbf is None or gm is None:
        return CheckResult("4.2.coverage", Verdict.UNKNOWN, reason="needs CBF + GM mask")

    gm_cov = coverage_fraction(cbf, gm, cfg.tissue_thresh)
    metric = {"gm_coverage": round(gm_cov, 4)}
    worst, worst_name = gm_cov, "GM"
    if wm is not None:
        wm_cov = coverage_fraction(cbf, wm, cfg.tissue_thresh)
        metric["wm_coverage"] = round(wm_cov, 4)
        if wm_cov < worst:
            worst, worst_name = wm_cov, "WM"

    if worst == 0.0:
        return CheckResult("4.2.coverage", Verdict.UNKNOWN, metric=metric,
                           reason="empty tissue mask - cannot assess coverage")
    metric["provenance"] = "uncalibrated - our defence against T1/ASL FOV mismatch"

    if worst >= cfg.coverage_warn:
        return CheckResult("4.2.coverage", Verdict.PASS, metric=metric,
                           reason=f"{worst*100:.1f}% of the {worst_name} ROI is covered by CBF data")
    if worst >= cfg.coverage_fail:
        return CheckResult("4.2.coverage", Verdict.WARN, metric=metric,
                           reason=f"only {worst*100:.1f}% of the {worst_name} ROI is covered - "
                                  "CBF level/ratio may be biased by a cropped FOV")
    v = Verdict.FAIL if cfg.strict else Verdict.WARN
    return CheckResult("4.2.coverage", v, metric=metric,
                       reason=f"only {worst*100:.1f}% of the {worst_name} ROI is covered - "
                              "large FOV mismatch (cerebellum outside the ASL slab?)")
