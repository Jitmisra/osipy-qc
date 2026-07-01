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

    if d >= cfg.dice_pass:
        v, why = Verdict.PASS, f"Dice {d:.3f} (well aligned)"
    elif d >= cfg.dice_warn:
        v, why = Verdict.WARN, f"Dice {d:.3f} (mild misalignment)"
    else:
        v, why = Verdict.FAIL, f"Dice {d:.3f} (misregistration)"

    metric = {
        "dice": round(d, 4), "jaccard": round(j, 4),
        "n_asl_voxels": int(np.count_nonzero(a)),
        "n_struct_voxels": int(np.count_nonzero(b)),
        "n_intersection": inter,
    }
    return CheckResult("4.1.coregistration", v, metric=metric, reason=why)
