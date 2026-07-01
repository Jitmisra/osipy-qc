"""
Module 1 — QEI engine (the anchor metric).

Faithful to the live ASLPrep `compute_qei` (Dolui 2024). One 0-1 number built
from three things a radiologist checks by eye, combined by a geometric mean so a
single disaster collapses the score:

    spCBF = 2.5*GM_prob + 1.0*WM_prob              # structural template (CSF excluded)
    rho   = clip(Pearson(cbf, spCBF), 0, inf)      # over (cbf!=0) & finite voxels
    DI    = pooled_within_tissue_var / |mean_gm|   # dispersion index
    p     = #(GM voxels cbf<0) / #(GM voxels)       # negative-GM fraction

    f1 = 1 - exp(-a * rho^b)
    f2 = exp(-c * DI^d)
    f3 = exp(-e * p^f)
    QEI = (f1 * f2 * f3)^(1/3)

The CBF map is smoothed at 5 mm FWHM *here* (not upstream) to avoid double-smoothing.
"""

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict
from ..utils.mathops import geometric_mean, pearson, pooled_within_tissue_variance
from ..utils.masks import clean_nonfinite, threshold_prob
from ..utils.smoothing import smooth_fwhm


def compute_qei(cbf, gm, wm, csf, cfg: QCConfig = QCConfig(),
                voxel_mm=(1.0, 1.0, 1.0), smooth: bool = True) -> dict:
    """Compute the classical QEI and its three components. Returns a metric dict."""
    cbf = clean_nonfinite(cbf)
    gm = np.asarray(gm, dtype=float)
    wm = np.asarray(wm, dtype=float)
    csf = np.asarray(csf, dtype=float)

    if smooth:
        cbf = smooth_fwhm(cbf, cfg.smooth_fwhm_mm, voxel_mm)

    # --- structural similarity (rho) ---
    spcbf = 2.5 * gm + 1.0 * wm
    mask = (cbf != 0) & np.isfinite(cbf) & np.isfinite(spcbf)
    rho = pearson(cbf[mask], spcbf[mask])
    rho = max(rho, 0.0)  # clip to [0, inf)

    # --- dispersion index (DI) ---
    gm_mask = threshold_prob(gm, cfg.tissue_thresh)
    wm_mask = threshold_prob(wm, cfg.tissue_thresh)
    csf_mask = threshold_prob(csf, cfg.tissue_thresh)
    gm_vals = cbf[gm_mask]
    var_pool = pooled_within_tissue_variance([gm_vals, cbf[wm_mask], cbf[csf_mask]])
    mean_gm = float(np.mean(gm_vals)) if gm_vals.size else 0.0
    di = var_pool / abs(mean_gm) if mean_gm != 0 else 0.0

    # --- negative-GM fraction (p) ---
    n_gm = int(gm_vals.size)
    n_neg = int(np.count_nonzero(gm_vals < 0))
    p = (n_neg / n_gm) if n_gm else 0.0

    # --- combine ---
    f1 = 1.0 - np.exp(-cfg.qei_a * rho ** cfg.qei_b)
    f2 = np.exp(-cfg.qei_c * di ** cfg.qei_d)
    f3 = np.exp(-cfg.qei_e * p ** cfg.qei_f)
    qei = geometric_mean([f1, f2, f3])

    return {
        "qei": round(qei, 4),
        "rho_ss": round(rho, 4),
        "DI": round(di, 4),
        "negGM_fraction": round(p, 4),
        "f1": round(float(f1), 4),
        "f2": round(float(f2), 4),
        "f3": round(float(f3), 4),
        "mean_gm_cbf": round(mean_gm, 2),
        "n_gm": n_gm,
    }


@register_qc_check("1.qei", stream="B", required=True)
def qei_check(cbf=None, gm=None, wm=None, csf=None, cfg: QCConfig = QCConfig(),
              voxel_mm=(1.0, 1.0, 1.0), **_) -> CheckResult:
    """QC check wrapper around `compute_qei` with PASS/WARN/FAIL verdict."""
    if cbf is None or gm is None or wm is None or csf is None:
        return CheckResult("1.qei", Verdict.UNKNOWN,
                           reason="needs CBF map + GM/WM/CSF tissue maps in ASL space")
    m = compute_qei(cbf, gm, wm, csf, cfg, voxel_mm)
    q = m["qei"]
    if q >= cfg.qei_pass:
        v, why = Verdict.PASS, f"QEI {q} (>= {cfg.qei_pass})"
    elif q >= cfg.qei_warn:
        v, why = Verdict.WARN, f"QEI {q} borderline ({cfg.qei_warn}-{cfg.qei_pass})"
    else:
        v, why = Verdict.FAIL, f"QEI {q} (< {cfg.qei_warn})"
    return CheckResult("1.qei", v, metric=m, reason=why)
