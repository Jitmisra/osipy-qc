"""
Module 6 — M0 calibration.

M0 is the denominator in CBF = (control - label)/M0 x constants, so any M0 error
scales every voxel.

6.1 present + type     — separate / included / absent (absent -> WARN, derive).
6.2 TR >= 5 s          — short TR underestimates M0; WARN + a correction factor, never a hard fail.
6.3 acquired WITHOUT BS — BS on the M0 crushes tissue signal -> FAIL (a flag).
6.5 geometry match     — M0 on the same voxel grid as the ASL (else resampling needed).
"""

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict


@register_qc_check("6.1.m0_present", stream="A", required=True)
def m0_present_check(m0_type: str | None = None, detected: dict | None = None,
                     **_) -> CheckResult:
    """PASS if an M0 reference exists; WARN if absent (calibration must derive one)."""
    if isinstance(detected, dict) and detected.get("any_data") is False:
        return CheckResult("6.1.m0_present", Verdict.UNKNOWN,
                           reason="no imaging files were found - an absent M0 is a finding "
                                  "about a dataset, and there is no dataset here")
    if m0_type is None:
        return CheckResult("6.1.m0_present", Verdict.UNKNOWN, reason="M0 type not determined")
    if m0_type in ("separate", "included"):
        return CheckResult("6.1.m0_present", Verdict.PASS,
                           metric={"m0_type": m0_type}, reason=f"M0 present ({m0_type})")
    return CheckResult("6.1.m0_present", Verdict.WARN, metric={"m0_type": m0_type},
                       reason="no M0 - calibration falls back to a derived/default M0")


@register_qc_check("6.2.m0_tr", stream="A", required=True)
def m0_tr_check(m0_tr_s: float | None = None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """TR >= 5 s recovers M0 fully. Below that, flag (WARN) and report the
    T1-correction factor 1/(1 - exp(-TR/T1)); never a hard fail."""
    if m0_tr_s is None:
        return CheckResult("6.2.m0_tr", Verdict.UNKNOWN, reason="M0 TR unknown")
    if m0_tr_s >= cfg.m0_tr_min_s:
        return CheckResult("6.2.m0_tr", Verdict.PASS,
                           metric={"tr_seconds": m0_tr_s, "correction_factor": 1.0},
                           reason=f"TR {m0_tr_s:.1f}s >= {cfg.m0_tr_min_s:.0f}s (no correction needed)")
    factor = 1.0 / (1.0 - np.exp(-m0_tr_s / cfg.t1_tissue_s))
    return CheckResult("6.2.m0_tr", Verdict.WARN,
                       metric={"tr_seconds": m0_tr_s, "correction_factor": round(float(factor), 4)},
                       reason=f"TR {m0_tr_s:.1f}s < {cfg.m0_tr_min_s:.0f}s - correct by x{factor:.3f}")


@register_qc_check("6.3.m0_no_bs", stream="A", required=True)
def m0_no_bs_check(m0_background_suppression: bool | None = None,
                   m0_sidecar: dict | None = None, **_) -> CheckResult:
    """The M0 must be acquired WITHOUT background suppression (BS crushes the tissue
    signal M0 is meant to measure -> CBF over-estimated).

    Deliberately does NOT fall back to the ASL sidecar's BackgroundSuppression:
    that field describes the ASL pairs, where BS is expected ON - the exact
    opposite of the M0 rule. On the OSIPI Challenge data the ASL sidecar says
    true; inheriting it would FAIL a perfectly ordinary M0. UNKNOWN is the honest
    verdict when the M0's own metadata is silent; only the reason says how silent.
    """
    if m0_background_suppression is None:
        if m0_sidecar:
            return CheckResult(
                "6.3.m0_no_bs", Verdict.UNKNOWN,
                reason="M0 sidecar present but does not state BackgroundSuppression - "
                       "not inherited from the ASL sidecar (opposite rule: BS must be "
                       "OFF for M0, is expected ON for ASL)")
        return CheckResult("6.3.m0_no_bs", Verdict.UNKNOWN,
                           reason="M0 BS status unknown (no M0 metadata)")
    if m0_background_suppression:
        return CheckResult("6.3.m0_no_bs", Verdict.FAIL,
                           reason="M0 acquired WITH background suppression - invalid calibration")
    return CheckResult("6.3.m0_no_bs", Verdict.PASS, reason="M0 acquired without background suppression")


@register_qc_check("6.5.m0_geometry", stream="A", required=False)
def m0_geometry_check(m0_shape=None, asl_shape=None, **_) -> CheckResult:
    """M0 should sit on the same voxel grid as the ASL; otherwise resampling
    (with its interpolation artifacts) is required first."""
    if m0_shape is None or asl_shape is None:
        return CheckResult("6.5.m0_geometry", Verdict.UNKNOWN, reason="missing M0 or ASL geometry")
    m0_spatial = tuple(m0_shape[:3])
    asl_spatial = tuple(asl_shape[:3])
    if m0_spatial == asl_spatial:
        return CheckResult("6.5.m0_geometry", Verdict.PASS,
                           metric={"m0_shape": m0_spatial, "asl_shape": asl_spatial},
                           reason=f"M0 {m0_spatial} matches ASL grid")
    return CheckResult("6.5.m0_geometry", Verdict.WARN,
                       metric={"m0_shape": m0_spatial, "asl_shape": asl_spatial},
                       reason=f"M0 {m0_spatial} != ASL {asl_spatial} - resampling required")
