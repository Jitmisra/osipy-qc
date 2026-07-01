"""
Module 7 — Motion.

7.1 Framewise Displacement (FWD, Power 2012) + DVARS.

FWD per consecutive frame-pair, motion params [tx, ty, tz, rx, ry, rz]
(translations mm, rotations rad):

    D_t   = params[t] - params[t-1]
    FWD_t = |Dtx| + |Dty| + |Dtz| + R*(|Drx| + |Dry| + |Drz|),   R = 50 mm

DVARS = RMS over voxels of the temporal difference of the (optionally mean-scaled)
4D signal, per frame-pair.
"""

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict


def framewise_displacement(motion_params: np.ndarray, radius_mm: float = 50.0) -> np.ndarray:
    """FWD series (length T-1) from a (T, 6) motion-parameter array."""
    mp = np.asarray(motion_params, dtype=float)
    if mp.ndim != 2 or mp.shape[1] != 6 or mp.shape[0] < 2:
        return np.array([])
    d = np.abs(np.diff(mp, axis=0))           # (T-1, 6)
    trans = d[:, 0:3].sum(axis=1)
    rot = radius_mm * d[:, 3:6].sum(axis=1)
    return trans + rot


def dvars(series_4d: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """DVARS series (length T-1): RMS over voxels of the frame-to-frame difference."""
    arr = np.asarray(series_4d, dtype=float)
    if arr.ndim != 4 or arr.shape[3] < 2:
        return np.array([])
    diff = np.diff(arr, axis=3)               # (..., T-1)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        flat = diff[m]                        # (n_vox, T-1)
        return np.sqrt(np.mean(flat ** 2, axis=0))
    return np.sqrt(np.mean(diff ** 2, axis=(0, 1, 2)))


@register_qc_check("7.1.motion", stream="A", required=True)
def motion_check(motion_params=None, asl_4d=None, brain=None,
                 cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Grade head motion from FWD (and DVARS if a 4D series is supplied)."""
    if motion_params is None and asl_4d is None:
        return CheckResult("7.1.motion", Verdict.UNKNOWN,
                           reason="needs motion parameters or a 4D series to estimate motion")

    metric: dict = {}
    verdict = Verdict.UNKNOWN
    reason = "insufficient data"

    if motion_params is not None:
        fwd = framewise_displacement(motion_params, cfg.head_radius_mm)
        if fwd.size:
            mean_fwd, max_fwd = float(fwd.mean()), float(fwd.max())
            metric.update({"mean_fwd_mm": round(mean_fwd, 4), "max_fwd_mm": round(max_fwd, 4),
                           "n_frames": int(fwd.size + 1)})
            if mean_fwd < 0.2 and max_fwd <= cfg.fd_fail_mm:
                verdict, reason = Verdict.PASS, f"mean FWD {mean_fwd:.3f} mm (low motion)"
            elif mean_fwd <= cfg.fd_warn_mm:
                verdict, reason = Verdict.WARN, f"mean FWD {mean_fwd:.3f} mm (moderate motion)"
            else:
                verdict, reason = Verdict.FAIL, f"mean FWD {mean_fwd:.3f} mm (excessive motion)"

    if asl_4d is not None:
        dv = dvars(asl_4d, brain)
        if dv.size:
            metric["mean_dvars"] = round(float(dv.mean()), 4)
            if verdict == Verdict.UNKNOWN:   # no motion params -> report DVARS, no hard verdict
                verdict, reason = Verdict.INFO, f"mean DVARS {dv.mean():.2f} (no motion params for FWD)"

    return CheckResult("7.1.motion", verdict, metric=metric, reason=reason)
