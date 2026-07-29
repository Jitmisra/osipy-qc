"""
Module 7 — Motion.

7.1 Framewise Displacement (FWD, Power 2012) + DVARS.

FWD per consecutive frame-pair, motion params [tx, ty, tz, rx, ry, rz]
(translations mm, rotations rad):

    D_t   = params[t] - params[t-1]
    FWD_t = |Dtx| + |Dty| + |Dtz| + R*(|Drx| + |Dry| + |Drz|),   R = 50 mm

DVARS = RMS over voxels of the temporal difference of the (optionally mean-scaled)
4D signal, per frame-pair.

Which threshold grades which statistic
--------------------------------------
Both published motion numbers are real, but they describe DIFFERENT statistics
and were previously wired to the wrong ones:

    fd_mean_fail_mm  = 1.0   Adebimpe 2022 (ASLPrep): "we excluded participants
                             with MEAN frame-wise displacement greater than 1 mm."
                             -> applied to the MEAN. This is the citable verdict.

    fd_frame_censor_mm = 0.5 Power 2012: a PER-FRAME censoring threshold, chosen
                             (in Power's own words) by studying plots of dozens of
                             healthy adults. It is not a subject-level mean cutoff,
                             so here it only COUNTS how many frames a censoring
                             pass would drop.
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
        # A NaN row poisons the two differences either side of it, and those
        # frame pairs are genuinely unmeasurable — not zero motion. Filling them
        # with 0.0 would fabricate stillness, and leaving them in made every
        # comparison False, so a scan with a NaN row graded PASS on "mean FWD
        # nan mm". BIDS writes n/a and MCFLIRT can emit NaN, so this is ordinary
        # input, and grading it PASS is the exact failure a QC tool exists to
        # prevent.
        n_bad = int(np.count_nonzero(~np.isfinite(fwd))) if fwd.size else 0
        fwd = fwd[np.isfinite(fwd)] if fwd.size else fwd
        if n_bad and not fwd.size:
            return CheckResult(
                "7.1.motion", Verdict.UNKNOWN, metric={"n_nonfinite_frame_pairs": n_bad},
                reason="motion parameters are entirely non-finite (NaN/Inf) - "
                       "motion cannot be graded")
        if fwd.size:
            mean_fwd, max_fwd = float(fwd.mean()), float(fwd.max())
            # Frames a per-frame censoring rule would drop (Power 2012's 0.5 mm).
            n_censor = int(np.count_nonzero(fwd > cfg.fd_frame_censor_mm))
            censor_frac = n_censor / fwd.size
            metric.update({
                "mean_fwd_mm": round(mean_fwd, 4),
                "max_fwd_mm": round(max_fwd, 4),
                "n_frames": int(fwd.size + 1),
                "n_frames_over_censor": n_censor,
                "censored_fraction": round(censor_frac, 4),
                "censor_threshold_mm": cfg.fd_frame_censor_mm,
            })
            if n_bad:
                metric["n_nonfinite_frame_pairs"] = n_bad
            # Each threshold is now applied to the statistic its source defines:
            #   mean FWD > 1.0 mm  -> Adebimpe 2022's subject-exclusion rule (published).
            #   per-frame > 0.5 mm -> Power 2012's censoring rule (published) - counted,
            #                         not used as a subject-level mean cutoff.
            if mean_fwd > cfg.fd_mean_fail_mm:
                verdict = Verdict.FAIL if cfg.strict else Verdict.WARN
                reason = (f"mean FWD {mean_fwd:.3f} mm > {cfg.fd_mean_fail_mm} mm "
                          "(ASLPrep excludes these)")
            elif censor_frac > cfg.fd_censor_frac_warn:
                verdict = Verdict.WARN
                reason = (f"mean FWD {mean_fwd:.3f} mm, but {censor_frac*100:.0f}% of frames "
                          f"exceed the {cfg.fd_frame_censor_mm} mm censoring line")
            elif n_bad:
                # measurable frames look fine, but some could not be measured at
                # all, so the scan cannot be called clean
                verdict = Verdict.WARN
                reason = (f"mean FWD {mean_fwd:.3f} mm over the measurable frames, but "
                          f"{n_bad} frame pair(s) are non-finite (failed or NaN volumes)")
            else:
                verdict = Verdict.PASS
                reason = f"mean FWD {mean_fwd:.3f} mm ({n_censor}/{fwd.size} frames over censor line)"

    if asl_4d is not None:
        dv = dvars(asl_4d, brain)
        dv = dv[np.isfinite(dv)] if dv.size else dv
        if dv.size:
            metric["mean_dvars"] = round(float(dv.mean()), 4)
            if verdict == Verdict.UNKNOWN:   # no motion params -> report DVARS, no hard verdict
                verdict, reason = Verdict.INFO, f"mean DVARS {dv.mean():.2f} (no motion params for FWD)"

    return CheckResult("7.1.motion", verdict, metric=metric, reason=reason)
