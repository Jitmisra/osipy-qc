"""
All tunable thresholds in one place, so calibration later is a config change,
not a code change. Defaults follow QC_DESIGN.md; values tagged "provisional"
there are the engineering placeholders to be calibrated with real data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QCConfig:
    # --- QEI (Module 1) ---
    qei_pass: float = 0.55          # provisional margin above the paper's ~0.50
    qei_warn: float = 0.50          # paper's validated operating point
    tissue_thresh: float = 0.7      # prob > 0.7 strict, matches live ASLPrep compute_qei
    smooth_fwhm_mm: float = 5.0     # QEI constants fitted on 5 mm-smoothed data

    # QEI curve constants (ASLPrep compute_qei implementation values)
    qei_a: float = 3.0126
    qei_b: float = 2.4419
    qei_c: float = 0.054
    qei_d: float = 0.9272
    qei_e: float = 2.8478
    qei_f: float = 0.5196

    # --- Noise & distribution (Module 2) ---
    scov_warn: float = 0.55         # >= -> vascular-contrast tier (WARN)
    scov_macro: float = 0.67        # > -> macrovascular-dominant (FAIL), ExploreASL cutoff
    skew_lo: float = -0.5           # healthy GM CBF skew band
    skew_hi: float = 1.0
    hist_neg_warn: float = 0.05     # >5% negatives in the histogram check

    # --- CBF level & contrast (Module 3) ---
    gm_cbf_lo: float = 40.0         # PASS band, mL/100g/min (White Paper p.17)
    gm_cbf_hi: float = 100.0
    gm_cbf_fail_lo: float = 10.0    # below -> FAIL
    gm_cbf_fail_hi: float = 150.0   # above -> FAIL
    wm_cbf_lo: float = 15.0         # PASS band (community convention)
    wm_cbf_hi: float = 30.0
    wm_cbf_fail_lo: float = 5.0     # QC_DESIGN.md L510: WM < 5 -> FAIL
    wm_cbf_fail_hi: float = 50.0
    ratio_pass: float = 1.5         # GM should be ~2-2.5x WM
    ratio_min: float = 1.0          # at/below this -> FAIL (no contrast / inverted)
    neg_gm_warn: float = 0.10
    neg_gm_fail: float = 0.20

    # --- Co-registration (Module 4) ---
    dice_pass: float = 0.9
    dice_warn: float = 0.7

    # --- M0 (Module 6) ---
    m0_tr_min_s: float = 5.0        # White Paper
    t1_tissue_s: float = 1.4        # overridable default for TR correction (GM @3T ~1.3-1.4)

    # --- Motion (Module 7) ---
    fd_warn_mm: float = 0.5         # provisional (Power 2012 family)
    fd_fail_mm: float = 1.0
    head_radius_mm: float = 50.0    # rotation -> mm conversion sphere
