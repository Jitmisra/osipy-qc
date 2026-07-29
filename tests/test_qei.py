"""
Known-answer + property tests for the QEI engine.

Two kinds of guarantee:
1. An INDEPENDENT reference implementation (written plainly here) must match
   compute_qei byte-for-byte on the same inputs -> the formula is implemented right.
2. Property tests on synthetic data: clean > borderline > garbage, and the
   verdicts come out PASS / (WARN|PASS) / FAIL.
"""

import numpy as np

from osipy_qc.checks.qei import compute_qei, qei_check
from osipy_qc.core import QCConfig, Verdict
from osipy_qc.synth import synthetic_case


def _reference_qei(cbf, gm, wm, csf, cfg):
    """A deliberately naive, independent re-derivation of the QEI math."""
    spcbf = 2.5 * gm + 1.0 * wm
    mask = (cbf != 0) & np.isfinite(cbf) & np.isfinite(spcbf)
    x, y = cbf[mask], spcbf[mask]
    xc, yc = x - x.mean(), y - y.mean()
    rho = max(float((xc * yc).sum() / np.sqrt((xc**2).sum() * (yc**2).sum())), 0.0)

    def tvar(prob):
        v = cbf[prob > cfg.tissue_thresh]
        return v, (v.size, v.var() if v.size > 0 else 0.0)  # ddof=0, ASLPrep-faithful

    gmv, (ng, vg) = tvar(gm)
    _, (nw, vw) = tvar(wm)
    _, (nc, vc) = tvar(csf)
    num = (ng - 1) * vg + (nw - 1) * vw + (nc - 1) * vc
    den = (ng - 1) + (nw - 1) + (nc - 1)
    V = num / den
    mean_gm = gmv.mean()
    DI = V / abs(mean_gm)
    p = np.count_nonzero(gmv < 0) / ng
    f1 = 1 - np.exp(-cfg.qei_a * rho**cfg.qei_b)
    f2 = np.exp(-cfg.qei_c * DI**cfg.qei_d)
    f3 = np.exp(-cfg.qei_e * p**cfg.qei_f)
    return float((f1 * f2 * f3) ** (1 / 3))


def test_qei_matches_independent_reference():
    cfg = QCConfig()
    case = synthetic_case(quality="clean", seed=1)
    # compare WITHOUT smoothing so the reference (which doesn't smooth) matches exactly
    m = compute_qei(case.cbf, case.gm, case.wm, case.csf, cfg,
                    voxel_mm=case.voxel_mm, smooth=False)
    ref = _reference_qei(case.cbf, case.gm, case.wm, case.csf, cfg)
    assert abs(m["qei"] - round(ref, 4)) < 1e-3


def test_qei_in_unit_interval():
    case = synthetic_case(quality="clean", seed=2)
    m = compute_qei(case.cbf, case.gm, case.wm, case.csf)
    assert 0.0 <= m["qei"] <= 1.0


def test_qei_orders_clean_above_garbage():
    clean = synthetic_case(quality="clean", seed=3)
    border = synthetic_case(quality="borderline", seed=3)
    garbage = synthetic_case(quality="garbage", seed=3)
    qc = compute_qei(clean.cbf, clean.gm, clean.wm, clean.csf)["qei"]
    qb = compute_qei(border.cbf, border.gm, border.wm, border.csf)["qei"]
    qg = compute_qei(garbage.cbf, garbage.gm, garbage.wm, garbage.csf)["qei"]
    assert qc > qb > qg


def test_qei_verdicts():
    clean = synthetic_case(quality="clean", seed=4)
    garbage = synthetic_case(quality="garbage", seed=4)
    assert qei_check(cbf=clean.cbf, gm=clean.gm, wm=clean.wm, csf=clean.csf,
                     voxel_mm=clean.voxel_mm).verdict == Verdict.PASS
    assert qei_check(cbf=garbage.cbf, gm=garbage.gm, wm=garbage.wm, csf=garbage.csf,
                     voxel_mm=garbage.voxel_mm).verdict == Verdict.FAIL


def test_qei_missing_input_is_unknown():
    assert qei_check(cbf=None).verdict == Verdict.UNKNOWN


def test_negative_gm_penalty_drops_qei():
    """Injecting negatives into GM must reduce the score (f3 penalty)."""
    case = synthetic_case(quality="clean", seed=5)
    base = compute_qei(case.cbf, case.gm, case.wm, case.csf, smooth=False)["qei"]
    cbf2 = case.cbf.copy()
    gm_mask = case.gm > 0.7
    idx = np.argwhere(gm_mask)
    for z, y, x in idx[: len(idx) // 3]:        # flip a third of GM voxels negative
        cbf2[z, y, x] = -abs(cbf2[z, y, x]) - 5
    worse = compute_qei(cbf2, case.gm, case.wm, case.csf, smooth=False)["qei"]
    assert worse < base


# --------------------------------------------------------------------------- #
# Regressions found by the 2026-07 codebase audit. Both bugs inflated the QEI,
# i.e. they made bad scans look good - the worst direction for a QC tool to fail.
# --------------------------------------------------------------------------- #
def test_derived_csf_must_not_admit_the_air_background():
    """When CSF is not supplied it is derived as 1 - GM - WM. Outside the head
    GM and WM are both 0, so that complement is 1 across the whole background.
    Pooling the air into the within-tissue variance deflates the dispersion index
    and inflates the score, so the derived map has to be masked to the head."""
    import numpy as np

    from osipy_qc.checks.qei import qei_check
    from osipy_qc.synth import synthetic_case

    c = synthetic_case(quality="borderline", seed=3)
    truth = qei_check(cbf=c.cbf, gm=c.gm, wm=c.wm, csf=c.csf).metric["qei"]

    unmasked = np.clip(1.0 - c.gm - c.wm, 0.0, 1.0)          # the old, wrong form
    masked = unmasked * (c.cbf != 0)                          # what io.py now builds
    q_unmasked = qei_check(cbf=c.cbf, gm=c.gm, wm=c.wm, csf=unmasked).metric["qei"]
    q_masked = qei_check(cbf=c.cbf, gm=c.gm, wm=c.wm, csf=masked).metric["qei"]

    assert int((unmasked > 0.7).sum()) > 5 * int((masked > 0.7).sum())
    assert q_unmasked > truth + 0.04, "the unmasked form should visibly inflate the score"
    assert abs(q_masked - truth) < 0.02, "the masked form should track the true CSF result"


def test_load_cbf_inputs_masks_the_derived_csf_to_the_head(tmp_path):
    import nibabel as nib
    import numpy as np

    from osipy_qc.io import load_cbf_inputs
    from osipy_qc.synth import synthetic_case

    c = synthetic_case(quality="clean", seed=0)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])

    def w(name, arr):
        p = tmp_path / name
        nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), p)
        return str(p)

    got = load_cbf_inputs(w("cbf.nii.gz", c.cbf), w("gm.nii.gz", c.gm), w("wm.nii.gz", c.wm))
    assert got["csf_derived"] is True
    # nothing outside the imaged volume may be called CSF
    assert not np.any(got["csf"][np.asarray(c.cbf) == 0])


def test_degenerate_tissue_masks_report_unknown_rather_than_a_high_score():
    """An empty GM mask sends both penalty terms to their best value (a
    dispersion of 0 and a negative fraction of 0 both map to 1.0), so the old
    fallback handed the worst-prepared scans the highest QEI."""
    from osipy_qc.checks.qei import qei_check
    from osipy_qc.core import Verdict
    from osipy_qc.synth import synthetic_case

    c = synthetic_case(quality="borderline", seed=3)
    # scaled below the 0.7 tissue threshold: the maps exist but select nothing
    r = qei_check(cbf=c.cbf, gm=c.gm * 0.6, wm=c.wm * 0.6, csf=c.csf * 0.6)
    assert r.verdict is Verdict.UNKNOWN
    assert r.metric["qei"] is None
    assert r.metric["n_gm"] == 0
    assert "misaligned" in r.reason


def test_a_zero_cbf_map_with_valid_masks_is_unknown_not_scored():
    """Mean GM CBF of zero makes the dispersion index undefined."""
    import numpy as np

    from osipy_qc.checks.qei import qei_check
    from osipy_qc.core import Verdict
    from osipy_qc.synth import synthetic_case

    c = synthetic_case(quality="clean", seed=0)
    r = qei_check(cbf=np.zeros_like(c.cbf), gm=c.gm, wm=c.wm, csf=c.csf)
    assert r.verdict is Verdict.UNKNOWN


def test_the_five_millimetre_smoothing_is_actually_applied():
    """QEI's constants were fitted on 5 mm FWHM smoothed CBF, so the smoothing is
    part of the metric rather than a tidy-up before it.

    This pins it. Removing the smooth_fwhm call, or ignoring the configured
    width, previously left all 165 tests green — the step could have been
    deleted in a refactor and nothing would have said so.
    """
    from osipy_qc.checks.qei import compute_qei
    from osipy_qc.core.config import QCConfig

    rng = np.random.default_rng(0)
    shape = (28, 28, 22)
    gm = np.zeros(shape); gm[6:22, 6:22, 5:17] = 0.95
    wm = np.zeros(shape); wm[10:18, 10:18, 8:14] = 0.95
    csf = np.clip(1.0 - gm - wm, 0.0, 1.0)
    # high-frequency noise is exactly what a 5 mm kernel removes
    cbf = 60.0 * gm + 22.0 * wm + rng.normal(0, 18.0, shape)

    inputs = dict(cbf=cbf, gm=gm, wm=wm, csf=csf, voxel_mm=(3.0, 3.0, 3.0))

    none = compute_qei(**inputs, cfg=QCConfig(smooth_fwhm_mm=0.0))["qei"]
    five = compute_qei(**inputs, cfg=QCConfig(smooth_fwhm_mm=5.0))["qei"]
    wide = compute_qei(**inputs, cfg=QCConfig(smooth_fwhm_mm=12.0))["qei"]

    # smoothing raises the structural correlation, so the score moves with it
    assert none != five, "the configured smoothing width had no effect"
    assert five != wide, "changing the smoothing width had no effect"
    assert five > none, "5 mm smoothing should improve agreement with the template"

    # and the width is physical: the same request on bigger voxels is a smaller
    # kernel in voxels, so it cannot produce the same number
    coarse = compute_qei(**{**inputs, "voxel_mm": (6.0, 6.0, 6.0)},
                         cfg=QCConfig(smooth_fwhm_mm=5.0))["qei"]
    assert coarse != five, "voxel size was ignored when converting mm to voxels"
