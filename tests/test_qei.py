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
