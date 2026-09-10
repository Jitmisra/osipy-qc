"""QEI-Net, the optional deep-learning quality index.

The model is not part of this package and cannot be in CI, so these tests cover
everything around it: that an unconfigured install is unaffected, that a
mis-scaled map is refused rather than scored, and that a score is reported and
never graded. The one test that does run the model is skipped unless the
environment points at it.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from osipy_qc.checks.qei_net import (_CLIP, _SATURATION_LIMIT, _brain_from_tissue,
                                     _saturated_fraction, qei_net_check)
from osipy_qc.core import Verdict
from osipy_qc.core.config import QCConfig
from osipy_qc.core.registry import all_checks

CFG = QCConfig()
ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIGURED = bool(os.environ.get("OSIPY_QEI_NET_PYTHON")
                  and os.environ.get("OSIPY_QEI_NET_SCRIPT"))


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch, request):
    """Most tests assert the unconfigured path, so the ambient env must not leak in."""
    if "uses_real_model" not in request.keywords:
        monkeypatch.delenv("OSIPY_QEI_NET_PYTHON", raising=False)
        monkeypatch.delenv("OSIPY_QEI_NET_SCRIPT", raising=False)


def _plausible_cbf(shape=(20, 20, 12), level=55.0, seed=0):
    rng = np.random.RandomState(seed)
    return np.abs(rng.normal(level, 8.0, shape))


def test_an_unconfigured_install_is_not_applicable_rather_than_unknown():
    """The model is optional. Not having it must never look like a failure, and
    must not dent the coverage figure either.

    UNKNOWN means the DATA was missing something. A model the operator chose not
    to install says nothing about the data, and marking it UNKNOWN would leave
    every report on every ordinary install permanently "incomplete".
    """
    r = qei_net_check(cbf=_plausible_cbf(), cfg=CFG)
    assert r.verdict is Verdict.NA
    assert "OSIPY_QEI_NET_PYTHON" in r.reason
    assert "not part of this package" in r.reason


def test_a_configured_path_that_does_not_exist_is_reported_not_crashed(monkeypatch):
    monkeypatch.setenv("OSIPY_QEI_NET_PYTHON", "/nope/python")
    monkeypatch.setenv("OSIPY_QEI_NET_SCRIPT", "/nope/run_qei.py")
    r = qei_net_check(cbf=_plausible_cbf(), cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN
    assert "not at the path given" in r.reason


def test_a_saturated_map_is_refused_rather_than_scored(monkeypatch):
    """The guard that motivates this whole check.

    QEI-Net normalises with clip(cbf, -100, 100) / 100. On a real GE map whose
    calibration was about fifty times too high, 74% of brain voxels pinned to
    +1.0, only two distinct values survived, and the network still returned
    0.634 while the classical QEI scored the same map 0.0006. A number computed
    from a flattened volume is not a quality measurement.
    """
    monkeypatch.setenv("OSIPY_QEI_NET_PYTHON", sys.executable)
    monkeypatch.setenv("OSIPY_QEI_NET_SCRIPT", __file__)   # exists, never reached
    cbf = _plausible_cbf(level=2700.0)                     # the mis-scaled case
    r = qei_net_check(cbf=cbf, gm=np.ones(cbf.shape), cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN
    assert "clip bound" in r.reason and "3.1.cbf_level" in r.reason
    assert r.metric["saturated_fraction"] > _SATURATION_LIMIT


def test_a_well_scaled_map_is_not_refused_by_the_guard(monkeypatch):
    """The guard must not fire on ordinary data, or it would suppress every score."""
    cbf = _plausible_cbf(level=55.0)
    assert _saturated_fraction(cbf, np.ones(cbf.shape, bool)) < _SATURATION_LIMIT
    monkeypatch.setenv("OSIPY_QEI_NET_PYTHON", sys.executable)
    monkeypatch.setenv("OSIPY_QEI_NET_SCRIPT", __file__)
    r = qei_net_check(cbf=cbf, gm=np.ones(cbf.shape), affine=np.eye(4), cfg=CFG)
    # It gets past the guard and fails later, on running this file as the model.
    assert "clip bound" not in r.reason


def test_the_saturated_fraction_matches_the_authors_clip():
    """Half the voxels at the bound must read as half, not as some other number."""
    cbf = np.concatenate([np.full(500, 150.0), np.full(500, 50.0)]).reshape(10, 10, 10)
    assert _saturated_fraction(cbf, np.ones(cbf.shape, bool)) == pytest.approx(0.5)
    assert _CLIP == 100.0          # mirrors preprocess.py; if that changes, this fails


def test_a_bare_array_with_no_affine_is_refused(monkeypatch):
    """The model resamples to its own grid, which a bare array cannot support."""
    monkeypatch.setenv("OSIPY_QEI_NET_PYTHON", sys.executable)
    monkeypatch.setenv("OSIPY_QEI_NET_SCRIPT", __file__)
    r = qei_net_check(cbf=_plausible_cbf(), cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN
    assert "affine" in r.reason


def test_the_brain_mask_is_the_union_of_the_tissue_maps():
    gm = np.zeros((4, 4, 4)); gm[0] = 0.9
    wm = np.zeros((4, 4, 4)); wm[1] = 0.9
    brain = _brain_from_tissue(gm, wm, None)
    assert brain[0].all() and brain[1].all() and not brain[2].any()
    assert _brain_from_tissue(None, None, None) is None


def test_not_installing_the_model_does_not_make_a_report_incomplete():
    """The coverage figure must keep meaning "some of this scan could not be
    looked at", not "you did not install an optional model"."""
    from osipy_qc.core.result import coverage
    r = qei_net_check(cbf=_plausible_cbf(), cfg=CFG)
    cov = coverage([r])
    assert cov["unknown"] == 0 and cov["complete"] is True


def test_a_configured_model_that_fails_IS_unknown(monkeypatch):
    """The other side of the same rule: once configured, a failure is a real gap."""
    monkeypatch.setenv("OSIPY_QEI_NET_PYTHON", "/nope/python")
    monkeypatch.setenv("OSIPY_QEI_NET_SCRIPT", "/nope/run_qei.py")
    r = qei_net_check(cbf=_plausible_cbf(), cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN


def test_it_is_optional_so_the_registry_marks_it_not_required():
    entry = all_checks("brain")["1.1.qei_net"]
    assert entry["required"] is False, (
        "a model that is not shipped must not be a required check")
    assert entry["stream"] == "B"


def test_the_weights_are_ignored_by_git():
    """Xavi asked for these to stay private. Prove the rules actually bite."""
    for probe in ("weights/best_model.pth", "qei_inference_package/x.pth", "a.ckpt"):
        out = subprocess.run(["git", "check-ignore", "-q", probe],
                             cwd=ROOT, capture_output=True)
        assert out.returncode == 0, f"{probe} is NOT gitignored"


@pytest.mark.uses_real_model
@pytest.mark.skipif(not CONFIGURED, reason="QEI-Net not configured in this environment")
def test_it_scores_a_real_map_and_records_which_model_did_it(tmp_path):
    """Runs the actual model when the environment points at it."""
    cbf_path = ROOT / "example_data" / "example_cbf.nii.gz"
    img = nib.load(cbf_path)
    tissue = {k: np.asanyarray(nib.load(ROOT / "example_data" / f"example_{k}.nii.gz").dataobj)
              for k in ("gm", "wm", "csf")}
    r = qei_net_check(cbf=np.asanyarray(img.dataobj, dtype=float),
                      cbf_path=str(cbf_path), affine=img.affine, cfg=CFG, **tissue)
    assert r.verdict is Verdict.INFO, r.reason
    assert 0.0 <= r.metric["qei_net"] <= 1.0
    assert r.metric["model"] != "unknown"          # the fingerprint must be real
    assert "not graded" in r.reason                # never decides a verdict
