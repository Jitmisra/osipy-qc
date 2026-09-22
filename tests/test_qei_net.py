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

from osipy_qc.checks.qei_net import (_CLIP, _MIN_COVERAGE, _SATURATION_LIMIT,
                                     _brain_from_tissue, _covered_fraction,
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


# --------------------------------------------------------------------------- #
# the opposite failure: a map with nothing in it
# --------------------------------------------------------------------------- #
def test_an_empty_map_is_refused_rather_than_scored(monkeypatch):
    """The mirror of the saturation guard, and the same underlying problem.

    The network answers whatever it is asked, including when it is asked about
    nothing, and the answer looks like every other answer. Measured against this
    model: an all-zero volume scores 0.109, a flat constant volume 0.242, a
    1%-sparse volume 0.019 - not zero, not an error, and not even ordered by how
    much signal is present.

    Two of this project's own oxford_asl outputs came out 97.5% and 88.3% empty
    inside the brain mask and scored 0.259 and 0.104, which a reader would take
    for a poor-but-real quality estimate rather than for the absence of one.
    """
    monkeypatch.setenv("OSIPY_QEI_NET_PYTHON", sys.executable)
    monkeypatch.setenv("OSIPY_QEI_NET_SCRIPT", __file__)     # exists, never reached
    shape = (20, 20, 12)
    cbf = np.zeros(shape)
    cbf[:, :2] = 45.0                                        # ~10% covered
    r = qei_net_check(cbf=cbf, gm=np.ones(shape), affine=np.eye(4), cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN
    assert "grey and white matter carries" in r.reason and "4.2.coverage" in r.reason
    assert r.metric["covered_fraction"] < _MIN_COVERAGE


def test_a_totally_blank_map_is_refused(monkeypatch):
    monkeypatch.setenv("OSIPY_QEI_NET_PYTHON", sys.executable)
    monkeypatch.setenv("OSIPY_QEI_NET_SCRIPT", __file__)
    shape = (12, 12, 8)
    r = qei_net_check(cbf=np.zeros(shape), gm=np.ones(shape), affine=np.eye(4), cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN
    assert r.metric["covered_fraction"] == 0.0


def test_a_bad_map_is_still_scored_because_empty_is_not_the_same_as_bad(monkeypatch):
    """The guard must not become a way to duck grading poor data.

    A genuinely terrible map still has signal everywhere - the synthetic
    "garbage" case covers 100% of its mask - and QEI-Net is still asked about it.
    Only the absence of a measurement is refused.
    """
    from osipy_qc.synth import synthetic_case
    c = synthetic_case(quality="garbage", seed=0)
    assert _covered_fraction(c.cbf, c.gm, c.wm, CFG) > _MIN_COVERAGE
    monkeypatch.setenv("OSIPY_QEI_NET_PYTHON", sys.executable)
    monkeypatch.setenv("OSIPY_QEI_NET_SCRIPT", __file__)
    r = qei_net_check(cbf=c.cbf, gm=c.gm, wm=c.wm, csf=c.csf, affine=np.eye(4), cfg=CFG)
    assert "carries data" not in r.reason, "the emptiness guard fired on a map with data"


def test_the_coverage_limit_sits_in_the_measured_gap():
    """Not a round number picked by feel.

    Every map with real signal to hand covers at least 74% of its brain mask;
    both maps that came out near-empty cover 11.7% and 2.5%. The limit has to sit
    between those, with room on each side.
    """
    assert 0.20 < _MIN_COVERAGE < 0.70


def test_coverage_cannot_be_measured_without_tissue_maps():
    """4.2.coverage is silent without them too, so a refusal here would point
    the reader at a check that says nothing. The guard is skipped instead."""
    import math
    assert math.isnan(_covered_fraction(np.zeros((6, 6, 4)), None, None, CFG))
    # and a probability map on the wrong grid cannot answer it either
    assert math.isnan(_covered_fraction(np.zeros((6, 6, 4)), np.ones((8, 8, 4)), None, CFG))


def test_the_guard_measures_the_same_roi_as_the_check_it_cites():
    """It first measured (gm+wm+csf) > 0.5, and the two answered different
    questions.

    On a good real GE map whose CBF had been tissue-masked by its own pipeline,
    the guard read 55% while 4.2.coverage read 99.6% and PASSed - so it sat one
    point from refusing a perfectly good map for not measuring blood flow in
    cerebrospinal fluid, and would have sent the reader to a check reporting
    that nothing was wrong. CSF holding no perfusion is not a missing
    measurement.
    """
    shape = (12, 12, 8)
    gm = np.zeros(shape); gm[:, :4] = 0.9
    wm = np.zeros(shape); wm[:, 4:7] = 0.9
    csf = np.zeros(shape); csf[:, 7:] = 0.9          # 40% of a gm+wm+csf mask
    cbf = np.zeros(shape)
    cbf[(gm > 0.7) | (wm > 0.7)] = 45.0              # perfusion exactly where it belongs
    assert _covered_fraction(cbf, gm, wm, CFG) == pytest.approx(1.0)
    # the old denominator would have called this map 60% covered
    brain = (gm + wm + csf) > 0.5
    assert np.mean(cbf[brain] != 0) < 0.7


def test_an_epsilon_cannot_defeat_the_guard():
    """`!= 0` was not enough. A pipeline padding with 1e-9 rather than an exact
    zero passed it outright, and the two near-empty maps it was written for
    scored exactly as before."""
    shape = (12, 12, 8)
    gm = np.ones(shape)
    cbf = np.full(shape, 1e-9)
    cbf[:, :1] = 45.0                                 # ~8% real signal
    assert _covered_fraction(cbf, gm, None, CFG) < _MIN_COVERAGE
    # and a volume that is uniformly tiny carries nothing at all
    assert _covered_fraction(np.full(shape, 1e-9), gm, None, CFG) == 0.0
    # while a real map is unaffected by the floor
    assert _covered_fraction(np.full(shape, 45.0), gm, None, CFG) == pytest.approx(1.0)


def test_the_floor_is_relative_so_it_survives_a_change_of_units():
    """Units are not declared for brain, so a fixed physical floor would be
    wrong for a map in %M0 or in arbitrary units."""
    from osipy_qc.checks.qei_net import _ABSOLUTE_FLOOR, _data_floor
    # an all-zero map floors absolutely, so nothing in it counts as data
    assert _data_floor(np.array([0.0, 0.0])) == _ABSOLUTE_FLOOR
    # only a map with NO finite value at all has nothing to scale against
    assert _data_floor(np.array([np.nan, np.inf])) == 0.0
    assert _data_floor(np.full(100, 1e-9)) == _ABSOLUTE_FLOOR   # relative alone is blind
    assert _data_floor(np.linspace(0, 70, 100)) > _ABSOLUTE_FLOOR
    # a map scaled to ~1 (%M0 / a.u.) still admits its own real values
    small = np.linspace(0, 1.0, 100)
    assert _covered_fraction(small.reshape(10, 10, 1), np.ones((10, 10, 1)), None, CFG) > 0.9


def test_a_saturated_map_is_still_reported_as_saturated_not_as_empty(monkeypatch):
    """Both guards can look at the same map; the more specific diagnosis wins."""
    monkeypatch.setenv("OSIPY_QEI_NET_PYTHON", sys.executable)
    monkeypatch.setenv("OSIPY_QEI_NET_SCRIPT", __file__)
    cbf = _plausible_cbf(level=2700.0)
    r = qei_net_check(cbf=cbf, gm=np.ones(cbf.shape), affine=np.eye(4), cfg=CFG)
    assert "clip bound" in r.reason and "carries data" not in r.reason


# --------------------------------------------------------------------------- #
# the mentor-facing setup scripts
# --------------------------------------------------------------------------- #
# These exist so someone who is not a developer can enable QEI-Net with two
# commands. An adversarial review of the first version confirmed twenty defects,
# most of them in the same place: the scripts decided things by looking for a
# FILE rather than by checking whether anything worked.

SCRIPTS = ROOT / "scripts"


def _script(name: str) -> str:
    return (SCRIPTS / name).read_text()


def test_the_setup_scripts_are_valid_python():
    import py_compile
    for name in ("setup_qei_net.py", "run_ui.py"):
        py_compile.compile(str(SCRIPTS / name), doraise=True)


def test_the_drop_folder_ships_with_instructions_but_no_model():
    """The folder is tracked so the instructions travel with the repo; anything
    put INTO it is not."""
    drop = ROOT / "qei_net_model"
    assert (drop / "README.md").is_file()
    out = subprocess.run(["git", "check-ignore", "-q",
                          "qei_net_model/qei_inference_package.zip"],
                         cwd=ROOT, capture_output=True)
    assert out.returncode == 0, "a zip dropped in the folder is NOT gitignored"
    out = subprocess.run(["git", "check-ignore", "-q", "qei_net_model/README.md"],
                         cwd=ROOT, capture_output=True)
    assert out.returncode != 0, "the instructions must stay tracked"


def test_a_half_built_environment_is_detected_rather_than_reused():
    """The worst defect the review found, and the one a mentor would actually hit.

    `python -m venv` succeeds in seconds and creates the interpreter; every pip
    call after it can fail on its own. Gating reuse on "the interpreter exists"
    meant one failed install poisoned the folder permanently - every later run
    printed "reusing the environment", installed nothing, and died in a
    traceback, with the cure (delete a folder by hand) documented nowhere.
    """
    src = _script("setup_qei_net.py")
    assert "def env_is_usable" in src
    assert "import " in src and "DEPS" in src
    assert "if not rebuild and env_is_usable(py)" in src, (
        "reuse is not gated on the environment actually working")
    assert '"--rebuild"' in src


def test_run_ui_does_not_claim_the_model_is_on_without_checking():
    """It printed "QEI-Net: ON" from two filenames existing. A half-built
    environment passed that, and then every scan returned UNKNOWN "No module
    named 'torch'" - the console and the report saying opposite things."""
    src = _script("run_ui.py")
    assert "def _runnable" in src
    assert "_runnable(py)" in src, "find_model returns without probing the env"
    assert "environment is incomplete" in src, "no message for the half-built case"


def test_the_new_package_is_validated_before_the_old_one_is_deleted():
    """A truncated zip used to delete a working install first and then fail,
    leaving no model at all and a BadZipFile traceback to explain it."""
    src = _script("setup_qei_net.py")
    stage_at = src.index("def stage_zip")
    assert "testzip()" in src, "the archive is never checked for damage"
    # check_layout runs against the staged copy, and only then is PKG replaced
    assert src.index("check_layout(staged)") < src.index("shutil.rmtree(PKG")


def test_failures_are_sentences_not_tracebacks():
    """The audience is stated: researchers who will not debug a traceback."""
    src = _script("setup_qei_net.py")
    assert "except Exception as exc:" in src, "main() is not wrapped"
    assert "except KeyboardInterrupt" in src
    assert "STOPPED:" in src


def test_the_instructions_use_the_interpreter_each_platform_actually_has():
    """macOS ships no `python`, only `python3`; Windows is the other way round.
    Printing one to both audiences sends half the readers to a shell error."""
    for name in ("setup_qei_net.py", "run_ui.py"):
        assert "def py_cmd" in _script(name), name
    readme = (ROOT / "qei_net_model" / "README.md").read_text()
    assert "python3 scripts/setup_qei_net.py" in readme
    assert "Windows: python scripts" in readme


def test_a_partial_weights_package_is_run_with_the_folds_it_has():
    """The first version said a partial package "will still run". It does not:
    run_qei.py defaults to folds 0-4 and exits when one is missing. It now
    passes the folds actually present, and says the score is not comparable."""
    src = _script("setup_qei_net.py")
    assert "def fold_args" in src and '"--folds"' in src
    assert "NOT the published one" in src


def test_the_smoke_test_exercises_the_path_a_real_map_takes():
    """An on-grid volume skips resampling and torchio entirely, so "it works"
    would have been true only of the one case no real CBF map is in."""
    src = _script("setup_qei_net.py")
    assert "3.0, 3.0, 3.0" in src, "the test volume is not off-grid"
    assert '"--mask", mask' in src, (
        "an off-grid volume needs an explicit mask; --derive_mask_from_cbf "
        "requires the model's own grid")
