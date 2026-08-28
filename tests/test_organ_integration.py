"""End-to-end integration across every organ and every entry point.

The unit tests check each piece; this checks that the pieces are WIRED. Most of
the defects this file exists to catch were wiring, not logic: a control rendered
on the form with no server-side reader, a map arriving under a key its own
organ's checks do not read, an import naming a module that does not exist.
"""

from __future__ import annotations

import contextlib
import io as _io
import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from osipy_qc import web
from osipy_qc.batch import tunable_groups
from osipy_qc.cli import main as cli_main
from osipy_qc.core.config import ORGANS, QCConfig, THRESHOLD_PROVENANCE, for_organ
from osipy_qc.core.registry import organs_covered
from osipy_qc.report import run_qc
from osipy_qc.synth import (synthetic_case, synthetic_kidney_case,
                            synthetic_placenta_case)

ORGAN_LIST = ("brain", "kidney", "placenta")
BOUNDARY = "----osipyqcintegration"


def _part(name, filename=None, data=b""):
    head = f'form-data; name="{name}"'
    if filename is not None:
        head += f'; filename="{filename}"'
    return (b"--" + BOUNDARY.encode() + b"\r\nContent-Disposition: "
            + head.encode() + b"\r\n\r\n" + data + b"\r\n")


def _nii(arr, voxel=(3.0, 3.0, 3.0)):
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "x.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32),
                                 np.diag(list(voxel) + [1.0])), p)
        with open(p, "rb") as fh:
            return fh.read()


def _grade(body):
    fields = web._parse_multipart(body + b"--" + BOUNDARY.encode() + b"--\r\n",
                                  f"multipart/form-data; boundary={BOUNDARY}")
    out = web._grade_upload(fields)
    return out[0] if isinstance(out, tuple) else out


# --------------------------------------------------------------------------- #
# registry / config wiring
# --------------------------------------------------------------------------- #
def test_every_organ_is_registered_profiled_and_tunable():
    assert organs_covered() == {"brain": 20, "kidney": 19, "placenta": 15}
    assert set(ORGANS) == set(ORGAN_LIST)
    cfg = QCConfig()
    for organ in ORGAN_LIST:
        assert for_organ(organ).organ == organ
        missing = [f for _g, fields in tunable_groups(organ) for f, _l in fields
                   if not hasattr(cfg, f)]
        assert not missing, f"{organ} exposes thresholds that do not exist: {missing}"


def test_no_provenance_row_names_a_field_that_does_not_exist():
    """--provenance and the report both read this table; a stale name there
    prints a threshold nobody can set."""
    cfg = QCConfig()
    ghosts = [k for k in THRESHOLD_PROVENANCE if not hasattr(cfg, k)]
    assert not ghosts, ghosts


@pytest.mark.parametrize("organ", ORGAN_LIST)
def test_an_organ_runs_with_no_inputs_at_all_and_never_crashes(organ):
    """run_qc swallows exceptions into 'check error: ...', a stack-trace fragment
    where a reader needs an instruction. No check may reach that path."""
    rep = run_qc({}, cfg=QCConfig(organ=organ))
    assert len(rep.results) == organs_covered()[organ]
    assert not [r for r in rep.results if "check error" in r.reason]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("organ", ORGAN_LIST)
def test_organ_demo_runs_from_the_cli(organ):
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli_main(["--organ-demo", organ])
    assert rc == 0 and "OVERALL" in buf.getvalue()


def test_organ_demo_honours_html_and_json(tmp_path):
    """--organ-demo returned before the --html block, and the import it then
    gained named a module (osipy_qc.render) that does not exist."""
    out = str(tmp_path / "r.html")
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli_main(["--organ-demo", "kidney", "--html", out])
    assert os.path.getsize(out) > 5000
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli_main(["--organ-demo", "placenta", "--json"])
    assert '"overall_verdict"' in buf.getvalue()


# --------------------------------------------------------------------------- #
# the console
# --------------------------------------------------------------------------- #
def test_the_form_offers_each_organ_its_own_controls():
    page = web._upload_page()
    for organ in ORGAN_LIST:
        assert f'name="organ" value="{organ}"' in page
    # masks, per side, because R10.1 reports the two kidneys separately
    for kind in ("kidney", "cortex", "medulla"):
        for side in ("left", "right"):
            assert f'name="kidney__{kind}_{side}"' in page
    assert 'name="placenta__placenta_mask"' in page
    # the facts that gate whole checks
    for field in ("kidney__units", "kidney__pld_or_ti_s", "placenta__declared_units",
                  "placenta__gestational_age_wk", "placenta__labelling_scheme"):
        assert f'name="{field}"' in page, field
    # thresholds are organ-scoped, not the brain's for everyone
    assert page.count('data-thr="kidney_') == 20
    assert page.count('data-thr="placenta_') == 19
    assert "function applyOrgan" in page


def test_the_error_page_still_renders():
    assert len(web._upload_page("something went wrong")) > 40_000


def test_brain_upload_is_unaffected_by_the_organ_work():
    c = synthetic_case(quality="clean", seed=0)
    body = (_part("cbf", "cbf.nii.gz", _nii(c.cbf))
            + _part("gm", "gm.nii.gz", _nii(c.gm))
            + _part("wm", "wm.nii.gz", _nii(c.wm))
            + _part("organ", data=b"brain") + _part("population", data=b"adult")
            + _part("files", "", b"") * 2)
    data = _grade(body)
    assert data["verdict"] == "PASS"
    assert not [x for x in data["checks"] if x["id"][0] in "kp"]


def test_kidney_upload_carries_masks_and_facts_all_the_way_through():
    """Before the organ sections existed this reached ONE of nine CBF-map checks,
    because the masks had nowhere to arrive and the facts had no fields."""
    c = synthetic_kidney_case(quality="clean", seed=0)
    vox = (2.0, 2.0, 8.0)
    body = _part("cbf", "rbf.nii.gz", _nii(c.rbf, vox))
    for side in ("left", "right"):
        for kind, masks in (("kidney", c.kidney_masks), ("cortex", c.cortex_masks),
                            ("medulla", c.medulla_masks)):
            body += _part(f"kidney__{kind}_{side}", f"{kind}.nii.gz", _nii(masks[side], vox))
    for key, val in (("units", "mL/100g/min"), ("labelling", "pCASL"),
                     ("pld_or_ti_s", "1.4"), ("field_strength_t", "3"),
                     ("breathing_strategy", "free breathing"), ("readout", "3D")):
        body += _part(f"kidney__{key}", data=val.encode())
    body += (_part("organ", data=b"kidney") + _part("population", data=b"adult")
             + _part("files", "", b"") * 2)
    data = _grade(body)
    ids = {x["id"]: x for x in data["checks"]}
    assert data["coverage"]["graded"] >= 4
    # the consensus quantity, per kidney, actually computed from the uploaded map
    assert ids["k3.1.cortical_rbf"]["verdict"] == "INFO"
    assert "300" in ids["k3.1.cortical_rbf"]["reason"]
    assert ids["k4.1.mask_integrity"]["verdict"] == "PASS"
    assert "6 mask" in ids["k4.1.mask_integrity"]["reason"]
    assert not [x for x in data["checks"] if x["id"][0].isdigit()]


def test_placenta_upload_can_declare_units_and_context():
    """P2.1 gates every magnitude check on a declared unit; with no field for it
    a placenta upload could never get past."""
    c = synthetic_placenta_case(quality="clean", seed=0)
    body = (_part("cbf", "perf.nii.gz", _nii(c.perfusion))
            + _part("placenta__placenta_mask", "pm.nii.gz", _nii(c.placenta_mask)))
    for key, val in (("declared_units", "mL/100g/min"), ("labelling_scheme", "VSASL"),
                     ("gestational_age_wk", "30"), ("maternal_position", "lateral"),
                     ("field_strength_T", "3"), ("mask_source", "manual_on_m0"),
                     ("normalisation_mode", "scalar"),
                     ("registration_model", "non-rigid (DSVR)"),
                     ("lambda", "0.9"), ("alpha", "0.767"), ("t1_blood_ms", "1650")):
        body += _part(f"placenta__{key}", data=val.encode())
    body += (_part("organ", data=b"placenta") + _part("population", data=b"adult")
             + _part("files", "", b"") * 2)
    data = _grade(body)
    ids = {x["id"]: x for x in data["checks"]}
    assert ids["p2.1.units_declaration"]["verdict"] == "PASS"
    assert "manual_on_m0" in ids["p3.1.mask_integrity"]["reason"]
    assert data["verdict"] == "PASS"
    assert data["coverage"]["complete"] is True


def test_a_mask_on_the_wrong_grid_is_actionable_not_a_stack_trace():
    c = synthetic_kidney_case(quality="clean", seed=0)
    vox = (2.0, 2.0, 8.0)
    body = (_part("cbf", "rbf.nii.gz", _nii(c.rbf, vox))
            + _part("kidney__cortex_left", "c.nii.gz", _nii(np.ones((4, 4, 4)), vox))
            + _part("kidney__units", data=b"mL/100g/min")
            + _part("organ", data=b"kidney") + _part("population", data=b"adult")
            + _part("files", "", b"") * 2)
    data = _grade(body)
    assert any("resample" in x["reason"] for x in data["checks"])
    assert not [x for x in data["checks"] if "check error" in x["reason"]]
