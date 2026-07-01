"""Tests for the pipeline-agnostic CBF loader (load_cbf_inputs / grade_cbf)."""

import nibabel as nib
import numpy as np
import pytest

from osipy_qc.core import Verdict
from osipy_qc.io import grade_cbf, load_cbf_inputs
from osipy_qc.synth import synthetic_case


def _write(tmp_path, name, arr, voxel=(3.0, 3.0, 3.0)):
    aff = np.diag(list(voxel) + [1.0])
    p = str(tmp_path / name)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), p)
    return p


def test_load_cbf_inputs_from_paths(tmp_path):
    c = synthetic_case(quality="clean", seed=0)
    cbf = _write(tmp_path, "cbf.nii.gz", c.cbf)
    gm = _write(tmp_path, "gm.nii.gz", c.gm)
    wm = _write(tmp_path, "wm.nii.gz", c.wm)
    inputs = load_cbf_inputs(cbf, gm, wm)          # csf derived from gm+wm
    assert "cbf" in inputs and "gm" in inputs and "wm" in inputs and "csf" in inputs
    assert inputs["cbf"].shape == c.cbf.shape
    # derived CSF = clip(1-gm-wm, 0, 1)
    assert inputs["csf"].max() <= 1.0 and inputs["csf"].min() >= 0.0


def test_grade_cbf_end_to_end(tmp_path):
    c = synthetic_case(quality="clean", seed=0)
    paths = {n: _write(tmp_path, f"{n}.nii.gz", getattr(c, n)) for n in ("cbf", "gm", "wm", "csf")}
    report = grade_cbf(paths["cbf"], paths["gm"], paths["wm"], paths["csf"])
    qei = next(r for r in report.results if r.check == "1.qei")
    assert qei.verdict == Verdict.PASS            # real QEI runs from files
    assert report.overall in (Verdict.PASS, Verdict.WARN)


def test_cbf_only_qei_unknown(tmp_path):
    c = synthetic_case(quality="clean", seed=0)
    cbf = _write(tmp_path, "cbf.nii.gz", c.cbf)
    report = grade_cbf(cbf)                         # no tissue maps
    qei = next(r for r in report.results if r.check == "1.qei")
    assert qei.verdict == Verdict.UNKNOWN          # needs tissue maps


def test_shape_mismatch_raises(tmp_path):
    c = synthetic_case(quality="clean", seed=0)
    cbf = _write(tmp_path, "cbf.nii.gz", c.cbf)
    gm = _write(tmp_path, "gm_wrong.nii.gz", np.zeros((4, 4, 4)))
    with pytest.raises(ValueError, match="resample"):
        load_cbf_inputs(cbf, gm, gm)
