"""Loading a folder of PIPELINE OUTPUT, not just a folder of raw acquisitions.

Every dataset this project was built against was raw: an ASL series, an M0, a
T1. So the loader only ever learned three roles, and pointing it at a
derivatives folder - a quantified CBF map with tissue maps resampled beside it,
which is what ASLPrep and oxford_asl actually emit - graded nothing at all.

Measured on a mentor dataset that has both halves in one folder:

    before   coverage:  3 of 17 applicable checks decided
    after    coverage: 13 of 15

The gap was not one bug. The CBF map was filed as the ASL acquisition because
`..._PCASL3D_label-meancbf.nii` contains "pcasl"; the tissue maps were filed the
same way because `..._probseg_aslspace.nii.gz` contains "asl"; and once a 3-D
map had claimed the ASL role, the real 12-volume series underneath it was never
read, so the motion and control/label checks reported "no series" about a series
in the same folder.

These tests pin each of those separately, because they fail separately.
"""

from __future__ import annotations

import pathlib

import nibabel as nib
import numpy as np
import pytest

from osipy_qc.checks.schema import (classify_derivative, classify_role, detect_dataset,
                                    primary_asl, spatial_ndim, swap_check)
from osipy_qc.core import Verdict
from osipy_qc.core.config import QCConfig
from osipy_qc.io import _tr_from_header, _variant_rank, load_folder
from osipy_qc.report import run_qc

MENTOR = pathlib.Path("/Users/agnik/Desktop/gsoc-osipi/qcdata/brain/Agnik_Data")


# --------------------------------------------------------------------------- #
# the vocabulary
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,expected", [
    # the mentor dataset, verbatim
    ("C03_C03_20160825_1_PCASL3D_label-meancbf.nii", "cbf"),
    ("C03_C03_20160825_1_T1w_label-GM_probseg_aslspace.nii.gz", "gm"),
    ("C03_C03_20160825_1_T1w_label-WM_probseg_aslspace.nii.gz", "wm"),
    ("C03_C03_20160825_1_T1w_label-CSF_probseg_aslspace.nii.gz", "csf"),
    # oxford_asl
    ("perfusion_calib.nii.gz", "cbf"),
    ("pvgm_inasl.nii.gz", "gm"),
    ("pvwm_inasl.nii.gz", "wm"),
    # ASLPrep / BIDS
    ("sub-01_space-asl_cbf.nii.gz", "cbf"),
    ("gm.nii.gz", "gm"),
])
def test_real_pipeline_output_is_recognised(name, expected):
    assert classify_derivative(name) == expected


@pytest.mark.parametrize("name", [
    # "gm" and "wm" as bare substrings match all of these; the boundary in the
    # pattern is the only thing keeping them out.
    "segmentation.nii.gz", "Sigma_map.nii.gz", "swMask.nii.gz", "augmented.nii.gz",
    # acquisitions, which belong to the OTHER vocabulary
    "ASL.nii.gz", "PCASL.nii.gz", "M0.nii.gz", "MPRAGE.nii.gz", "T1.nii.gz",
    "calib.nii.gz",
    # a subtraction is not a quantified CBF map, and grading one against
    # published mL/100g/min bands would fail every scan that produced it
    "deltam.nii.gz", "delta_m.nii.gz",
])
def test_names_that_must_not_be_read_as_derivative_maps(name):
    assert classify_derivative(name) is None


def test_a_real_series_is_never_a_derivative_however_it_is_named():
    assert classify_derivative("sub-01_cbf.nii.gz", shape=(64, 56, 34, 12)) is None
    assert classify_derivative("sub-01_cbf.nii.gz", shape=(64, 56, 34)) == "cbf"


def test_a_trailing_singleton_axis_is_not_a_volume():
    """ASLPrep, FSL and SPM all write a 3-D map as (x, y, z, 1).

    Counting that as 4-D read every ASLPrep CBF map as a control/label series
    and kept all of them out of Stream B.
    """
    assert spatial_ndim((81, 101, 43, 1)) == 3
    assert spatial_ndim((81, 101, 43, 1, 1)) == 3
    assert spatial_ndim((64, 56, 34, 12)) == 4
    assert spatial_ndim((64, 56, 34)) == 3
    assert classify_derivative("sub01_cbf.nii.gz", shape=(81, 101, 43, 1)) == "cbf"


def test_the_acquisition_vocabulary_is_untouched():
    """classify_role is also the upload page's per-role vocabulary and its
    answers are pinned elsewhere. The derivative rules are a separate layer
    applied by the loader, not a widening of this one."""
    assert classify_role("perfusion_calib.nii.gz") == "asl"
    assert classify_role("C03_C03_20160825_1_T1w_label-GM_probseg_aslspace.nii.gz") == "asl"


# --------------------------------------------------------------------------- #
# picking the right ASL out of several
# --------------------------------------------------------------------------- #
def test_the_series_wins_over_a_3d_map_whatever_the_walk_order():
    """os.walk yields a folder's own files before its subdirectories, so the
    3-D map at the top answered for `raw/ASL.nii.gz` underneath it."""
    cbf_map = {"name": "meancbf.nii", "shape": (64, 56, 34)}
    series = {"name": "ASL.nii.gz", "shape": (64, 56, 34, 12)}
    assert primary_asl([cbf_map, series]) is series
    assert primary_asl([series, cbf_map]) is series
    assert primary_asl([cbf_map]) is cbf_map
    assert primary_asl([]) is None


def test_detect_dataset_does_not_crash_on_a_derivative_role():
    """It indexed a fixed four-key dict, so the first folder containing a CBF
    map raised KeyError instead of describing the dataset."""
    files = [{"name": "cbf.nii.gz", "shape": (10, 10, 10), "role": "cbf"},
             {"name": "gm.nii.gz", "shape": (10, 10, 10), "role": "gm"},
             {"name": "ASL.nii.gz", "shape": (10, 10, 10, 8), "role": "asl"}]
    det = detect_dataset(files)
    assert det["structure"] == "control/label series (8 volumes)"


def test_a_single_volume_map_is_pre_subtracted_not_a_one_volume_series():
    files = [{"name": "ASL.nii.gz", "shape": (10, 10, 10, 1), "role": "asl"}]
    assert detect_dataset(files)["structure"] == "pre-subtracted deltaM"


# --------------------------------------------------------------------------- #
# the M0 TR read from the header
# --------------------------------------------------------------------------- #
def _write(path, arr, tr=None, t_unit="sec"):
    img = nib.Nifti1Image(np.asarray(arr, dtype=np.float32), np.eye(4))
    if tr is not None:
        img.header["pixdim"][4] = tr
        img.header.set_xyzt_units(xyz="mm", t=t_unit)
    nib.save(img, str(path))
    return str(path)


def test_the_m0_tr_is_read_from_a_4d_header(tmp_path):
    p = _write(tmp_path / "m0.nii.gz", np.ones((4, 4, 4, 2)), tr=6.0)
    assert _tr_from_header(p) == pytest.approx(6.0)


def test_milliseconds_are_converted(tmp_path):
    p = _write(tmp_path / "m0.nii.gz", np.ones((4, 4, 4, 2)), tr=6000.0, t_unit="msec")
    assert _tr_from_header(p) == pytest.approx(6.0)


def test_pixdim4_on_a_3d_image_is_not_a_repetition_time(tmp_path):
    """The MPRAGE in this project's own test data carries 1.000 there and the T1
    carries 2.400. Read as a TR, either would WARN and hand the reader a
    relaxation correction computed from a number that means nothing."""
    p = _write(tmp_path / "t1.nii.gz", np.ones((4, 4, 4)), tr=2.4)
    assert _tr_from_header(p) is None


def test_an_undeclared_time_unit_is_not_assumed_to_be_seconds(tmp_path):
    p = _write(tmp_path / "m0.nii.gz", np.ones((4, 4, 4, 2)), tr=6.0, t_unit="unknown")
    assert _tr_from_header(p) is None


def test_zero_is_how_unset_is_spelled(tmp_path):
    p = _write(tmp_path / "m0.nii.gz", np.ones((4, 4, 4, 2)), tr=0.0)
    assert _tr_from_header(p) is None


def test_a_stated_tr_always_beats_the_header(tmp_path):
    """A sidecar states the TR; a header only implies it."""
    _write(tmp_path / "ASL.nii.gz", np.ones((4, 4, 4, 4)))
    _write(tmp_path / "M0.nii.gz", np.ones((4, 4, 4, 2)), tr=6.0)
    (tmp_path / "M0.json").write_text('{"RepetitionTimePreparation": 3.2}')
    got = load_folder(str(tmp_path), load_arrays=False)
    assert got["m0_tr_s"] == pytest.approx(3.2)
    assert got["m0_tr_source"] == "BIDS sidecar"


# --------------------------------------------------------------------------- #
# choosing among pipeline variants
# --------------------------------------------------------------------------- #
def test_the_unqualified_cbf_map_is_graded_not_a_denoised_variant():
    """ASLPrep writes _cbf plus _desc-score_ and _desc-scrub_ versions of it.
    The denoised ones score better by construction, so grading whichever the
    walk reached first would quietly flatter the scan."""
    names = ["sub-01_desc-score_cbf.nii.gz", "sub-01_cbf.nii.gz",
             "sub-01_desc-scrub_cbf.nii.gz", "sub-01_desc-basil_cbf.nii.gz"]
    ranked = sorted(({"path": n} for n in names), key=_variant_rank)
    assert ranked[0]["path"] == "sub-01_cbf.nii.gz"


def test_the_calibrated_map_wins_over_the_uncalibrated_one():
    """oxford_asl writes both, and only `_calib` is in mL/100g/min.

    The uncalibrated name is SHORTER, so the shortest-name tie-break would pick
    it - and 3.1.cbf_level would then FAIL a scan for having the wrong units
    rather than the wrong CBF.
    """
    ranked = sorted(({"path": n} for n in ["perfusion.nii.gz", "perfusion_calib.nii.gz"]),
                    key=_variant_rank)
    assert ranked[0]["path"] == "perfusion_calib.nii.gz"


def test_find_aslprep_uses_the_shared_vocabulary(tmp_path):
    """Its private glob list had drifted: it wanted `*_cbf.nii.gz` or
    `*_cbf.nii`, so a real dataset's `..._label-meancbf.nii` matched neither -
    uncompressed, and no underscore before "cbf" - and the adapter handed back
    three tissue maps with `cbf: None`."""
    from osipy_qc.io import find_aslprep
    for n in ("sub-01_PCASL3D_label-meancbf.nii",
              "sub-01_T1w_label-GM_probseg_aslspace.nii.gz",
              "sub-01_T1w_label-WM_probseg_aslspace.nii.gz",
              "sub-01_T1w_label-CSF_probseg_aslspace.nii.gz"):
        _write(tmp_path / n, np.ones((6, 6, 4)))
    got = find_aslprep(str(tmp_path))
    assert got["cbf"] is not None and got["cbf"].endswith("label-meancbf.nii")
    assert all(got[k] is not None for k in ("gm", "wm", "csf"))


# --------------------------------------------------------------------------- #
# the swap check no longer presents an assumption as evidence
# --------------------------------------------------------------------------- #
def _label_first_series(shape=(6, 6, 4), n=6):
    """Odd volumes brighter: a label-first acquisition, or a swap. Identical."""
    arr = np.zeros((*shape, n * 2))
    arr[..., 0::2] = 100.0
    arr[..., 1::2] = 118.0
    return arr


def test_an_assumed_order_cannot_produce_a_hard_failure():
    """Found on a real mentor dataset: all six pairs had the odd volume brighter
    by 15-17%, on a scan whose own CBF map is positive and well formed. Whatever
    produced that map read the order correctly; only this check could not."""
    r = swap_check(asl_4d=_label_first_series(), structure="control/label series",
                   cfg=QCConfig(strict=True))
    assert r.verdict is Verdict.FAIL
    assert r.provisional is True
    assert "label-first acquisition" in r.reason and "aslcontext" in r.reason


def test_and_it_softens_with_strict_off():
    r = swap_check(asl_4d=_label_first_series(), structure="control/label series",
                   cfg=QCConfig(strict=False))
    assert r.verdict is Verdict.WARN and r.provisional is True


def test_but_a_swap_contradicting_a_stated_order_is_hard_evidence():
    """aslcontext.tsv named which volumes are control. They are the darker ones.
    Nothing was assumed, so nothing is provisional."""
    rows = ["control", "label"] * 6
    r = swap_check(asl_4d=_label_first_series(), aslcontext_rows=rows,
                   structure="control/label series", cfg=QCConfig(strict=True))
    assert r.verdict is Verdict.FAIL
    assert r.provisional is False
    assert "aslcontext.tsv" in r.reason


def test_a_normal_series_still_passes():
    arr = _label_first_series()[..., ::-1].copy()   # even volumes brighter
    r = swap_check(asl_4d=arr, structure="control/label series", cfg=QCConfig())
    assert r.verdict is Verdict.PASS and r.provisional is False


# --------------------------------------------------------------------------- #
# end to end on a synthetic derivatives folder
# --------------------------------------------------------------------------- #
def _derivatives_folder(tmp_path, gm_shape=(12, 12, 8)):
    rng = np.random.RandomState(0)
    shape = (12, 12, 8)
    # A CSF band is not decoration: QEI pools within-tissue variance over GM, WM
    # AND CSF, and refuses to score when one of them is empty. GM+WM covering
    # the whole volume leaves the derived CSF at zero voxels.
    gm = np.zeros(shape); gm[:, :5] = 0.9
    wm = np.zeros(shape); wm[:, 5:9] = 0.9
    cbf = np.full(shape, 5.0) + rng.normal(0, 1, shape)      # the CSF band
    cbf[:, :5] = rng.normal(60, 5, (shape[0], 5, shape[2]))
    cbf[:, 5:9] = rng.normal(22, 3, (shape[0], 4, shape[2]))
    _write(tmp_path / "sub-01_label-meancbf.nii", cbf)
    _write(tmp_path / "sub-01_label-GM_probseg_aslspace.nii.gz",
           np.broadcast_to(gm, gm_shape) if gm_shape != shape else gm)
    _write(tmp_path / "sub-01_label-WM_probseg_aslspace.nii.gz",
           np.broadcast_to(wm, gm_shape) if gm_shape != shape else wm)
    raw = tmp_path / "raw"; raw.mkdir()
    series = np.zeros((*shape, 8)); series[..., 0::2] = 500.0; series[..., 1::2] = 470.0
    _write(raw / "ASL.nii.gz", series)
    _write(raw / "M0.nii.gz", np.ones((*shape, 2)) * 900, tr=6.0)
    return tmp_path


def test_a_derivatives_folder_grades_both_streams(tmp_path):
    """The whole point. Before this, a folder like the mentor's decided 3 of 17
    checks; the CBF map, the tissue maps and the raw series were all in it."""
    got = load_folder(str(_derivatives_folder(tmp_path)))
    assert got["cbf"] is not None and "gm" in got and "wm" in got
    assert got["cbf_path"].endswith("meancbf.nii")
    assert got["affine"] is not None           # QEI-Net resamples; needs this
    assert got["csf_derived"] is True          # no CSF map supplied
    assert got["asl_4d"].shape[3] == 8         # the series, not the 3-D map
    assert got["m0_tr_s"] == pytest.approx(6.0)
    assert got["m0_tr_source"].startswith("NIfTI header")

    rep = run_qc(got, cfg=QCConfig())
    decided = {r.check for r in rep.results
               if r.verdict in (Verdict.PASS, Verdict.WARN, Verdict.FAIL)}
    for need in ("1.qei", "3.1.cbf_level", "3.2.gm_wm_ratio", "5.2.volume_integrity",
                 "5.3.swap", "6.2.m0_tr"):
        assert need in decided, f"{need} was not decided on a full derivatives folder"


def test_a_tissue_map_on_the_wrong_grid_is_reported_not_raised(tmp_path):
    """A folder can hold a good ASL series next to tissue maps still in T1
    space. load_cbf_inputs raises on that, which is right when a caller named
    the files; here it would lose the Stream A findings too."""
    folder = _derivatives_folder(tmp_path, gm_shape=(12, 12, 8))
    _write(folder / "sub-01_label-GM_probseg_aslspace.nii.gz", np.ones((20, 20, 10)))
    got = load_folder(str(folder))
    assert "cbf" not in got
    assert "resample" in got["stream_b_skipped"]
    # and the reader is told, rather than reading "needs a CBF map" about a map
    # that was in the folder
    rep = run_qc(got, cfg=QCConfig())
    r = next(r for r in rep.results if r.check == "8.2.data_type")
    assert "CBF map found but not graded" in r.reason
    # Stream A survived
    assert any(r.check == "5.2.volume_integrity" and r.verdict is Verdict.PASS
               for r in rep.results)


def test_paths_are_still_reported_without_loading_arrays(tmp_path):
    got = load_folder(str(_derivatives_folder(tmp_path)), load_arrays=False)
    assert got["cbf_path"].endswith("meancbf.nii")
    assert "cbf" not in got


# --------------------------------------------------------------------------- #
# the real dataset this was all found on
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not MENTOR.exists(), reason="mentor dataset not in this checkout")
def test_the_mentor_dataset_grades_both_streams():
    got = load_folder(str(MENTOR))
    rep = run_qc(got, cfg=QCConfig())
    cov = rep.coverage
    assert cov["graded"] >= 13, f"only {cov['graded']} decided: {cov['missing']}"

    by = {r.check: r for r in rep.results}
    # Stream B, from the derivative maps
    assert by["1.qei"].verdict is Verdict.PASS
    assert by["3.2.gm_wm_ratio"].verdict is Verdict.PASS
    assert 1.4 < by["3.2.gm_wm_ratio"].metric["gm_wm_ratio"] < 1.7
    assert 35 < by["3.1.cbf_level"].metric["mean_gm_cbf"] < 40
    # Stream A, from raw/ - none of which was reachable before
    assert by["5.2.volume_integrity"].verdict is Verdict.PASS
    assert by["6.2.m0_tr"].verdict is Verdict.PASS
    assert by["6.2.m0_tr"].metric["tr_seconds"] == pytest.approx(6.0)
    assert by["7.1.motion"].metric.get("mean_dvars") is not None
