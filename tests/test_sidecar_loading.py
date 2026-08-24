"""Regression tests for BIDS sidecar loading (`_find_sidecars` + load_folder).

Written against the OSIPI ASL Challenge dataset, the first data with metadata:
before these, load_folder never read the sidecars and inferred a Philips 2D
acquisition as "unknown 3D" while `MRAcquisitionType: "2D"` sat unread in the
folder. Stated metadata must beat inferred metadata.
"""

import json

import nibabel as nib
import numpy as np

from osipy_qc.checks.m0 import m0_no_bs_check
from osipy_qc.checks.schema import data_type_check
from osipy_qc.core import Verdict
from osipy_qc.io import load_folder


def _write_nifti(tmp_path, name, shape, voxel=(3.4, 3.4, 6.0)):
    aff = np.diag(list(voxel) + [1.0])
    arr = np.random.default_rng(0).normal(100.0, 5.0, shape).astype(np.float32)
    nib.save(nib.Nifti1Image(arr, aff), str(tmp_path / name))


def _challenge_like_folder(tmp_path):
    """Mimic sub-PopulationAverage: 4D ASL + M0 + asl.json + m0scan.json + aslcontext."""
    _write_nifti(tmp_path, "sub-X_asl.nii.gz", (8, 8, 4, 6))
    _write_nifti(tmp_path, "sub-X_m0scan.nii.gz", (8, 8, 4))
    (tmp_path / "sub-X_asl.json").write_text(json.dumps({
        "Manufacturer": "Philips",
        "MRAcquisitionType": "2D",
        "ArterialSpinLabelingType": "PCASL",
        "PostLabelingDelay": 1.8,
        "BackgroundSuppression": True,
        "M0Type": "Separate",
    }))
    # the real challenge m0scan.json states TR but NOT BackgroundSuppression
    (tmp_path / "sub-X_m0scan.json").write_text(json.dumps({
        "Manufacturer": "Philips",
        "RepetitionTimePreparation": 10.0,
    }))
    (tmp_path / "sub-X_aslcontext.tsv").write_text(
        "volume_type\n" + "control\nlabel\n" * 3)
    return tmp_path


def test_stated_metadata_beats_inference(tmp_path):
    inp = load_folder(str(_challenge_like_folder(tmp_path)), load_arrays=False)
    det = inp["detected"]
    # 6 mm slices would make the shape-based guess say 2D here, but the point is
    # the value must come from the sidecar, and the source must say so
    assert det["vendor"] == "Philips"
    assert det["readout"] == "2D"
    assert det["background_suppression"] is True
    assert det["labelling"] == "PCASL"
    assert det["m0"] == "separate"
    assert det["source"] == "BIDS sidecar"
    assert inp["sidecar"]["PostLabelingDelay"] == 1.8


def test_m0_sidecar_yields_tr_but_not_bs(tmp_path):
    inp = load_folder(str(_challenge_like_folder(tmp_path)), load_arrays=False)
    assert inp["m0_tr_s"] == 10.0                    # RepetitionTimePreparation
    # BS absent from the M0 sidecar must stay absent - NOT inherited from the
    # ASL sidecar's true, because the rules are opposite (M0 needs BS OFF)
    assert "m0_background_suppression" not in inp
    r = m0_no_bs_check(**inp)
    assert r.verdict == Verdict.UNKNOWN
    assert "not inherited from the ASL sidecar" in r.reason


def test_m0_no_bs_reason_distinguishes_no_metadata_at_all():
    r = m0_no_bs_check()
    assert r.verdict == Verdict.UNKNOWN
    assert "no M0 metadata" in r.reason


def test_aslcontext_rows_loaded(tmp_path):
    inp = load_folder(str(_challenge_like_folder(tmp_path)), load_arrays=False)
    assert len(inp["aslcontext_rows"]) == 6          # header row stripped
    assert inp["aslcontext_rows"][0] == "control"
    # n_volumes is deliberately NOT set from the tsv: 5.2 derives its counts
    # from the rows itself, and only against an actual series
    assert "n_volumes" not in inp


def test_malformed_sidecar_degrades_not_crashes(tmp_path):
    _write_nifti(tmp_path, "sub-X_asl.nii.gz", (8, 8, 4, 6))
    (tmp_path / "sub-X_asl.json").write_text("{not valid json")
    inp = load_folder(str(tmp_path), load_arrays=False)
    assert inp["detected"]["source"] == "inferred from NIfTI shape + filenames"
    assert "sidecar" not in inp


def test_no_sidecar_keeps_inference_path(tmp_path):
    _write_nifti(tmp_path, "Siemens_2D_PCASL.nii.gz", (8, 8, 4, 6))
    inp = load_folder(str(tmp_path), load_arrays=False)
    assert inp["detected"]["vendor"] == "Siemens"    # from the filename, as before
    assert inp["detected"]["source"] == "inferred from NIfTI shape + filenames"


def test_data_type_check_prefers_loader_detected(tmp_path):
    inp = load_folder(str(_challenge_like_folder(tmp_path)), load_arrays=False)
    r = data_type_check(**inp)
    assert r.verdict == Verdict.INFO
    # must report the sidecar-corrected values, not re-derive its own guess
    assert "Philips" in r.reason and "2D" in r.reason and "BIDS sidecar" in r.reason


def test_data_type_check_still_derives_without_detected():
    files = [{"name": "GE_3D_pcasl.nii.gz", "shape": (64, 64, 36)}]
    r = data_type_check(files=files, context="")
    assert r.verdict == Verdict.INFO
    assert "GE" in r.reason


# --------------------------------------------------------------------------- #
# Findings from the adversarial review of the first sidecar implementation.
# Every test here reproduces a defect that was demonstrated by execution.
# --------------------------------------------------------------------------- #

def test_string_suppression_value_does_not_disable_swap(tmp_path):
    """BackgroundSuppression: "no" is a truthy string; untyped it switched the
    required swap check off with a reason claiming BS was ON.

    (Named without 'bs': pytest puts the test name in tmp_path, and a folder
    called test_bs_* trips the filename heuristic this test must isolate.)"""
    _write_nifti(tmp_path, "sub-X_asl.nii.gz", (8, 8, 4, 6))
    (tmp_path / "sub-X_asl.json").write_text(json.dumps(
        {"ArterialSpinLabelingType": "PCASL", "BackgroundSuppression": "no"}))
    inp = load_folder(str(tmp_path), load_arrays=False)
    # wrong type degrades to "not stated", it does not become truthy
    assert inp["detected"]["background_suppression"] is None
    assert inp["background_suppression"] is None


def test_boolean_tr_does_not_fabricate_one_second(tmp_path):
    """bool is an int in Python: RepetitionTimePreparation: true produced a
    confident m0_tr_s = 1.0 and a correction factor computed from a type error."""
    _write_nifti(tmp_path, "sub-X_asl.nii.gz", (8, 8, 4, 6))
    _write_nifti(tmp_path, "sub-X_m0scan.nii.gz", (8, 8, 4))
    (tmp_path / "sub-X_m0scan.json").write_text(
        json.dumps({"RepetitionTimePreparation": True}))
    inp = load_folder(str(tmp_path), load_arrays=False)
    assert "m0_tr_s" not in inp


def test_tr_array_form_is_reduced_not_dropped(tmp_path):
    """BIDS allows RepetitionTimePreparation as a per-volume array; dropping it
    threw stated metadata away - the exact failure _find_sidecars exists to fix.
    The White Paper rule cares about full relaxation, so the shortest TR grades."""
    _write_nifti(tmp_path, "sub-X_asl.nii.gz", (8, 8, 4, 6))
    _write_nifti(tmp_path, "sub-X_m0scan.nii.gz", (8, 8, 4))
    (tmp_path / "sub-X_m0scan.json").write_text(
        json.dumps({"RepetitionTimePreparation": [10.0, 6.0]}))
    inp = load_folder(str(tmp_path), load_arrays=False)
    assert inp["m0_tr_s"] == 6.0


def test_m0_sidecar_routing_uses_the_shared_vocabulary(tmp_path):
    """M0.json and calib.json are M0 sidecars under classify_role; the ad-hoc
    '"m0scan" in low or "_m0" in low' router dropped both, so a stated TR and
    BS=false never reached checks 6.2/6.3."""
    for name in ("M0", "calib"):
        d = tmp_path / name
        d.mkdir()
        _write_nifti(d, f"{name}.nii.gz", (8, 8, 4))
        _write_nifti(d, "PCASL.nii.gz", (8, 8, 4, 6))
        (d / f"{name}.json").write_text(json.dumps(
            {"RepetitionTimePreparation": 10.0, "BackgroundSuppression": False}))
        inp = load_folder(str(d), load_arrays=False)
        assert inp.get("m0_tr_s") == 10.0, f"{name}.json was not routed as an M0 sidecar"
        assert inp.get("m0_background_suppression") is False


def test_derivative_cbf_json_does_not_shadow_the_real_asl_sidecar(tmp_path):
    """classify_role('cbf.json') is 'asl', and cbf sorts before sub-X, so
    first-wins latched ASLPrep's {"Units": ...} as THE ASL sidecar and 5.1
    WARNed about missing fields a valid asl.json supplied."""
    _write_nifti(tmp_path, "sub-X_asl.nii.gz", (8, 8, 4, 6))
    (tmp_path / "cbf.json").write_text(json.dumps({"Units": "mL/100g/min"}))
    (tmp_path / "sub-X_asl.json").write_text(json.dumps({
        "ArterialSpinLabelingType": "PCASL", "MRAcquisitionType": "2D",
        "PostLabelingDelay": 1.8}))
    inp = load_folder(str(tmp_path), load_arrays=False)
    assert inp["sidecar"]["ArterialSpinLabelingType"] == "PCASL"


def test_multi_subject_folder_pairs_sidecars_with_the_loaded_image(tmp_path):
    """JSON was first-wins, the TSV last-wins: two individually valid subjects in
    one tree graded sub-01's 6-volume image against sub-02's 8-row aslcontext and
    manufactured a 5.2 FAIL out of two clean runs."""
    from osipy_qc.checks.schema import volume_integrity_check
    for sub, n in (("sub-01", 6), ("sub-02", 8)):
        d = tmp_path / sub
        d.mkdir()
        _write_nifti(d, f"{sub}_asl.nii.gz", (8, 8, 4, n))
        (d / f"{sub}_aslcontext.tsv").write_text(
            "volume_type\n" + "control\nlabel\n" * (n // 2))
    inp = load_folder(str(tmp_path), load_arrays=False)
    r = volume_integrity_check(**inp)
    assert r.verdict == Verdict.PASS, r.reason
    assert len(inp["aslcontext_rows"]) == inp["asl_shape"][3]


def test_multi_run_folder_pairs_by_run(tmp_path):
    """Same defect inside one directory: run-1 and run-2 side by side."""
    from osipy_qc.checks.schema import volume_integrity_check
    for run, n in (("run-1", 20), ("run-2", 8)):
        _write_nifti(tmp_path, f"sub-X_{run}_asl.nii.gz", (8, 8, 4, n))
        (tmp_path / f"sub-X_{run}_aslcontext.tsv").write_text(
            "volume_type\n" + "control\nlabel\n" * (n // 2))
    inp = load_folder(str(tmp_path), load_arrays=False)
    r = volume_integrity_check(**inp)
    assert r.verdict == Verdict.PASS, r.reason


def test_m0_included_series_is_not_an_incomplete_pair():
    """Per BIDS an M0Type=Included acquisition lists its m0scan row inside
    aslcontext, so 1 m0scan + 4 pairs is a valid 9-row file - the raw count fed
    to the even/odd test called that odd and FAILed a fully valid dataset."""
    from osipy_qc.checks.schema import volume_integrity_check
    rows = ["m0scan"] + ["control", "label"] * 4
    r = volume_integrity_check(asl_shape=(8, 8, 4, 9), aslcontext_rows=rows)
    assert r.verdict == Verdict.PASS, r.reason
    assert r.metric["n_pairs"] == 4 and r.metric["n_other"] == 1


def test_unequal_control_label_rows_fail():
    from osipy_qc.checks.schema import volume_integrity_check
    rows = ["control", "label", "control"]
    r = volume_integrity_check(asl_shape=(8, 8, 4, 3), aslcontext_rows=rows)
    assert r.verdict == Verdict.FAIL
    assert "unpaired" in r.reason


def test_aslcontext_alone_cannot_pass_the_series():
    """A tsv uploaded with no image graded '6 volumes -> 3 pairs PASS' - a PASS
    asserting the integrity of a series that was never seen."""
    from osipy_qc.checks.schema import volume_integrity_check
    r = volume_integrity_check(aslcontext_rows=["control", "label"] * 3)
    assert r.verdict == Verdict.UNKNOWN
    assert "no ASL series" in r.reason


def test_swap_uses_the_stated_volume_order():
    """A valid label-first acquisition graded FAIL 'likely swap' under the
    even=control heuristic while the true order sat in the same inputs dict -
    with a metric claiming '(no aslcontext.tsv)'."""
    from osipy_qc.checks.schema import swap_check
    rng = np.random.default_rng(0)
    arr = np.empty((6, 6, 4, 6))
    for i in range(6):                       # label-first: odd volumes brighter
        level = 100.0 if i % 2 else 99.0     # even=label (dim), odd=control
        arr[..., i] = level + rng.normal(0, 0.1, (6, 6, 4))
    rows = ["label", "control"] * 3
    r = swap_check(asl_4d=arr, aslcontext_rows=rows)
    assert r.verdict == Verdict.PASS, r.reason
    assert r.metric["assumption"] == "volume order from aslcontext.tsv"
    # and without the rows the heuristic still applies, stated as such
    r2 = swap_check(asl_4d=arr)
    assert r2.verdict == Verdict.FAIL
    assert "no aslcontext.tsv" in r2.metric["assumption"]


def test_data_type_check_survives_a_partial_detected_dict():
    """run_qc documents 'detected' as caller-suppliable; a partial dict
    KeyErrored and demoted INFO to UNKNOWN 'check error'."""
    files = [{"name": "GE_3D_pcasl.nii.gz", "shape": (64, 64, 36)}]
    r = data_type_check(files=files, detected={"structure": "pre-subtracted deltaM"})
    assert r.verdict == Verdict.INFO
    assert "GE" in r.reason                  # re-derived around the gap


def test_motion_reason_names_too_few_volumes_not_nonfinite():
    """A (X,Y,Z,1) NIfTI with a NaN slab was misdiagnosed as '4D series is
    entirely non-finite' when the true cause is a singleton time axis."""
    from osipy_qc.checks.motion import motion_check
    vol = np.full((6, 6, 6, 1), 100.0)
    vol[0, :, :, :] = np.nan
    r = motion_check(asl_4d=vol)
    assert r.verdict == Verdict.UNKNOWN
    assert "fewer than 2 volumes" in r.reason
    assert "entirely non-finite" not in r.reason
