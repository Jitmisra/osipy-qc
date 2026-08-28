"""
The upload path, exercised with the body a REAL BROWSER sends.

Why this file exists, stated plainly so the mistake is not repeated:

A previous fix made the check set follow the supplied inputs, so that a CBF map
uploaded alone would be graded by the 9 checks that can see it rather than all 19.
It was tested with a hand-built multipart body and reported as working. It was
not working. An unfilled `<input type=file>` still submits a part - empty filename,
empty body - and the form carries two of them (the file picker and the folder
picker). The hand-built test body omitted them; every real browser sends them.

So the fix was inert in exactly the situation it was written for, and the suite
was green. The distinguishing input was the one the test did not have.

Every test here therefore builds the body the way a browser does, including the
empty parts. `_browser_body` is the point of the file.
"""

from __future__ import annotations

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from osipy_qc import web
from osipy_qc.synth import synthetic_case

BOUNDARY = "----osipyqctest"


def _part(name: str, filename: str | None = None, data: bytes = b"") -> bytes:
    head = f'form-data; name="{name}"'
    if filename is not None:
        head += f'; filename="{filename}"'
    return (b"--" + BOUNDARY.encode() + b"\r\nContent-Disposition: "
            + head.encode() + b"\r\n\r\n" + data + b"\r\n")


def _nifti_bytes(arr: np.ndarray) -> bytes:
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "x.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32),
                                 np.diag([3., 3., 3., 1.])), p)
        with open(p, "rb") as fh:
            return fh.read()


def _browser_body(files: dict[str, bytes], *, population: str = "adult",
                  raw: dict[str, bytes] | None = None,
                  empty_file_inputs: int = 2) -> bytes:
    """A multipart body shaped like a real browser's.

    `files` are the single-field uploads, keyed by form field (cbf, gm, wm, csf).
    `raw` are the multi-file raw acquisition parts, keyed by the FILENAME, because
    the filename is what routes them - it is all classify_role has to go on.

    `empty_file_inputs` is the whole point: the form has two file inputs that a
    user filling in only the CBF field leaves untouched, and the browser submits
    both of them as empty parts regardless. Defaulting to 2 means a test cannot
    accidentally exercise the easier, unrealistic path.
    """
    body = b""
    for field, blob in files.items():
        body += _part(field, f"{field}.nii.gz", blob)
    for name, blob in (raw or {}).items():
        body += _part("files", name, blob)
    body += _part("population", data=population.encode())
    for _ in range(empty_file_inputs):
        body += _part("files", "", b"")
    return body + b"--" + BOUNDARY.encode() + b"--\r\n"


def _grade(body: bytes) -> dict:
    fields = web._parse_multipart(body, f"multipart/form-data; boundary={BOUNDARY}")
    out = web._grade_upload(fields)
    return out[0] if isinstance(out, tuple) else out


@pytest.fixture(scope="module")
def clean_case():
    return synthetic_case(quality="clean", seed=0)


@pytest.fixture(scope="module")
def raw_series():
    """A 4-D control/label series as NIfTI bytes - a raw acquisition, no CBF map.

    Named PCASL.nii.gz because the filename is the only routing signal the raw
    zone has, and that is the name two of the three real datasets use.
    """
    from osipy_qc.synth import synthetic_control_label
    return _nifti_bytes(synthetic_control_label(n_pairs=8))


# --------------------------------------------------------------------------
# The parser: empty parts must not be mistaken for uploaded files
# --------------------------------------------------------------------------

def test_empty_file_inputs_are_not_collected(clean_case):
    """The bug, at its source. Two unfilled inputs arrived as
    [('', b''), ('', b'')], which downstream became a 0-byte file on disk and a
    phantom 'raw acquisition folder'."""
    body = _browser_body({"cbf": _nifti_bytes(clean_case.cbf)})
    fields = web._parse_multipart(body, f"multipart/form-data; boundary={BOUNDARY}")
    assert not fields.get("files_multi"), (
        "an unfilled file input was collected as an uploaded file")


def test_a_real_file_part_is_still_collected(clean_case):
    """The guard must reject empty parts without rejecting real ones."""
    body = (_part("cbf", "cbf.nii.gz", _nifti_bytes(clean_case.cbf))
            + _part("files", "PCASL.nii.gz", _nifti_bytes(clean_case.cbf))
            + _part("files", "", b"")                       # the phantom
            + b"--" + BOUNDARY.encode() + b"--\r\n")
    fields = web._parse_multipart(body, f"multipart/form-data; boundary={BOUNDARY}")
    collected = fields.get("files_multi") or []
    assert len(collected) == 1
    assert collected[0][0] == "PCASL.nii.gz"


# --------------------------------------------------------------------------
# The consequence: the check set, and the fabricated warnings
# --------------------------------------------------------------------------

def test_cbf_only_upload_runs_only_the_cbf_map_checks(clean_case):
    from osipy_qc.batch import cbf_map_checks
    data = _grade(_browser_body({"cbf": _nifti_bytes(clean_case.cbf)}))
    # the upload payload is the camelCase shape the React app consumes
    assert data["nChecks"] == len(cbf_map_checks()), (
        f"expected the {len(cbf_map_checks())}-check CBF-map set, "
        f"got {data['nChecks']} - the input-driven check set is inert again")


def test_cbf_only_upload_does_not_fabricate_raw_warnings(clean_case):
    """The mentor's report said 'Needs attention: Schema, M0 present' for an
    upload containing no raw acquisition files at all. Those two WARNs were
    manufactured by the phantom folder, not measured."""
    data = _grade(_browser_body({"cbf": _nifti_bytes(clean_case.cbf)}))
    ids = {c["id"]: c["verdict"] for c in data["checks"]}
    assert "5.1.schema" not in ids, "schema graded an upload with no raw files"
    assert "6.1.m0_present" not in ids, "M0 presence graded an upload with no M0"
    assert data["summary"].get("WARN", 0) == 0, f"fabricated WARNs: {data['summary']}"


def test_good_cbf_map_alone_is_not_warn(clean_case):
    """The finding as the mentor phrased it: 'This is a good CBF map. It should
    not be marked as Warn.'"""
    data = _grade(_browser_body({"cbf": _nifti_bytes(clean_case.cbf)}))
    assert data["verdict"] != "WARN", (
        f"a good CBF map was graded WARN: {data['summary']}")


def test_good_map_with_tissue_maps_passes_through_the_browser_path(clean_case):
    """Not a tautology like the assertion this file replaces: an exact verdict."""
    c = clean_case
    data = _grade(_browser_body({
        "cbf": _nifti_bytes(c.cbf), "gm": _nifti_bytes(c.gm),
        "wm": _nifti_bytes(c.wm), "csf": _nifti_bytes(c.csf),
    }))
    assert data["verdict"] == "PASS", f"{data['verdict']} {data['summary']}"
    assert data["coverage"]["complete"] is True


def test_coverage_travels_with_the_verdict(clean_case):
    """Since UNKNOWN stopped escalating, this is the only thing carrying 'the
    report is partial' - so the upload response must contain it."""
    data = _grade(_browser_body({"cbf": _nifti_bytes(clean_case.cbf)}))
    assert "coverage" in data
    assert data["coverage"]["complete"] is False
    # 3.5.brain_cbf is the one check a lone CBF map can grade; the rest have no
    # tissue maps to look at, and coverage is what says so.
    assert data["coverage"]["graded"] == 1
    assert data["coverage"]["unknown"] > 0


# --------------------------------------------------------------------------
# The mirror image: Stream A must not need a CBF map
# --------------------------------------------------------------------------

def test_raw_only_upload_is_graded_at_all(raw_series):
    """The reviewer's finding: "It seems that stream A always needs a CBF map to
    work. This should not be the case."

    /run raised ValueError("No CBF map was uploaded.") before it ever looked at the
    raw files, so the whole raw-acquisition stream was unreachable from the
    browser - the half of the toolbox that grades the acquisition rather than the
    map.
    """
    data = _grade(_browser_body({}, raw={"PCASL.nii.gz": raw_series}))
    ids = {c["id"]: c["verdict"] for c in data["checks"]}
    assert ids.get("5.2.volume_integrity") == "PASS"      # 16 volumes -> 8 pairs
    assert ids.get("5.3.swap") == "PASS"                  # control brighter than label


def test_raw_only_upload_runs_only_the_stream_a_checks(raw_series):
    """The check set follows the inputs in this direction too. Running the whole
    registry would report every CBF-map check as a gap in a report that was never
    given a CBF map."""
    from osipy_qc.core.registry import all_checks
    # scoped to brain: the registry also holds the kidney and placenta checks,
    # and the upload console grades brain
    stream_a = [n for n, e in all_checks("brain").items() if e.get("stream") == "A"]
    data = _grade(_browser_body({}, raw={"PCASL.nii.gz": raw_series}))
    assert data["nChecks"] == len(stream_a)
    assert [c["id"] for c in data["checks"] if c["stream"] == "B"] == []
    # and no other organ's checks leak into a brain report - a kidney check
    # given brain inputs would return UNKNOWN and read as a gap in coverage
    assert [c["id"] for c in data["checks"] if c["id"].startswith(("k", "p"))] == []


def test_raw_only_coverage_does_not_count_the_cbf_checks_as_missing(raw_series):
    """coverage() is the only thing left carrying "this report is partial", so it
    has to be right for a raw-only upload too: what is missing is what a raw file
    could have supplied, not what the user never asked for."""
    data = _grade(_browser_body({}, raw={"PCASL.nii.gz": raw_series}))
    cov = data["coverage"]
    assert cov["graded"] > 0, "a raw-only upload graded nothing at all"
    assert not [m for m in cov["missing"] if m.startswith(("1.", "2.", "3.", "4."))], (
        f"CBF-map checks reported as gaps in a raw-only report: {cov['missing']}")


def test_raw_only_upload_renders_the_html_report(raw_series):
    """Without a CBF map the report has to degrade, not crash: no mosaic, no KPI
    tiles, and nothing advertised that cannot be drawn - figure_bytes() raises
    KeyError with no CBF map, so a listed figure would 404."""
    fields = web._parse_multipart(_browser_body({}, raw={"PCASL.nii.gz": raw_series}),
                                  f"multipart/form-data; boundary={BOUNDARY}")
    html, token = web._grade_upload_html(fields)
    assert html.startswith("<!DOCTYPE html>")
    assert "5.3.swap" in html                        # the Stream A cards are there
    assert web._UPLOADS[token].inputs.get("cbf") is None
    assert web._grade_upload(fields)["figures"] == []


def test_raw_and_cbf_together_still_run_both_streams(clean_case, raw_series):
    """The fix must not turn a full upload into a one-stream report."""
    data = _grade(_browser_body({"cbf": _nifti_bytes(clean_case.cbf),
                                 "gm": _nifti_bytes(clean_case.gm),
                                 "wm": _nifti_bytes(clean_case.wm)},
                                raw={"PCASL.nii.gz": raw_series}))
    assert {c["stream"] for c in data["checks"]} == {"A", "B"}


def test_an_empty_upload_says_what_may_be_uploaded():
    """The old message named only the CBF map, which is the misconception itself."""
    with pytest.raises(ValueError) as exc:
        _grade(_browser_body({}))
    assert "Nothing to grade" in str(exc.value)
    assert "raw acquisition" in str(exc.value)


def test_tissue_maps_without_a_cbf_map_are_not_silently_dropped(clean_case):
    """Silently discarding an uploaded file is the bug class this review is about.
    Every Stream-B check reads the CBF map, so tissue maps alone are ungradeable,
    and the upload has to say so rather than return a report that omits them."""
    with pytest.raises(ValueError) as exc:
        _grade(_browser_body({"gm": _nifti_bytes(clean_case.gm),
                              "wm": _nifti_bytes(clean_case.wm)}))
    assert "without a CBF map" in str(exc.value)


# --------------------------------------------------------------------------
# BIDS sidecars must survive the upload round-trip (OSIPI Challenge data)
# --------------------------------------------------------------------------

def test_sidecars_survive_the_upload_round_trip(raw_series):
    """Three blockers used to stand between an uploaded sidecar and load_folder:
    the folder-picker JS dropped non-NIfTIs, the accept attribute refused them,
    and the server renamed sub-01_asl.json to sub-01_asl.json.nii.gz - hiding the
    metadata AND handing JSON bytes to nib.load. This exercises the server half
    with the body a browser sends."""
    import json as _j
    asl_json = _j.dumps({"Manufacturer": "Philips", "MRAcquisitionType": "2D",
                         "ArterialSpinLabelingType": "PCASL",
                         "PostLabelingDelay": 1.8}).encode()
    m0_json = _j.dumps({"RepetitionTimePreparation": 10.0}).encode()
    ctx = ("volume_type\n" + "control\nlabel\n" * 8).encode()
    data = _grade(_browser_body({}, raw={
        "sub-01_asl.nii.gz": raw_series,
        "sub-01_asl.json": asl_json,
        "sub-01_m0scan.json": m0_json,
        "sub-01_aslcontext.tsv": ctx,
    }))
    ids = {c["id"]: c for c in data["checks"]}
    assert ids["5.1.schema"]["verdict"] == "PASS", (
        f"sidecar did not reach the grader: {ids['5.1.schema']['reason']}")
    assert ids["6.2.m0_tr"]["verdict"] == "PASS"       # TR 10s from m0scan.json
    assert "Philips" in ids["8.2.data_type"]["reason"]
    assert ids["5.2.volume_integrity"]["verdict"] == "PASS"   # 16 rows == 16 vols


def test_aslcontext_that_contradicts_the_series_fails(raw_series):
    """A context file listing a different volume count than the series holds is a
    truncated export or the wrong file - pairing cannot be trusted."""
    ctx = ("volume_type\n" + "control\nlabel\n" * 5).encode()   # 10 rows, 16 vols
    data = _grade(_browser_body({}, raw={
        "sub-01_asl.nii.gz": raw_series,
        "sub-01_aslcontext.tsv": ctx,
    }))
    ids = {c["id"]: c for c in data["checks"]}
    assert ids["5.2.volume_integrity"]["verdict"] == "FAIL"
    assert "aslcontext" in ids["5.2.volume_integrity"]["reason"]


def test_uppercase_sidecar_does_not_kill_the_upload(raw_series):
    """The JS folder filter admits SUB01_ASL.JSON case-insensitively, but the
    server's extension check was case-sensitive: it appended .nii.gz, nib.load
    then crashed on JSON bytes, and the whole upload died. The extension is now
    matched case-insensitively and normalised to lowercase."""
    import json as _j
    asl_json = _j.dumps({"Manufacturer": "Philips", "MRAcquisitionType": "2D",
                         "ArterialSpinLabelingType": "PCASL",
                         "PostLabelingDelay": 1.8}).encode()
    data = _grade(_browser_body({}, raw={
        "sub-01_asl.nii.gz": raw_series,
        "SUB01_ASL.JSON": asl_json,
    }))
    ids = {c["id"]: c for c in data["checks"]}
    assert ids["5.1.schema"]["verdict"] == "PASS", ids["5.1.schema"]["reason"]
