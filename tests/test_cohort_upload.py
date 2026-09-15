"""Grading a whole cohort through the browser.

The package could already grade a cohort, with `--dashboard FOLDER`. That reads
a folder on the SERVER'S disk and renders through the React app, so neither half
is available to the person the deployed site is actually for: someone with a
folder on their own laptop and a browser. They could grade one scan at a time,
and nothing on the page said otherwise.

Three things had to be true for a folder of subjects to survive the upload:

  * the directory structure has to be kept. Uploads were flattened to their
    basenames, which is harmless for one subject and destroys a cohort - every
    subject ships a `cbf.nii.gz`, so sub-02's landed on top of sub-01's and the
    "cohort" became one scan assembled from several people.
  * "is this a cohort?" has to mean "are there two or more SUBJECTS here", not
    "are there two or more folders". A BIDS subject is `sub-01/anat` +
    `sub-01/perf`, which is two folders and one person.
  * the answer has to be reachable without loading every array twice.
"""

from __future__ import annotations

import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from osipy_qc import web
from osipy_qc.batch import grade_folder, subject_dirs, summarise
from osipy_qc.core import Verdict
from osipy_qc.report_html import render_cohort_html
from osipy_qc.synth import synthetic_case

BOUNDARY = "----osipyqctest"


def _nifti_bytes(arr) -> bytes:
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "x.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32),
                                 np.diag([3., 3., 3., 1.])), p)
        with open(p, "rb") as fh:
            return fh.read()


def _write(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32),
                             np.diag([3., 3., 3., 1.])), str(path))


def _subject_folder(root, sid, quality="clean", seed=0, with_raw=False):
    c = synthetic_case(quality=quality, seed=seed)
    d = os.path.join(root, sid)
    _write(os.path.join(d, f"{sid}_cbf.nii.gz"), c.cbf)
    _write(os.path.join(d, f"{sid}_label-GM_probseg.nii.gz"), c.gm)
    _write(os.path.join(d, f"{sid}_label-WM_probseg.nii.gz"), c.wm)
    if with_raw:
        from osipy_qc.synth import synthetic_control_label
        _write(os.path.join(d, "raw", "PCASL.nii.gz"), synthetic_control_label(n_pairs=4))
    return d


# --------------------------------------------------------------------------- #
# what counts as a subject
# --------------------------------------------------------------------------- #
def test_two_subject_folders_are_a_cohort(tmp_path):
    _subject_folder(str(tmp_path), "sub-01")
    _subject_folder(str(tmp_path), "sub-02", seed=1)
    assert subject_dirs(str(tmp_path)) == ["sub-01", "sub-02"]


def test_a_bids_subject_is_one_person_not_a_two_subject_cohort(tmp_path):
    """`sub-01/anat` + `sub-01/perf` is two folders and one scan.

    Counting folders would read it as a cohort of two, and the upload console
    would answer a single-subject upload with a two-row ledger.
    """
    c = synthetic_case(quality="clean", seed=0)
    _write(tmp_path / "anat" / "sub-01_T1w.nii.gz", c.brain.astype(float))
    _write(tmp_path / "perf" / "sub-01_cbf.nii.gz", c.cbf)
    assert subject_dirs(str(tmp_path)) == ["perf"], (
        "a folder holding only a T1 was counted as a subject")


def test_a_folder_with_nothing_gradeable_is_not_a_subject(tmp_path):
    os.makedirs(tmp_path / "notes")
    (tmp_path / "notes" / "readme.txt").write_text("hello")
    _subject_folder(str(tmp_path), "sub-01")
    assert subject_dirs(str(tmp_path)) == ["sub-01"]


def test_the_header_survey_agrees_with_the_full_grade(tmp_path):
    """`subject_dirs` reads headers only so the console can ask "cohort?" cheaply.
    If it disagreed with `grade_folder` the console would promise a cohort and
    then render a different one."""
    _subject_folder(str(tmp_path), "sub-01")
    _subject_folder(str(tmp_path), "sub-02", seed=1)
    _subject_folder(str(tmp_path), "sub-03", quality="garbage", seed=2)
    assert subject_dirs(str(tmp_path)) == [s.sid for s in grade_folder(str(tmp_path))]


# --------------------------------------------------------------------------- #
# per-subject check sets
# --------------------------------------------------------------------------- #
def test_a_cbf_only_cohort_is_not_dragged_down_by_stream_a(tmp_path):
    """Asking a folder of bare CBF maps for an M0 adds ten UNKNOWNs per subject
    and pushes a clean cohort to WARN."""
    _subject_folder(str(tmp_path), "sub-01")
    _subject_folder(str(tmp_path), "sub-02", seed=1)
    for s in grade_folder(str(tmp_path)):
        ran = {r.check for r in s.report.results}
        assert not any(c.startswith(("5.", "6.", "7.")) for c in ran), (
            f"{s.sid} was asked for raw-data checks it has no files for: {sorted(ran)}")


def test_a_subject_that_shipped_its_raw_series_gets_stream_a_too(tmp_path):
    """The other direction. The checks follow each subject's own files, so two
    subjects in one cohort can legitimately be graded on different sets."""
    _subject_folder(str(tmp_path), "sub-01")
    _subject_folder(str(tmp_path), "sub-02", seed=1, with_raw=True)
    by = {s.sid: {r.check for r in s.report.results} for s in grade_folder(str(tmp_path))}
    assert "5.2.volume_integrity" in by["sub-02"]
    assert "5.2.volume_integrity" not in by["sub-01"]
    assert "1.qei" in by["sub-01"] and "1.qei" in by["sub-02"]


def test_an_explicit_check_list_still_forces_one_set_across_the_cohort(tmp_path):
    """Which is what you want when the cohort has to be compared column by column."""
    _subject_folder(str(tmp_path), "sub-01")
    _subject_folder(str(tmp_path), "sub-02", seed=1, with_raw=True)
    subs = grade_folder(str(tmp_path), checks=["1.qei"])
    assert all({r.check for r in s.report.results} == {"1.qei"} for s in subs)


# --------------------------------------------------------------------------- #
# uploaded paths
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sent,expected", [
    ("cohort/sub-01/perf/cbf.nii.gz", "sub-01/perf/cbf.nii.gz"),
    ("cohort/sub-02/cbf.nii.gz", "sub-02/cbf.nii.gz"),
    ("cbf.nii.gz", "cbf.nii.gz"),
    # the reason this function exists rather than os.path.join
    ("../../etc/passwd", "etc/passwd"),
    ("a/../../../b/c.nii", "a/b/c.nii"),
    ("cohort/x/y/z/w/deep.nii.gz", "z/w/deep.nii.gz"),   # depth capped
    ("cohort/../secret.nii", "secret.nii"),
])
def test_uploaded_paths_cannot_climb_out_of_the_upload_directory(sent, expected):
    got = web._safe_relpath(sent, "cohort")
    assert got == expected
    assert ".." not in got.split("/")
    assert not got.startswith("/")


def test_a_path_that_sanitises_to_nothing_still_gets_a_name():
    assert web._safe_relpath("../..", "cohort") == "upload.nii.gz"


# --------------------------------------------------------------------------- #
# the rendered cohort page
# --------------------------------------------------------------------------- #
def _cohort(tmp_path, n=3):
    for i in range(n):
        _subject_folder(str(tmp_path), f"sub-{i:02d}",
                        quality=("clean", "garbage", "borderline")[i % 3], seed=i)
    subs = grade_folder(str(tmp_path))
    return subs, summarise(subs)


def test_the_cohort_page_is_self_contained(tmp_path):
    """Same promise the single-scan report makes: one file, openable offline,
    attachable to an email."""
    import re
    subs, summary = _cohort(tmp_path)
    html = render_cohort_html(subs, summary, dataset="test")
    assert not re.search(r'(?:src|href)="https?://', html), "the page fetches something"
    assert html.count('src="data:image/') >= 1, "figures are not embedded"


def test_every_subject_appears_once_in_the_ledger_and_once_in_full(tmp_path):
    subs, summary = _cohort(tmp_path)
    html = render_cohort_html(subs, summary)
    assert html.count('class="subj"') == len(subs)
    for s in subs:
        assert s.sid in html


def test_the_worst_subject_is_listed_first(tmp_path):
    """The ledger is a triage tool. Alphabetical order buries the failure."""
    subs, summary = _cohort(tmp_path)
    html = render_cohort_html(subs, summary)
    worst = next(s for s in subs if s.overall == "FAIL")
    others = [s for s in subs if s.overall != "FAIL"]
    assert others, "fixture no longer has a mix of verdicts"
    assert all(html.index(worst.sid) < html.index(o.sid) for o in others)


def test_figures_are_dropped_for_a_big_cohort_and_the_page_says_so(tmp_path):
    """Four mosaics each would run an eighty-subject page into the tens of MB."""
    subs, summary = _cohort(tmp_path)
    small = render_cohort_html(subs, summary, with_figures=True)
    big = render_cohort_html(subs, summary, with_figures=False)
    assert len(big) < len(small)
    assert "images are omitted above" in big
    assert "images are omitted above" not in small


# --------------------------------------------------------------------------- #
# end to end through the upload handler
# --------------------------------------------------------------------------- #
def _part(name, filename=None, data=b""):
    head = f'form-data; name="{name}"'
    if filename is not None:
        head += f'; filename="{filename}"'
    return (b"--" + BOUNDARY.encode() + b"\r\nContent-Disposition: "
            + head.encode() + b"\r\n\r\n" + data + b"\r\n")


def _cohort_body(n=3, root="mycohort"):
    body = b""
    for i in range(n):
        c = synthetic_case(quality=("clean", "garbage", "clean")[i % 3], seed=i)
        sid = f"sub-{i:02d}"
        for kind, arr in (("cbf", c.cbf), ("label-GM_probseg", c.gm),
                          ("label-WM_probseg", c.wm)):
            body += _part("files", f"{root}/{sid}/{sid}_{kind}.nii.gz", _nifti_bytes(arr))
    body += _part("population", data=b"adult")
    for _ in range(2):                       # the empty inputs a browser always sends
        body += _part("files", "", b"")
    return body + b"--" + BOUNDARY.encode() + b"--\r\n"


def _grade(body):
    fields = web._parse_multipart(body, f"multipart/form-data; boundary={BOUNDARY}")
    return web._grade_upload(fields)


def test_an_uploaded_cohort_is_graded_as_a_cohort(tmp_path):
    out = _grade(_cohort_body(3))
    assert out.get("cohort") is True
    assert len(out["subjects"]) == 3
    assert {r["sid"] for r in out["subjects"]} == {"sub-00", "sub-01", "sub-02"}


def test_subjects_do_not_overwrite_each_other(tmp_path):
    """The flattening bug, stated as a measurement rather than a path.

    Every subject ships the same basenames. If the structure were lost they
    would collapse onto one file and the three subjects would be graded on one
    person's data - so their QEIs would be identical.
    """
    out = _grade(_cohort_body(3))
    qeis = [r["qei"] for r in out["subjects"] if r["qei"] is not None]
    assert len(set(qeis)) > 1, f"every subject graded identically: {qeis}"


def test_one_subject_uploaded_as_a_folder_is_still_a_single_report(tmp_path):
    """The detection must not turn an ordinary single-subject upload into a
    one-row cohort table."""
    out = _grade(_cohort_body(1))
    assert not out.get("cohort")
    assert out.get("uploaded") is True


def test_the_cohort_html_comes_back_from_the_html_path(tmp_path):
    fields = web._parse_multipart(_cohort_body(3),
                                  f"multipart/form-data; boundary={BOUNDARY}")
    html, token = web._grade_upload_html(fields)
    assert "subjects graded" in html
    assert html.count('class="subj"') == 3


def test_too_many_subjects_is_refused_with_a_reason(monkeypatch):
    """The limit protects real memory: every subject's arrays are held at once
    so its figures can be drawn."""
    monkeypatch.setattr(web, "MAX_COHORT_SUBJECTS", 2)
    with pytest.raises(ValueError, match="limit is 2"):
        _grade(_cohort_body(3))
