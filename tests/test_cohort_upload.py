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


# --------------------------------------------------------------------------- #
# the client-side half
# --------------------------------------------------------------------------- #
# The page detects a cohort in the browser, before anything is uploaded, so it
# can say "3 subjects found" and flip the mode for you. That logic is JavaScript
# and the rest of this suite cannot see it, so these run it under node - skipped
# where node is absent rather than left as an untested claim.

def _node() -> str | None:
    import shutil
    return shutil.which("node")


def _page_js() -> str:
    import re
    return "\n".join(re.findall(r"<script[^>]*>(.*?)</script>",
                                web._upload_page(), flags=re.S))


def _extract_counter() -> str:
    """`subjectCount` plus what it needs: the shipped rule table and roleKey.

    It deliberately reuses the page's own filename vocabulary rather than
    carrying a second copy, so the pieces cannot be tested apart.
    """
    import re
    js = _page_js()
    out = []
    for pat in (r"var ROLES = \[.*?\];", r"var LABELS = \{.*?\};",
                r"function roleKey\(n\)\{.*?\n  \}",
                r"function subjectCount\(files\)\{.*?\n  \}"):
        m = re.search(pat, js, flags=re.S)
        assert m, f"the page no longer contains {pat}"
        out.append(m.group(0))
    return "\n".join(out)


@pytest.mark.skipif(not _node(), reason="node not available")
def test_the_generated_javascript_parses():
    """The page is built by string interpolation, so a stray brace ships a page
    whose script silently does nothing - no mode buttons, no detection, no
    upload progress - and every server-side test still passes."""
    import re
    import subprocess
    import tempfile
    js = "\n".join(re.findall(r"<script[^>]*>(.*?)</script>", web._upload_page(), flags=re.S))
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(js)
        path = fh.name
    out = subprocess.run([_node(), "--check", path], capture_output=True, text=True)
    assert out.returncode == 0, f"generated JS does not parse:\n{out.stderr}"


@pytest.mark.skipif(not _node(), reason="node not available")
@pytest.mark.parametrize("label,paths,expected", [
    ("a real cohort", ["c/sub-01/sub-01_cbf.nii.gz", "c/sub-02/sub-02_cbf.nii.gz",
                       "c/sub-03/sub-03_cbf.nii.gz"], 3),
    # THE bug this rule exists for. A BIDS subject is `sub-01/anat` + `sub-01/perf`:
    # two folders, one person. Counting folders read it as a 2-subject cohort,
    # auto-flipped the mode, disabled the CBF box, and dropped the map the user had
    # already chosen from the upload without a word.
    ("one BIDS subject", ["sub-01/anat/sub-01_T1w.nii.gz",
                          "sub-01/perf/sub-01_asl.nii.gz"], 1),
    ("one subject with raw/", ["Agnik_Data/x_meancbf.nii",
                               "Agnik_Data/raw/ASL.nii.gz"], 1),
    # a folder is only a subject once it holds something gradeable
    ("folders of masks only", ["c/a/mask.nii.gz", "c/b/mask.nii.gz"], 0),
    ("loose files, no folder", ["a.nii.gz", "b.nii.gz"], 0),
    ("subjects with subfolders", ["c/s1/perf/s1_cbf.nii.gz",
                                  "c/s2/perf/s2_cbf.nii.gz"], 2),
])
def test_the_browser_counts_subjects_the_way_the_server_does(label, paths, expected):
    import json
    import subprocess
    script = _extract_counter() + f"""
const paths = {json.dumps(paths)};
const files = paths.map(p => ({{webkitRelativePath: p, name: p.split('/').pop()}}));
console.log(subjectCount(files));
"""
    out = subprocess.run([_node(), "-e", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == expected, f"{label}: got {out.stdout.strip()}"


def test_a_cbf_map_cannot_silently_cancel_cohort_grading():
    """The server decided cohort-vs-single on `not paths["cbf"]`.

    So one stale CBF part - which the page could produce by re-enabling a hidden
    input that still held a file - routed a whole cohort into the single-scan
    branch. The cohort's files were still uploaded and still graded, folded into
    one report named after the single map: eleven checks became twenty, with
    WARNs computed from three different people's raw data, and no error. The
    12-subject cap was never consulted either, because it lives on the branch
    that was skipped.
    """
    body = _cohort_body(3).rsplit(b"--" + BOUNDARY.encode() + b"--\r\n", 1)[0]
    c = synthetic_case(quality="clean", seed=9)
    body += _part("cbf", "cbf.nii.gz", _nifti_bytes(c.cbf))
    body += b"--" + BOUNDARY.encode() + b"--\r\n"
    with pytest.raises(ValueError, match="two different reports"):
        _grade(body)


def test_the_subject_cap_cannot_be_bypassed_the_same_way(monkeypatch):
    """The cap is a memory ceiling, so slipping past it is not a cosmetic bug."""
    monkeypatch.setattr(web, "MAX_COHORT_SUBJECTS", 2)
    body = _cohort_body(3).rsplit(b"--" + BOUNDARY.encode() + b"--\r\n", 1)[0]
    c = synthetic_case(quality="clean", seed=9)
    body += _part("cbf", "cbf.nii.gz", _nifti_bytes(c.cbf))
    body += b"--" + BOUNDARY.encode() + b"--\r\n"
    # refused for being contradictory, never silently graded as one scan
    with pytest.raises(ValueError):
        _grade(body)


# --------------------------------------------------------------------------- #
# defects the adversarial review confirmed after the redesign
# --------------------------------------------------------------------------- #
def test_hidden_actually_hides():
    """`.dbtn{display:inline-flex}` is an author rule and the UA's
    `[hidden]{display:none}` is the weakest rule there is, so it lost.

    `pickfiles.hidden = true` in cohort mode therefore changed nothing on
    screen, leaving a "Choose files..." button that opens the single-file picker
    in the one mode where only a folder makes sense.
    """
    assert "[hidden]{display:none!important}" in web._upload_page()


def test_every_upload_box_has_its_own_accessible_name():
    """Without one they all report the UA fallback "Choose file", so a screen
    reader meets four identical buttons and cannot tell CBF from CSF."""
    page = web._upload_page()
    for title in ("Grey matter", "White matter", "CSF"):
        assert f'aria-label="{title}"' in page, title


def test_the_cohort_banner_is_announced():
    """The banner appears, the mode flips and the submit button renames itself.
    Silently, to anyone not watching the screen."""
    page = web._upload_page()
    assert "'role', 'status'" in page and "'aria-live', 'polite'" in page


def test_a_filename_cannot_inject_markup():
    """show() rendered each picked filename with innerHTML. A file called
    `<img src=x onerror=...>.nii.gz` is legal on every platform this runs on."""
    page = web._upload_page()
    assert "nm.textContent = f.name" in page
    assert "'<span>'+f.name+'</span>'" not in page


def test_the_organ_chip_and_its_tooltip_agree():
    """It rendered "brain (21)" with the title "20 checks." - the count came
    from the registry, the tooltip from a hand-typed string that had gone stale."""
    from osipy_qc.core.registry import all_checks
    page = web._upload_page()
    n = len(all_checks("brain"))
    assert f"{n} checks." in page and f"brain ({n})" in page


def test_the_placenta_vsasl_fields_exist_on_the_page():
    """_organ_inputs reads these two whenever the scheme is VSASL. The page
    rendered neither, so p4.1 always WARNed that they were missing and no user
    action could clear it - and VSASL is the first option in the select."""
    page = web._upload_page()
    for f in ("cutoff_velocity_cm_s", "post_labeling_delay_s"):
        assert f"placenta__{f}" in page, f


def test_the_per_role_boxes_work_for_kidney_too(tmp_path):
    """They are the escape hatch for a scanner that exports anon_0042.nii.gz.
    `load_organ_folder` took no role_overrides, so for kidney and placenta the
    box did nothing and the file was re-classified from its unreadable name."""
    from osipy_qc.io import load_organ_folder
    c = synthetic_case(quality="clean", seed=0)
    _write(tmp_path / "anon_0042.nii.gz", c.cbf)
    plain = load_organ_folder(str(tmp_path), "kidney")
    assert plain.get("rbf_map") is None, "fixture name should be unrecognisable"
    named = load_organ_folder(str(tmp_path), "kidney",
                              role_overrides={"anon_0042.nii.gz": "asl"})
    assert [f["name"] for f in named["files"]] == ["anon_0042.nii.gz"]
