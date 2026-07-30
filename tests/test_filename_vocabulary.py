"""
One filename vocabulary, checked from both ends.

The upload page and the server each decided what a dropped raw file was, from two
hand-maintained copies of the same token list. The copies had drifted five ways:

    perfusion_calib.nii.gz               page said "M0", server said asl
    native_space_perfusion_calib.nii.gz  page said "M0", server said asl
    calib.nii.gz                         page said "M0", server said other
    calibration.nii                      page said "M0", server said other
    aCBV_calib.nii.gz                    page said "M0", server said other

The last three are the damaging ones: "other" is not routed anywhere, so the page
promised the reader an M0, load_folder dropped the file without a word, and check
8.2 then reported `m0: absent` with no error to explain it.

The unification goes both ways, because both sides were wrong about something.
`calib` genuinely does name a calibration (M0) scan in oxford_asl usage, so the
server had to learn it. But oxford_asl also uses a trailing `_calib` to mean
"calibrated", so perfusion_calib and aCBV_calib are derived output maps, not M0
scans, and the page had to stop claiming otherwise. Hence `calib` as a PREFIX
rule, applied after the ASL tokens.

The fix is not the token list, it is that there is now only one: `_ROLE_RULES` in
checks/schema.py, serialised into the page's JavaScript. These tests hold both
ends to it - the corpus test compares the server's answer against the rules the
served page actually carries, so a hand-edit of either one fails here.
"""

from __future__ import annotations

import json
import re

import nibabel as nib
import numpy as np
import pytest

from osipy_qc.checks.schema import classify_role, role_vocabulary
from osipy_qc.web import _ROLE_LABELS, _upload_page

# name -> the role BOTH sides must return. Every entry is a name a real pipeline
# or a real reviewer produced.
CORPUS: dict[str, str] = {
    # the five disagreements from the review
    "perfusion_calib.nii.gz": "asl",
    "native_space_perfusion_calib.nii.gz": "asl",
    "calib.nii.gz": "m0",
    "calibration.nii": "m0",
    # a calibrated arterial-blood-volume map: an oxford_asl output, so neither a
    # raw ASL series nor a calibration scan. "other" is the honest answer, and the
    # page must say so rather than promise an M0 it will not deliver.
    "aCBV_calib.nii.gz": "other",
    # the three real mentor-supplied datasets
    "PCASL_027_S_6327.nii.gz": "asl",
    "M0_027_S_6327.nii.gz": "m0",
    "MPRAGE_027_S_6327.nii.gz": "t1",
    "PCASL.nii.gz": "asl",
    "M0.nii.gz": "m0",
    "T1.nii.gz": "t1",
    "MPRAGE.nii.gz": "t1",
    # BIDS, which the page recommends as canonical
    "sub-01_asl.nii.gz": "asl",
    "sub-01_m0scan.nii.gz": "m0",
    "sub-01_T1w.nii.gz": "t1",
    "sub-01_ses-01_acq-pcasl_dir-AP_run-1_asl.nii.gz": "asl",
    "sub-01_ses-01_m0scan.nii.gz": "m0",
    "sub-01_ses-01_run-1_T1w.nii.gz": "t1",
    "sub-01_task-rest_asl.nii.gz": "asl",
    # other pipelines and hand-named files
    "ASL4D.nii.gz": "asl",              # ExploreASL
    "conAFeHe73_1_pairs.nii": "asl",    # a reviewer's own naming
    "deltam.nii.gz": "asl",
    "perfusion.nii.gz": "asl",
    "calib_head.nii.gz": "m0",
    "anat.nii.gz": "t1",
    # nothing in the vocabulary, and nothing should be invented for them
    "mask.nii.gz": "other",
    "brain.nii.gz": "other",
}

#: BIDS names the page tells the reader to prefer. Recommending them is only
#: honest if they pass, so this is asserted rather than assumed.
BIDS_NAMES = {n: r for n, r in CORPUS.items() if n.startswith("sub-")}


def _page_rules() -> list[dict]:
    """The rule table the served page actually carries, parsed back out of its JS.

    Read from the rendered page on purpose. Importing role_vocabulary() twice
    would prove only that a function equals itself; this proves the page ships it.
    """
    page = _upload_page()
    m = re.search(r"var ROLES = (\[.*?\]);", page, re.S)
    assert m, "the page no longer ships a generated rule table"
    return json.loads(m.group(1))


def _page_role(rules: list[dict], filename: str) -> str:
    """Apply the page's rules exactly as its role() does: first match wins."""
    n = filename.lower()
    for r in rules:
        for t in r["tokens"]:
            if n.startswith(t) if r["how"] == "starts" else t in n:
                return r["role"]
    return "other"


# --------------------------------------------------------------------------
# the two ends must agree
# --------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected", sorted(CORPUS.items()))
def test_page_and_server_agree(filename, expected):
    """The test that is the actual fix. Both classifiers, one corpus, one answer."""
    server = classify_role(filename)
    page = _page_role(_page_rules(), filename)
    assert server == expected, f"server called {filename} {server!r}"
    assert page == expected, f"the page called {filename} {page!r}"


def test_the_page_ships_the_python_table_verbatim():
    """Not "equivalent to" - the same table. Anything less is a second copy, and a
    second copy is what produced the five disagreements above."""
    assert _page_rules() == role_vocabulary()


def test_every_role_the_vocabulary_can_return_has_a_page_label():
    """The page maps role -> wording. A rule for a role with no label would render
    the reader `undefined`."""
    roles = {r["role"] for r in role_vocabulary()} | {"other"}
    assert roles <= set(_ROLE_LABELS), f"unlabelled roles: {roles - set(_ROLE_LABELS)}"


# --------------------------------------------------------------------------
# the two directional facts, stated on their own
# --------------------------------------------------------------------------

def test_a_calibration_scan_named_calib_is_no_longer_dropped(tmp_path):
    """End to end through the loader, which is where the damage was done.

    `calib.nii.gz` classified as "other", so load_folder filed it nowhere and 8.2
    reported `m0: absent` - a missing-M0 warning about an M0 the user had
    supplied and the page had confirmed.
    """
    from osipy_qc.io import load_folder

    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    nib.save(nib.Nifti1Image(np.ones((8, 8, 6, 4), dtype=np.float32), aff),
             str(tmp_path / "PCASL.nii.gz"))
    nib.save(nib.Nifti1Image(np.ones((8, 8, 6), dtype=np.float32), aff),
             str(tmp_path / "calib.nii.gz"))
    inputs = load_folder(str(tmp_path))
    assert inputs["detected"]["m0"] == "separate"
    assert inputs["m0_shape"] == (8, 8, 6)


def test_perfusion_calib_is_not_taken_for_a_calibration_scan():
    """Token ORDER, and why `calib` is a prefix rule and not a substring one.

    In oxford_asl output a trailing _calib means "calibrated": perfusion_calib is
    the CBF map. Reading it as the M0 would feed a CBF map to the M0 geometry and
    background-suppression checks.
    """
    assert classify_role("perfusion_calib.nii.gz") == "asl"
    assert classify_role("native_space_perfusion_calib.nii.gz") == "asl"
    assert classify_role("aCBV_calib.nii.gz") == "other"
    # and the name that is only "calib" still reaches m0
    assert classify_role("calib.nii.gz") == "m0"


def test_bids_names_pass_so_recommending_them_is_honest():
    for name, expected in BIDS_NAMES.items():
        assert classify_role(name) == expected, name
