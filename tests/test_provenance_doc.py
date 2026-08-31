"""THRESHOLD_PROVENANCE.md must exist and match the code.

Eleven places cite this file - the CLI's --provenance output, the footer of every
HTML report, the README, USAGE - and for a long time it did not exist, so all
eleven pointed at nothing. A reader who went looking for the sourcing behind a
FAIL found a dead reference, which is the opposite of what a citation is for.

It is generated from `THRESHOLD_PROVENANCE`, so these tests are about drift: the
checked-in file must still be the one the code would produce.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_cited_file_exists():
    assert (ROOT / "THRESHOLD_PROVENANCE.md").exists()


def test_the_checked_in_file_matches_the_code():
    """Run the generator in --check mode. If this fails, run
    `python tools/gen_provenance.py`."""
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_provenance.py"), "--check"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stdout + out.stderr


def test_every_reference_to_a_doc_file_resolves():
    """A citation pointing at a file that is not there is worse than no citation.
    POPULATION_BANDS.md was linked from README and USAGE and never existed."""
    import re
    missing = []
    for md in ROOT.glob("*.md"):
        for target in re.findall(r"\]\(([A-Z_]+\.md)\)", md.read_text()):
            if not (ROOT / target).exists():
                missing.append(f"{md.name} -> {target}")
    assert not missing, missing


def test_the_document_states_the_real_strict_contract():
    """It is the project's honesty artifact; it must not repeat the claim the
    code never implemented."""
    text = (ROOT / "THRESHOLD_PROVENANCE.md").read_text()
    assert "never drive a FAIL" not in text
    assert "provisional" in text and "--no-strict" in text


def test_the_counts_in_the_document_are_the_code_s_counts():
    from osipy_qc.core.config import THRESHOLD_PROVENANCE, Provenance
    text = (ROOT / "THRESHOLD_PROVENANCE.md").read_text()
    for level, label in ((Provenance.PUBLISHED, "published"),
                         (Provenance.IMPLEMENTATION, "implementation"),
                         (Provenance.UNCALIBRATED, "uncalibrated")):
        n = sum(1 for _k, (l, _c, _t) in THRESHOLD_PROVENANCE.items() if l is level)
        assert f"{label} | {n} " in text, f"{label} count drifted"
