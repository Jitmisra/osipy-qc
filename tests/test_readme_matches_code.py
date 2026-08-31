"""The README documents all 54 checks, and says nothing the code contradicts.

For most of this project's life the README described 20 brain checks and never
mentioned kidney or placenta - 34 of 54 checks, a third of the tool, undocumented.
It also told readers to select an organ with `run_qc(..., organ=...)`, which is not
a parameter that exists, and stated an aggregation rule the code does not implement.

None of that was caught by a test, because documentation drift is invisible to a
test suite that only exercises code. These tests close that: the README's check
tables, its counts, and its runnable snippet are all checked against the registry
and the live API. If a check is added without a README row, this fails.
"""

from __future__ import annotations

import inspect
import pathlib
import re

import pytest

from osipy_qc.core.registry import all_checks
from osipy_qc.core.result import coverage
from osipy_qc.report import run_qc

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()

ORGAN_COUNTS = {"brain": 20, "kidney": 19, "placenta": 15}


def _section(organ: str) -> str:
    """The README block documenting one organ's checks."""
    return README.split(f"### {organ} — ")[1].split("### ")[0]


def _cited_ids(body: str) -> set[str]:
    """Check ids named in backticks in a README section."""
    return {c for c in re.findall(r"`([kp]?\d[\w.]*)`", body) if "." in c}


@pytest.mark.parametrize("organ", sorted(ORGAN_COUNTS))
def test_every_registered_check_has_a_readme_row(organ):
    missing = sorted(set(all_checks(organ)) - _cited_ids(_section(organ)))
    assert not missing, (
        f"{organ} checks exist but are undocumented in the README: {missing}. "
        "A check nobody can read about is a check nobody will trust.")


@pytest.mark.parametrize("organ", sorted(ORGAN_COUNTS))
def test_the_readme_does_not_document_checks_that_do_not_exist(organ):
    bogus = sorted(_cited_ids(_section(organ)) - set(all_checks(organ)))
    assert not bogus, f"README documents non-existent {organ} checks: {bogus}"


@pytest.mark.parametrize("organ,expected", sorted(ORGAN_COUNTS.items()))
def test_the_per_organ_counts_in_the_readme_are_current(organ, expected):
    assert len(all_checks(organ)) == expected, (
        f"{organ} now has {len(all_checks(organ))} checks, not {expected} - "
        "update the organ table in README section 6.")
    assert f"[{organ}](#{organ}--{expected}-checks) | {expected} " in README, (
        f"the README organ table does not say {organ} has {expected} checks")


def test_the_headline_total_is_the_real_total():
    total = len(all_checks())
    assert total == sum(ORGAN_COUNTS.values())
    assert f"{total} checks, three organs" in README


def test_the_readme_organ_snippet_actually_runs(tmp_path):
    """The Python the README hands a new user must work as written.

    It told people to pass `organ=` to run_qc, which raises TypeError. Anyone
    following the README hit an error on their first call.
    """
    block = README.split("```python\nfrom osipy_qc import run_qc\n")[1].split("```")[0]
    code = "from osipy_qc import run_qc\n" + block

    folder = ROOT.parent / "test_upload" / "kidney_phantom"
    if not folder.exists():
        pytest.skip("kidney phantom folder not present in this checkout")
    code = code.replace('"data/my_kidney_scan"', repr(str(folder)))

    ns: dict = {}
    exec(compile(code, "<README section 6>", "exec"), ns)   # noqa: S102 - that is the test
    assert ns["report"].results, "the README snippet produced an empty report"


def test_run_qc_has_no_organ_parameter_so_the_readme_must_not_claim_one():
    """Guards the specific wording that was wrong.

    Organ selection lives on QCConfig, not on run_qc. If that ever changes, this
    test fails and the README sentence gets revisited deliberately.
    """
    assert "organ" not in inspect.signature(run_qc).parameters
    assert 'run_qc(..., organ=' not in README


def test_the_readme_aggregation_rule_matches_aggregate():
    """UNKNOWN does not escalate to WARN; the README said it did."""
    from osipy_qc.core.result import EXCLUDED_FROM_AGGREGATION, Verdict

    assert Verdict.UNKNOWN in EXCLUDED_FROM_AGGREGATION
    assert "any WARN/UNKNOWN" not in README, (
        "README states UNKNOWN escalates the overall verdict; aggregate() excludes it")
    assert "else **UNKNOWN**" in README


def test_the_coverage_keys_the_readme_prints_are_the_real_ones():
    shown = set(re.findall(r"'(\w+)':", README.split("coverage(report.results)")[1][:160]))
    assert shown <= set(coverage([])), f"README shows coverage keys that do not exist: {shown}"
