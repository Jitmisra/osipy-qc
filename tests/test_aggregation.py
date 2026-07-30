"""
The aggregation rule, and the coverage that must accompany it.

These tests exist because of a review finding: a mentor uploaded a good CBF map
and got WARN. Two separate defects caused it, and the suite had 167 passing tests
that touched neither - the only two assertions mentioning UNKNOWN were
tautologies of the form `verdict in {"PASS","WARN","FAIL","UNKNOWN"}`, which no
behaviour can fail.

So these are written to fail if the rule is ever reverted, and each states the
reasoning rather than just the expected value.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy_qc.batch import cbf_map_checks
from osipy_qc.core import CheckResult, Verdict, aggregate
from osipy_qc.core.result import EXCLUDED_FROM_AGGREGATION, coverage
from osipy_qc.report import run_qc
from osipy_qc.synth import synthetic_case


def r(verdict: Verdict, check: str = "x.test") -> CheckResult:
    return CheckResult(check, verdict)


# --------------------------------------------------------------------------
# aggregate(): an absence must not become a finding
# --------------------------------------------------------------------------

def test_unknown_alone_is_unknown_not_warn():
    """The core of the review finding.

    An UNKNOWN says "I was given nothing to measure". It used to escalate to
    WARN, which reported an absence as a concern - so a reviewer who supplied
    one legitimate input got a warning about the inputs they did not supply.
    """
    assert aggregate([r(Verdict.UNKNOWN)]) is Verdict.UNKNOWN


def test_unknown_does_not_drag_down_a_clean_pass():
    """A good measurement plus a missing input is still a good measurement."""
    assert aggregate([r(Verdict.PASS), r(Verdict.UNKNOWN)]) is Verdict.PASS


def test_unknown_is_excluded_like_na_and_info():
    """All three state an absence, so all three are excluded. If a future edit
    removes UNKNOWN from this set, the test above changes meaning silently -
    this pins the set itself."""
    assert EXCLUDED_FROM_AGGREGATION == {Verdict.NA, Verdict.INFO, Verdict.UNKNOWN}


def test_real_warn_still_warns():
    """Excluding UNKNOWN must not soften an actual measured problem."""
    assert aggregate([r(Verdict.PASS), r(Verdict.WARN)]) is Verdict.WARN


def test_fail_still_dominates():
    assert aggregate([r(Verdict.PASS), r(Verdict.WARN), r(Verdict.FAIL)]) is Verdict.FAIL
    assert aggregate([r(Verdict.FAIL), r(Verdict.UNKNOWN)]) is Verdict.FAIL


def test_nothing_gradeable_is_unknown_not_pass():
    """The guard against a PASS on a technicality: a report where every check
    was an absence has not been shown to be good, so it must not read as good."""
    assert aggregate([r(Verdict.UNKNOWN), r(Verdict.NA), r(Verdict.INFO)]) is Verdict.UNKNOWN
    assert aggregate([]) is Verdict.UNKNOWN


def test_na_and_info_alone_do_not_pass():
    assert aggregate([r(Verdict.NA), r(Verdict.INFO)]) is Verdict.UNKNOWN


# --------------------------------------------------------------------------
# coverage(): the fact the verdict no longer carries
# --------------------------------------------------------------------------

def test_coverage_reports_what_was_missing():
    results = [r(Verdict.PASS, "1.qei"), r(Verdict.UNKNOWN, "7.1.motion"),
               r(Verdict.UNKNOWN, "5.1.schema")]
    cov = coverage(results)
    assert cov["graded"] == 1
    assert cov["unknown"] == 2
    assert cov["total"] == 3
    assert cov["complete"] is False
    assert cov["missing"] == ["7.1.motion", "5.1.schema"]


def test_coverage_excludes_na_from_the_denominator():
    """A check that cannot apply to this data was never owed an answer, so
    counting it as un-covered would understate coverage. 3.4.deep_gm_ratio is
    N/A for an adult, and an adult report is not 'incomplete' because of it."""
    results = [r(Verdict.PASS, "1.qei"), r(Verdict.NA, "3.4.deep_gm_ratio"),
               r(Verdict.INFO, "8.2.data_type")]
    cov = coverage(results)
    assert cov["total"] == 1
    assert cov["complete"] is True
    assert cov["missing"] == []


def test_coverage_is_in_the_json():
    """A consumer reading only overall_verdict would not know the report was
    partial, so coverage travels with it.

    A lone CBF map grades exactly one check - 3.5.brain_cbf, the only magnitude
    check that needs no tissue maps - so coverage must report 1 graded and the
    report must still be marked incomplete.
    """
    case = synthetic_case(quality="clean", seed=0)
    rep = run_qc({"cbf": case.cbf, "voxel_mm": (3., 3., 3.)}, checks=cbf_map_checks())
    d = rep.to_dict()
    assert "coverage" in d
    assert d["coverage"]["graded"] == 1
    assert d["coverage"]["unknown"] > 0
    assert d["coverage"]["complete"] is False


# --------------------------------------------------------------------------
# The end-to-end cases from the review
# --------------------------------------------------------------------------

def test_good_map_with_tissue_maps_passes():
    """The control: this map is genuinely good, so nothing in the review finding
    is about the grading being wrong."""
    c = synthetic_case(quality="clean", seed=0)
    rep = run_qc({"cbf": c.cbf, "gm": c.gm, "wm": c.wm, "csf": c.csf,
                  "voxel_mm": (3., 3., 3.)}, checks=cbf_map_checks())
    assert rep.overall is Verdict.PASS
    assert rep.coverage["complete"] is True


def test_good_map_alone_is_not_warn():
    """The mentor's case: 'This is a good CBF map. It should not be marked as
    Warn.'

    Two changes together produce the right answer. Excluding UNKNOWN stops the
    absent inputs manufacturing a WARN, and 3.5.brain_cbf gives the map one thing
    that can actually be graded without tissue maps - so the PASS is earned by a
    measurement rather than granted by an absence of findings.
    """
    c = synthetic_case(quality="clean", seed=0)
    rep = run_qc({"cbf": c.cbf, "voxel_mm": (3., 3., 3.)}, checks=cbf_map_checks())
    assert rep.overall is Verdict.PASS
    assert rep.coverage["graded"] == 1
    # ...and the PASS must not pretend to be a full report
    assert rep.coverage["complete"] is False


def test_a_real_problem_still_surfaces_without_tissue_maps():
    """The risk of excluding UNKNOWN is that absences hide problems. Whatever
    can still be measured must still be able to condemn the scan."""
    c = synthetic_case(quality="clean", seed=0)
    bad = c.cbf.copy()
    bad[bad > 0] = -abs(bad[bad > 0])          # every voxel non-physical
    rep = run_qc({"cbf": bad, "gm": c.gm, "wm": c.wm, "csf": c.csf,
                  "voxel_mm": (3., 3., 3.)}, checks=cbf_map_checks())
    assert rep.overall in (Verdict.WARN, Verdict.FAIL)
