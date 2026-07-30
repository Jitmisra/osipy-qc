"""
Check 3.5 - whole-brain CBF over a self-derived mask.

Added in response to a mentor review: "This is a good CBF map. It should not be
marked as Warn. Do we have an overall check of CBF brain values?" There was no
such check - all ten CBF-map checks hard-required a grey-matter map - so a lone
CBF map had nothing gradeable at all.

The tests that matter most here are the ones pinning what this check DOES NOT do.
The obvious implementation (grade the brain mean against a normal band, as 3.1
grades GM) is indefensible: no published whole-brain band exists, and the mask
percentile moves the mean by ~18 mL/100g/min, which is a third of any plausible
band's width. test_mask_percentile_moves_the_mean_a_lot is the standing evidence
for that design decision - if it ever fails, the argument in the docstring is
wrong and the check should be revisited.
"""

from __future__ import annotations

import numpy as np
import pytest

from osipy_qc.checks.cbf_level import brain_cbf_check
from osipy_qc.core.config import QCConfig, for_population
from osipy_qc.core.result import Verdict
from osipy_qc.synth import synthetic_case
from osipy_qc.utils.masks import brain_mask_fallback


@pytest.fixture(scope="module")
def clean():
    return synthetic_case(quality="clean", seed=0)


# --------------------------------------------------------------------------
# What it grades
# --------------------------------------------------------------------------

def test_good_lone_cbf_map_passes(clean):
    """The mentor's case. This is the whole point of the check existing."""
    r = brain_cbf_check(cbf=clean.cbf)
    assert r.verdict is Verdict.PASS
    assert 30 < r.metric["mean_brain_cbf"] < 90


def test_negative_brain_mean_fails():
    """A brain mean at or below zero cannot be quantified CBF. The White Paper
    (p.17) attributes exactly this to a global labelling-efficiency loss or a
    mis-scaled PD/M0 scan."""
    c = synthetic_case(quality="garbage", seed=0)
    r = brain_cbf_check(cbf=c.cbf)
    assert r.verdict is Verdict.FAIL
    assert r.metric["mean_brain_cbf"] <= 0


def test_absurdly_high_mean_fails(clean):
    """A unit or scaling error, not perfusion."""
    r = brain_cbf_check(cbf=clean.cbf * 10.0)
    assert r.verdict is Verdict.FAIL
    assert "scaling" in r.reason or "unit" in r.reason


def test_verdicts_are_provisional():
    """Both bounds are UNCALIBRATED, so neither may be presented as a hard,
    evidence-backed failure."""
    c = synthetic_case(quality="garbage", seed=0)
    assert brain_cbf_check(cbf=c.cbf).provisional is True


# --------------------------------------------------------------------------
# What it deliberately does NOT grade - the design decisions
# --------------------------------------------------------------------------

def test_mask_percentile_moves_the_mean_a_lot(clean):
    """The evidence for not grading the mean against a normal band.

    If this ever fails, the docstring's argument is wrong: the mean would be
    stable enough to threshold, and 3.5 should be reconsidered.
    """
    means = []
    for p in (25.0, 40.0, 50.0, 60.0, 75.0):
        vals = clean.cbf[brain_mask_fallback(clean.cbf, p)]
        means.append(float(np.mean(vals)))
    spread = max(means) - min(means)
    assert spread > 10.0, (
        f"brain mean spread is only {spread:.1f} across the mask percentile - "
        "stable enough that a graded band might now be defensible")


def test_the_sign_of_the_mean_is_mask_stable():
    """The converse, and the reason a gross-implausibility grade IS defensible:
    the VALUE moves with the mask but the SIGN does not."""
    good = synthetic_case(quality="clean", seed=0).cbf
    bad = synthetic_case(quality="garbage", seed=0).cbf
    for p in (25.0, 40.0, 50.0, 60.0, 75.0):
        assert float(np.mean(good[brain_mask_fallback(good, p)])) > 0
        assert float(np.mean(bad[brain_mask_fallback(bad, p)])) < 0


def test_reports_the_mask_that_produced_the_number(clean):
    """A number whose value depends on a knob must carry the knob with it, or a
    reader will compare it against a published figure computed differently."""
    r = brain_cbf_check(cbf=clean.cbf)
    assert "self-derived" in r.metric["mask"]
    assert "p50" in r.metric["mask"]
    assert "no published whole-brain band" in r.metric["graded_against"]


def test_stands_aside_when_tissue_maps_are_present(clean):
    """3.1 grades GM and WM against published bands, which is better evidence.
    Two magnitude verdicts on one problem would double-count it."""
    r = brain_cbf_check(cbf=clean.cbf, gm=clean.gm)
    assert r.verdict is Verdict.NA
    assert "3.1" in r.reason


# --------------------------------------------------------------------------
# Degenerate inputs
# --------------------------------------------------------------------------

def test_no_cbf_map_is_unknown():
    assert brain_cbf_check(cbf=None).verdict is Verdict.UNKNOWN


def test_all_zero_map_cannot_derive_a_mask():
    r = brain_cbf_check(cbf=np.zeros((8, 8, 6)))
    assert r.verdict is Verdict.UNKNOWN
    assert "brain mask" in r.reason


def test_non_finite_values_do_not_crash_or_poison_the_mean(clean):
    """NaN/inf reach this check from real pipelines; the mean must stay finite."""
    arr = clean.cbf.copy()
    arr[0, 0, 0] = np.nan
    arr[0, 0, 1] = np.inf
    r = brain_cbf_check(cbf=arr)
    assert np.isfinite(r.metric["mean_brain_cbf"])
    assert r.verdict is Verdict.PASS


def test_population_does_not_silently_change_the_bounds(clean):
    """The absurdity bounds are about being CBF at all, not about being adult or
    neonatal CBF - a neonate's ~16 and an adult's ~60 are both plausible here.
    A neonatal map must not FAIL merely for being neonatal."""
    neonatal = clean.cbf * (16.0 / max(float(np.mean(clean.cbf[clean.cbf > 0])), 1e-9))
    for pop in ("adult", "neonate"):
        r = brain_cbf_check(cbf=neonatal, cfg=for_population(pop))
        assert r.verdict is Verdict.PASS, f"{pop}: {r.reason}"
