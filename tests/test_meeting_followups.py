"""Known-answer tests for the changes driven by the 2026-07 mentor review.

Each test pins a specific piece of mentor feedback so it cannot silently regress:

  * Sudipto  — GM CBF must not average structural zeros from an uncovered FOV
               ("the cerebellum is probably not covered in the ASL scan ... so the
               gray matter CBF artificially looks low").
  * Sudipto  — every threshold must carry its provenance.
  * Maria    — the toolbox must cover the lifespan, not just adults; in newborns
               CBF is higher in deep GM than cortical GM.
  * Maria    — it must be modular for other organs.
  * Zhiliang — an altered WM CBF can be a pathological finding, not an image
               failure, so the ratio check must inform rather than condemn.
"""

import numpy as np
import pytest

from osipy_qc.checks.cbf_level import (cbf_level_check, deep_gm_ratio_check,
                                       gm_wm_ratio_check)
from osipy_qc.checks.coreg import coverage_check
from osipy_qc.core import Verdict
from osipy_qc.core.config import (POPULATIONS, Provenance, QCConfig,
                                  THRESHOLD_PROVENANCE, for_organ, for_population,
                                  provenance_of, skipped_for_organ,
                                  uncalibrated_fields)
from osipy_qc.utils.masks import coverage_fraction, covered_tissue_mask


# --------------------------------------------------------------------------- #
# Sudipto: the uncovered-FOV / cerebellum bug
# --------------------------------------------------------------------------- #
def _partially_covered_case(covered_cols: int = 7):
    """GM slab whose CBF is only present in `covered_cols`/10 columns; the rest
    are structural zeros (outside the ASL FOV). WM slab fully covered at 20."""
    gm = np.zeros((10, 10, 2)); gm[:, :, 0] = 1.0
    wm = np.zeros((10, 10, 2)); wm[:, :, 1] = 1.0
    cbf = np.zeros((10, 10, 2))
    cbf[:, :covered_cols, 0] = 60.0
    cbf[:, :, 1] = 20.0
    return cbf, gm, wm


def test_uncovered_voxels_do_not_drag_gm_cbf_down():
    """The exact failure Sudipto described: zeros inside the ROI are averaged in
    and mean GM CBF 'artificially looks low'. Naive mean here is 42; truth is 60."""
    cbf, gm, wm = _partially_covered_case()
    naive = float(cbf[gm > 0.7].mean())
    assert abs(naive - 42.0) < 1e-6           # the bug, if unguarded

    r = cbf_level_check(cbf=cbf, gm=gm, wm=wm)
    assert abs(r.metric["mean_gm_cbf"] - 60.0) < 1e-6   # coverage-masked truth


def test_uncovered_voxels_do_not_corrupt_the_ratio():
    cbf, gm, wm = _partially_covered_case()
    r = gm_wm_ratio_check(cbf=cbf, gm=gm, wm=wm)
    assert abs(r.metric["gm_wm_ratio"] - 3.0) < 1e-6    # not 42/20 = 2.1


def test_coverage_check_surfaces_the_fov_mismatch():
    """The exclusion must be visible, not silent."""
    cbf, gm, wm = _partially_covered_case()
    r = coverage_check(cbf=cbf, gm=gm, wm=wm)
    assert abs(r.metric["gm_coverage"] - 0.7) < 1e-6
    assert abs(r.metric["wm_coverage"] - 1.0) < 1e-6
    assert r.verdict == Verdict.FAIL
    assert "FOV" in r.reason or "covered" in r.reason


def test_full_coverage_passes():
    cbf, gm, wm = _partially_covered_case(covered_cols=10)
    r = coverage_check(cbf=cbf, gm=gm, wm=wm)
    assert r.metric["gm_coverage"] == 1.0
    assert r.verdict == Verdict.PASS


def test_coverage_helpers():
    cbf, gm, _ = _partially_covered_case()
    assert abs(coverage_fraction(cbf, gm) - 0.7) < 1e-6
    assert int(np.count_nonzero(covered_tissue_mask(cbf, gm))) == 70
    # empty ROI must not divide by zero
    assert coverage_fraction(cbf, np.zeros_like(gm)) == 0.0


# --------------------------------------------------------------------------- #
# Sudipto: provenance on every threshold
# --------------------------------------------------------------------------- #
def test_published_thresholds_carry_a_real_citation():
    for field, (level, citation, _note) in THRESHOLD_PROVENANCE.items():
        if level is Provenance.PUBLISHED:
            assert citation != "NONE", f"{field} claims PUBLISHED but cites nothing"
            assert any(tok in citation for tok in ("doi:", "10.")), \
                f"{field} PUBLISHED citation lacks a DOI: {citation}"


def test_uncalibrated_thresholds_are_declared_not_hidden():
    """The honest list. These must never be presented as evidence-backed."""
    unc = uncalibrated_fields()
    for field in ("skew_lo", "skew_hi"):
        assert field not in unc, f"{field} should no longer exist at all"
    # the ones we know we cannot cite
    for field in ("qei_pass", "ratio_pass", "gm_cbf_fail_lo", "dice_pass", "t1_tissue_s"):
        assert field in unc


def test_the_retracted_skew_thresholds_are_gone():
    """skew_lo/-0.5 had no ASL basis and skew_hi was dead code. Both removed."""
    cfg = QCConfig()
    assert not hasattr(cfg, "skew_lo")
    assert not hasattr(cfg, "skew_hi")


def test_provenance_lookup_defaults_to_uncalibrated():
    level, citation, _ = provenance_of("a_field_that_does_not_exist")
    assert level is Provenance.UNCALIBRATED
    assert citation == "NONE"


def test_scov_provenance_records_the_broken_citation_chain():
    """0.67 is routinely attributed to Mutsaerts, but that chain does not hold.
    The provenance note must say so rather than repeat the attribution."""
    _level, _cite, note = provenance_of("scov_vascular")
    assert "BROKEN" in note.upper()


# --------------------------------------------------------------------------- #
# Maria: lifespan
# --------------------------------------------------------------------------- #
def test_population_profiles_exist_across_the_lifespan():
    for name in ("neonate_term", "neonate_preterm", "infant", "child",
                 "adolescent", "adult", "elderly"):
        assert name in POPULATIONS


def test_adult_bands_are_the_white_paper_band():
    cfg = for_population("adult")
    assert (cfg.gm_cbf_lo, cfg.gm_cbf_hi) == (40.0, 100.0)


def test_a_healthy_neonate_would_fail_adult_bands_but_pass_neonatal_ones():
    """The whole point of population profiles. Miranda 2006: term neonate
    cortical GM ~16, WM ~10 — far below the adult 40-100 band."""
    gm = np.ones((6, 6, 1)); wm = np.zeros((6, 6, 2))
    gm3 = np.zeros((6, 6, 2)); gm3[:, :, 0] = 1.0
    wm3 = np.zeros((6, 6, 2)); wm3[:, :, 1] = 1.0
    cbf = np.zeros((6, 6, 2))
    cbf[:, :, 0] = 16.0     # neonatal cortical GM
    cbf[:, :, 1] = 10.0     # neonatal WM

    adult = cbf_level_check(cbf=cbf, gm=gm3, wm=wm3, cfg=for_population("adult"))
    assert adult.verdict in (Verdict.WARN, Verdict.FAIL)   # condemned by adult bands

    neo = cbf_level_check(cbf=cbf, gm=gm3, wm=wm3, cfg=for_population("neonate_term"))
    assert neo.verdict == Verdict.PASS
    assert neo.metric["population"] == "neonate_term"


def test_unknown_population_raises_rather_than_defaulting_to_adult():
    """A silent adult default on a neonate would fail every scan."""
    with pytest.raises(ValueError):
        for_population("toddler_ish")


def test_deep_gm_check_is_na_outside_neonates():
    r = deep_gm_ratio_check(cfg=for_population("adult"))
    assert r.verdict == Verdict.NA


def test_neonatal_deep_gm_gradient_passes():
    """Miranda 2006: deep GM 30 vs cortical GM 16 -> ratio 1.88."""
    deep = np.zeros((6, 6, 2)); deep[:, :, 0] = 1.0
    cort = np.zeros((6, 6, 2)); cort[:, :, 1] = 1.0
    cbf = np.zeros((6, 6, 2))
    cbf[:, :, 0] = 30.0
    cbf[:, :, 1] = 16.0
    r = deep_gm_ratio_check(cbf=cbf, deep_gm=deep, cortical_gm=cort,
                            cfg=for_population("neonate_term"))
    assert abs(r.metric["deep_cortical_ratio"] - 1.875) < 1e-3
    assert r.verdict == Verdict.PASS


def test_neonatal_inverted_gradient_warns():
    """Cortical exceeding deep GM inverts the expected newborn pattern."""
    deep = np.zeros((6, 6, 2)); deep[:, :, 0] = 1.0
    cort = np.zeros((6, 6, 2)); cort[:, :, 1] = 1.0
    cbf = np.zeros((6, 6, 2))
    cbf[:, :, 0] = 16.0     # deep LOWER than cortical -> suspicious in a neonate
    cbf[:, :, 1] = 30.0
    r = deep_gm_ratio_check(cbf=cbf, deep_gm=deep, cortical_gm=cort,
                            cfg=for_population("neonate_term"))
    assert r.metric["deep_cortical_ratio"] < 1.0
    assert r.verdict == Verdict.WARN
    assert "invert" in r.reason.lower()


def test_deep_gm_needs_both_masks():
    r = deep_gm_ratio_check(cbf=np.ones((3, 3, 3)), cfg=for_population("neonate_term"))
    assert r.verdict == Verdict.UNKNOWN


# --------------------------------------------------------------------------- #
# Maria: organ modularity
# --------------------------------------------------------------------------- #
def test_kidney_profile_skips_the_brain_only_checks():
    """QEI's spatial template (2.5*GM + 1*WM) is a brain tissue model; it has no
    kidney equivalent, so it must not silently run on a kidney."""
    skipped = skipped_for_organ("kidney")
    assert "1.qei" in skipped
    assert "3.2.gm_wm_ratio" in skipped
    assert for_organ("kidney").organ == "kidney"


def test_brain_is_the_default_organ_and_skips_nothing():
    assert QCConfig().organ == "brain"
    assert skipped_for_organ("brain") == ()


def test_unknown_organ_raises():
    with pytest.raises(ValueError):
        for_organ("spleen")


# --------------------------------------------------------------------------- #
# Zhiliang: pathology vs artifact
# --------------------------------------------------------------------------- #
def _ratio_case(gm_val, wm_val):
    gm = np.zeros((4, 4, 2)); gm[:, :, 0] = 1.0
    wm = np.zeros((4, 4, 2)); wm[:, :, 1] = 1.0
    cbf = np.zeros((4, 4, 2))
    cbf[:, :, 0] = gm_val
    cbf[:, :, 1] = wm_val
    return cbf, gm, wm


def test_weak_ratio_warns_rather_than_fails():
    """A GE 3D spiral scan is smooth and legitimately lands near 1.2-1.3. Zhiliang's
    point: an altered WM CBF may be pathology, not an image failure. So a weak but
    >1 ratio must WARN, never FAIL."""
    cbf, gm, wm = _ratio_case(26.0, 20.0)      # ratio 1.3
    r = gm_wm_ratio_check(cbf=cbf, gm=gm, wm=wm)
    assert abs(r.metric["gm_wm_ratio"] - 1.3) < 1e-6
    assert r.verdict == Verdict.WARN
    assert "pathology" in r.metric["interpretation"]


def test_ratio_below_one_is_the_one_published_fail():
    """ratio < 1 is the only ratio rule with a source: ASLPrep excludes it."""
    cbf, gm, wm = _ratio_case(18.0, 20.0)      # ratio 0.9
    r = gm_wm_ratio_check(cbf=cbf, gm=gm, wm=wm)
    assert r.verdict == Verdict.FAIL


def test_non_strict_mode_softens_uncalibrated_fails_for_clinical_cohorts():
    cbf, gm, wm = _ratio_case(18.0, 20.0)
    r = gm_wm_ratio_check(cbf=cbf, gm=gm, wm=wm, cfg=QCConfig(strict=False))
    assert r.verdict == Verdict.WARN


# --------------------------------------------------------------------------- #
# Sudipto: quantification parameters must be user-supplied, never assumed
# --------------------------------------------------------------------------- #
def test_quantification_params_default_to_unknown_not_to_a_guess():
    """'How are you getting those parameters?' — we do not. They are None until
    the user supplies them, rather than silently assuming a default."""
    cfg = QCConfig()
    assert cfg.labeling_efficiency is None
    assert cfg.label_duration_s is None
    assert cfg.post_labeling_delay_s is None
    assert cfg.t1_blood_s is None


def test_quantification_params_are_settable():
    cfg = QCConfig(labeling_efficiency=0.85, label_duration_s=1.8,
                   post_labeling_delay_s=2.0, t1_blood_s=1.65)
    assert cfg.labeling_efficiency == 0.85
    assert cfg.post_labeling_delay_s == 2.0
