"""Known-answer tests for the 15 placenta checks.

The placenta has no consensus document at all, so almost every number in the
module is an engineering default. What these tests protect is therefore not the
numbers but the DESIGN DECISIONS that keep the module honest:

* units gate everything (P2.1), because the same physiology is published in
  mL/100g/min, %-of-M0 and arbitrary units;
* heterogeneity is physiology and is never graded (P2.3) - the opposite of the
  brain's spatial-CoV check;
* gestational age is mandatory context, so its absence is a definitional FAIL;
* the metrics that only work on structured images say so instead of reporting a
  confident number computed from noise (P6.3).
"""

import numpy as np
import pytest

from osipy_qc.core import Verdict
from osipy_qc.core.config import QCConfig
from osipy_qc.core.registry import all_checks
from osipy_qc.checks.placenta import (_holes_fraction, _unit_family,
                                      placenta_contraction_check,
                                      placenta_ga_context_check,
                                      placenta_implausible_check,
                                      placenta_labelling_check,
                                      placenta_m0_heterogeneity_check,
                                      placenta_m0_state_check,
                                      placenta_mask_integrity_check,
                                      placenta_pair_outliers_check,
                                      placenta_quant_constants_check,
                                      placenta_registration_check,
                                      placenta_segment_cov_check,
                                      placenta_slab_coverage_check,
                                      placenta_temporal_sd_check,
                                      placenta_units_check, placental_qei_check)
from osipy_qc.report import run_qc
from osipy_qc.synth import synthetic_placenta_case

CFG = QCConfig(organ="placenta")


@pytest.fixture(scope="module")
def clean():
    return synthetic_placenta_case(quality="clean", seed=0)


def test_all_15_placenta_checks_are_registered():
    names = list(all_checks("placenta"))
    assert len(names) == 15, names
    assert all(n.startswith("p") for n in names)


def test_a_placenta_run_never_includes_another_organs_checks():
    rep = run_qc({}, cfg=CFG)
    assert {r.check for r in rep.results} == set(all_checks("placenta"))


# --------------------------------------------------------------------------- #
# P1 - the empty quality-index slot
# --------------------------------------------------------------------------- #
def test_p1_is_unconditionally_na_with_a_null_score():
    r = placental_qei_check(perfusion_map=np.ones((4, 4, 4)))
    assert r.verdict is Verdict.NA
    assert r.metric["placental_qei"] is None


# --------------------------------------------------------------------------- #
# P2 - units, validity, distribution
# --------------------------------------------------------------------------- #
def test_unit_families_fold_synonyms():
    """Two spellings of the same quantity must not become two quantities."""
    assert _unit_family("mL/100g/min") == "per_mass"
    assert _unit_family("ml/min/100g") == "per_mass"
    assert _unit_family("%M0") == "percent_m0"
    assert _unit_family("a.u.") == "arbitrary"
    assert _unit_family(None) is None
    assert _unit_family("   ") is None


def test_p2_1_undeclared_units_only_fail_when_a_physiological_claim_is_made(clean):
    """A map in arbitrary units is legal and useful; grading it against a
    physiological bound is not."""
    quantified = placenta_units_check(perfusion_map=clean.perfusion, quantified=True)
    assert quantified.verdict is Verdict.FAIL
    unquantified = placenta_units_check(perfusion_map=clean.perfusion)
    assert unquantified.verdict is Verdict.WARN
    assert "without grading" in unquantified.reason


def test_p2_1_missing_constants_warn_but_do_not_fail(clean):
    r = placenta_units_check(perfusion_map=clean.perfusion, declared_units="mL/100g/min",
                             quantified=True, constants={"lambda": 0.9})
    assert r.verdict is Verdict.WARN
    assert set(r.metric["missing_constants"]) == {"alpha", "t1_blood_ms"}
    full = placenta_units_check(perfusion_map=clean.perfusion, declared_units="mL/100g/min",
                                quantified=True,
                                constants={"lambda": 0.9, "alpha": 0.767, "t1_blood_ms": 1650})
    assert full.verdict is Verdict.PASS


def test_p2_2_fence_is_data_driven_and_hand_checkable():
    """No published placental ceiling exists, so the upper bound is P75 + 3*IQR
    of this placenta's own distribution."""
    vals = np.concatenate([np.arange(100, dtype=float), [10_000.0]])
    perf = vals.reshape(101, 1, 1)
    mask = np.ones((101, 1, 1), bool)
    r = placenta_implausible_check(perfusion_map=perf, placenta_mask=mask,
                                   declared_units="mL/100g/min", cfg=CFG)
    p25, p75 = np.percentile(vals, [25, 75])
    assert r.metric["upper_fence"] == pytest.approx(p75 + 3.0 * (p75 - p25))
    assert r.metric["upper_outlier_fraction"] == pytest.approx(1 / 101)
    assert r.verdict is Verdict.PASS            # one outlier in 101 is under 5%


def test_p2_2_fails_only_when_the_map_is_mostly_absent():
    perf = np.full((10, 10, 2), np.nan)
    perf[:5] = 1.0                              # 50% finite exactly -> not yet a FAIL
    mask = np.ones((10, 10, 2), bool)
    assert placenta_implausible_check(perfusion_map=perf, placenta_mask=mask,
                                      cfg=CFG).verdict is not Verdict.FAIL
    perf[:5] = np.nan
    perf[0] = 1.0                               # 10% finite
    r = placenta_implausible_check(perfusion_map=perf, placenta_mask=mask, cfg=CFG)
    assert r.verdict is Verdict.FAIL
    assert "mostly absent" in r.reason


def test_p2_2_warns_on_a_majority_negative_map():
    c = synthetic_placenta_case(quality="garbage", seed=0)
    r = placenta_implausible_check(perfusion_map=c.perfusion, placenta_mask=c.placenta_mask,
                                   declared_units="mL/100g/min", cfg=CFG)
    assert r.verdict is Verdict.WARN
    assert r.metric["negative_fraction"] > 0.5


def test_p2_3_heterogeneity_is_reported_and_never_graded(clean):
    """Cotyledons and septa make a healthy placenta genuinely non-uniform;
    published healthy segment CoV is 0.58 +/- 0.10, a level that would be
    alarming in grey matter."""
    r = placenta_segment_cov_check(perfusion_map=clean.perfusion,
                                   placenta_mask=clean.placenta_mask, cfg=CFG)
    assert r.verdict is Verdict.INFO
    assert r.metric["graded"] is False
    assert r.metric["healthy_reference"] == {"mean": 0.58, "sd": 0.10}

    strongly_heterogeneous = synthetic_placenta_case(quality="borderline", seed=0)
    r2 = placenta_segment_cov_check(perfusion_map=strongly_heterogeneous.perfusion,
                                    placenta_mask=strongly_heterogeneous.placenta_mask, cfg=CFG)
    assert r2.verdict is Verdict.INFO           # still not a verdict
    assert r2.metric["segment_cov"] > r.metric["segment_cov"]


def test_p2_3_says_why_rather_than_printing_nan():
    c = synthetic_placenta_case(quality="garbage", seed=0)
    r = placenta_segment_cov_check(perfusion_map=c.perfusion, placenta_mask=c.placenta_mask,
                                   cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN
    assert "non-positive mean" in r.reason
    assert "nan" not in r.reason.lower()


def test_p2_3_needs_enough_complete_segments():
    small = np.ones((4, 4, 1))
    r = placenta_segment_cov_check(perfusion_map=small, placenta_mask=np.ones((4, 4, 1), bool),
                                   cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN and "complete" in r.reason


# --------------------------------------------------------------------------- #
# P3 - mask and coverage
# --------------------------------------------------------------------------- #
def test_holes_fraction_finds_enclosed_background_only():
    solid = np.zeros((9, 9, 9), bool)
    solid[2:7, 2:7, 2:7] = True
    assert _holes_fraction(solid) == pytest.approx(0.0)
    holed = solid.copy()
    holed[4, 4, 4] = False                       # one fully enclosed voxel
    n = int(holed.sum())
    assert _holes_fraction(holed) == pytest.approx(1 / (n + 1))
    # a dent open to the outside is NOT a hole
    dented = solid.copy()
    dented[2, 4, 4] = False
    assert _holes_fraction(dented) == pytest.approx(0.0)


def test_p3_1_records_provenance_and_flags_its_absence(clean):
    named = placenta_mask_integrity_check(placenta_mask=clean.placenta_mask,
                                          mask_source="manual, single rater",
                                          cfg=CFG)
    assert named.verdict is Verdict.PASS
    assert named.metric["mask_source"] == "manual, single rater"
    anonymous = placenta_mask_integrity_check(placenta_mask=clean.placenta_mask, cfg=CFG)
    assert anonymous.verdict is Verdict.WARN
    assert "provenance not stated" in anonymous.reason


def test_p3_1_grid_mismatch_and_empty_mask_are_fails(clean):
    mismatch = placenta_mask_integrity_check(placenta_mask=np.ones((4, 4, 4), bool),
                                             perfusion_map=np.ones((8, 8, 8)),
                                             mask_source="auto", cfg=CFG)
    assert mismatch.verdict is Verdict.FAIL and "resample" in mismatch.reason
    empty = placenta_mask_integrity_check(placenta_mask=np.zeros((4, 4, 4), bool),
                                          mask_source="auto", cfg=CFG)
    assert empty.verdict is Verdict.FAIL and "empty" in empty.reason


def test_p3_2_flags_a_placenta_clipped_by_the_slab_but_never_fails():
    clipped = np.zeros((8, 8, 4), bool)
    clipped[2:6, 2:6, :] = True                  # spans every slice, so it hits both faces
    r = placenta_slab_coverage_check(placenta_mask=clipped, cfg=CFG)
    assert r.verdict is Verdict.WARN
    assert "cut off by the slab" in r.reason
    assert r.metric["edge_voxel_fraction"] == pytest.approx(0.5)

    inside = np.zeros((8, 8, 6), bool)
    inside[2:6, 2:6, 2:4] = True
    assert placenta_slab_coverage_check(placenta_mask=inside, cfg=CFG).verdict is Verdict.PASS


def test_p3_2_compares_against_an_anatomical_mask_when_given():
    anat = np.zeros((8, 8, 6), bool)
    anat[2:6, 2:6, 1:5] = True                   # 64 voxels
    imaged = np.zeros((8, 8, 6), bool)
    imaged[2:6, 2:6, 2:4] = True                 # 32 of them
    r = placenta_slab_coverage_check(placenta_mask=imaged, anatomical_mask=anat, cfg=CFG)
    assert r.metric["covered_fraction_vs_anatomical"] == pytest.approx(0.5)
    assert r.verdict is Verdict.WARN


# --------------------------------------------------------------------------- #
# P4 - labelling scheme and gestational context
# --------------------------------------------------------------------------- #
def test_p4_1_undeclared_scheme_is_a_definitional_fail():
    """The placenta carries two circulations and the scheme decides which one
    was measured, so an undeclared scheme leaves the map's meaning undefined."""
    r = placenta_labelling_check(asl_4d=np.ones((4, 4, 4, 6)))
    assert r.verdict is Verdict.FAIL
    assert "two circulations" in r.reason


def test_p4_1_names_the_measured_compartment():
    """From the ISMRM review's own table. FAIR belongs with VSASL, NOT with
    pCASL: pCASL labels the maternal descending aorta to selectively label
    maternal perfusion, while VSASL and FAIR are both contributed to by maternal
    AND fetal flow. Grouping FAIR with pCASL told a reader the map described only
    the maternal circulation when it did not."""
    vsasl = placenta_labelling_check(labelling_scheme="VSASL",
                                     scheme_params={"cutoff_velocity_cm_s": 1.6,
                                                    "post_labeling_delay_s": 1.6})
    assert vsasl.verdict is Verdict.PASS
    assert vsasl.metric["measured_compartment"] == "maternal_and_fetal"
    fair = placenta_labelling_check(labelling_scheme="FAIR",
                                    scheme_params={"inversion_slab_thickness_mm": 100})
    assert fair.metric["measured_compartment"] == "maternal_and_fetal"
    pcasl = placenta_labelling_check(labelling_scheme="pCASL",
                                     scheme_params={"labelling_plane_position": "aortic bifurcation"})
    assert pcasl.metric["measured_compartment"] == "maternal"


def test_p4_1_missing_scheme_critical_params_warn():
    r = placenta_labelling_check(labelling_scheme="VSASL", scheme_params={})
    assert r.verdict is Verdict.WARN
    assert r.metric["missing_params"] == ["cutoff_velocity_cm_s", "post_labeling_delay_s"]


def test_p4_2_absent_gestational_age_is_a_definitional_fail():
    r = placenta_ga_context_check(maternal_position="lateral")
    assert r.verdict is Verdict.FAIL
    assert "changes across gestation" in r.reason


def test_p4_2_grades_the_studied_range_and_records_position():
    ok = placenta_ga_context_check(gestational_age_wk=30, maternal_position="lateral",
                                   field_strength_T=3, cfg=CFG)
    assert ok.verdict is Verdict.PASS
    early = placenta_ga_context_check(gestational_age_wk=8, maternal_position="lateral", cfg=CFG)
    assert early.verdict is Verdict.WARN and "outside the studied" in early.reason
    no_pos = placenta_ga_context_check(gestational_age_wk=30, cfg=CFG)
    assert no_pos.verdict is Verdict.WARN and "maternal position" in no_pos.reason
    assert ok.metric["magnitude_band_applied"] is False


# --------------------------------------------------------------------------- #
# P5 - M0 and quantification
# --------------------------------------------------------------------------- #
def test_p5_1_quantification_without_an_m0_is_a_fail():
    assert placenta_m0_state_check(quantified=True).verdict is Verdict.FAIL
    assert placenta_m0_state_check(quantified=False).verdict is Verdict.WARN


def test_p5_1_asymmetry_between_m0_and_asl_background_suppression():
    """BS must be OFF for the M0 but is expected ON for the ASL pairs."""
    m0 = np.ones((4, 4, 4))
    bad = placenta_m0_state_check(m0=m0, m0_labelled=False, m0_background_suppressed=True,
                                  quantified=True)
    assert bad.verdict is Verdict.FAIL and "crushed" in bad.reason
    soft = placenta_m0_state_check(m0=m0, m0_labelled=False, m0_background_suppressed=False,
                                   asl_background_suppressed=False, quantified=True)
    assert soft.verdict is Verdict.WARN          # only a WARN - no published rule
    assert "no published placental rule" in soft.reason


def test_p5_1_labelled_m0_is_never_a_calibration_image():
    r = placenta_m0_state_check(m0=np.ones((4, 4, 4)), m0_labelled=True,
                                m0_background_suppressed=False)
    assert r.verdict is Verdict.FAIL


def test_p5_2_voxelwise_normalisation_of_a_structured_m0_warns(clean):
    structured = clean.m0.copy().astype(float)
    zz, yy, xx = np.indices(structured.shape)
    # a steep sensitivity gradient - strong enough to push the in-mask CoV past
    # the structure-risk line, which a gentle ramp across a thin curved shell
    # does not do
    structured *= np.exp(2.0 * xx / xx.max())
    r = placenta_m0_heterogeneity_check(m0=structured, placenta_mask=clean.placenta_mask,
                                        normalisation_mode="voxel-wise", cfg=CFG)
    assert r.verdict is Verdict.WARN
    assert "imprinted" in r.reason
    scalar = placenta_m0_heterogeneity_check(m0=structured, placenta_mask=clean.placenta_mask,
                                             normalisation_mode="scalar", cfg=CFG)
    assert scalar.verdict is Verdict.PASS        # a scalar divide cannot imprint structure


def test_p5_3_checks_t1_blood_against_field_strength():
    """A T1-blood of 1650 ms at 1.5 T is a transcription error whatever the true
    placental value turns out to be."""
    good = placenta_quant_constants_check(constants={"lambda": 0.9, "alpha": 0.767,
                                                     "t1_blood_ms": 1650},
                                          field_strength_T=3)
    assert good.verdict is Verdict.PASS
    wrong = placenta_quant_constants_check(constants={"lambda": 0.9, "alpha": 0.767,
                                                      "t1_blood_ms": 1650},
                                           field_strength_T=1.5)
    assert wrong.verdict is Verdict.WARN and "not consistent" in wrong.reason


# --------------------------------------------------------------------------- #
# P6 - motion
# --------------------------------------------------------------------------- #
def test_p6_1_fails_when_too_few_pairs_survive():
    rng = np.random.default_rng(0)
    series = rng.normal(5.0, 1.0, (6, 6, 2, 6))
    series[..., :5] = np.nan
    r = placenta_pair_outliers_check(delta_m_4d=series,
                                     placenta_mask=np.ones((6, 6, 2), bool), cfg=CFG)
    assert r.verdict is Verdict.FAIL
    assert "not an average" in r.reason


def test_p6_1_passes_a_stable_series_and_names_its_rule():
    rng = np.random.default_rng(1)
    series = rng.normal(5.0, 0.5, (6, 6, 2, 12))
    r = placenta_pair_outliers_check(delta_m_4d=series,
                                     placenta_mask=np.ones((6, 6, 2), bool), cfg=CFG)
    assert r.verdict is Verdict.PASS
    assert "1.5 SD" in r.metric["rule"]
    # NOT 12. For Gaussian data P(|z| > 1.5) is about 13.4%, so the share of
    # deviating voxels sits just under the 20% limit and ordinary fluctuation
    # pushes the occasional pair over it. This is the documented behaviour of
    # the published rule - it fires on normal data, and the COUNT is the signal
    # rather than the firing - so the test asserts the verdict and the floor,
    # not a specific survivor count.
    assert r.metric["surviving_pairs"] >= CFG.placenta_good_surviving_pairs


def test_p6_2_reports_without_a_verdict_unless_the_cohort_is_comparable():
    rng = np.random.default_rng(2)
    wobbly = np.stack([np.full((5, 5, 2), 10.0 + 5.0 * t) + rng.normal(0, 0.1, (5, 5, 2))
                       for t in range(6)], -1)
    mask = np.ones((5, 5, 2), bool)
    plain = placenta_temporal_sd_check(delta_m_4d=wobbly, placenta_mask=mask, cfg=CFG)
    assert plain.verdict is Verdict.INFO
    assert "not cohort-comparable" in plain.reason
    graded = placenta_temporal_sd_check(delta_m_4d=wobbly, placenta_mask=mask,
                                        context={"cohort_comparable": True}, cfg=CFG)
    assert graded.verdict is Verdict.WARN
    assert graded.metric["reference"] == {"mean_pct": 6.7, "sd_pct": 3.1}


def test_p6_3_rigid_registration_is_flagged_because_the_placenta_deforms():
    rng = np.random.default_rng(3)
    base = rng.normal(10.0, 3.0, (8, 8, 3))       # structured enough for NCC
    series = np.stack([base + rng.normal(0, 0.2, base.shape) for _ in range(8)], -1)
    mask = np.ones((8, 8, 3), bool)
    rigid = placenta_registration_check(asl_source_4d=series, placenta_mask=mask,
                                        registration_model="rigid", cfg=CFG)
    assert rigid.verdict is Verdict.WARN
    assert "cannot in principle" in rigid.reason
    nonrigid = placenta_registration_check(asl_source_4d=series, placenta_mask=mask,
                                           registration_model="non-rigid (DSVR)", cfg=CFG)
    assert nonrigid.verdict is Verdict.PASS


def test_p6_3_does_not_grade_ncc_on_a_uniform_placenta():
    """NCC measures how well spatial STRUCTURE lines up, so it needs structure to
    exist. A uniform placenta gives a low NCC however perfect the registration
    is - the correlation is then between two noise fields."""
    rng = np.random.default_rng(4)
    flat = np.stack([np.full((8, 8, 3), 10.0) + rng.normal(0, 1.0, (8, 8, 3))
                     for _ in range(8)], -1)
    r = placenta_registration_check(asl_source_4d=flat, placenta_mask=np.ones((8, 8, 3), bool),
                                    registration_model="non-rigid", cfg=CFG)
    assert r.verdict is Verdict.INFO
    assert r.metric["ncc_informative"] is False
    assert "too little" in r.reason


def test_p6_4_occupancy_threshold_comes_from_outside_the_volume():
    """The reference level must come from the SERIES, not from each volume's own
    median. Using a volume's own median makes the statistic 0.5 for every volume
    by the definition of a median - a number that cannot detect anything."""
    rng = np.random.default_rng(5)
    vols = [np.full((8, 8, 2), 10.0) + rng.normal(0, 0.2, (8, 8, 2)) for _ in range(8)]
    vols[4] = np.full((8, 8, 2), 2.0) + rng.normal(0, 0.2, (8, 8, 2))   # signal collapses
    series = np.stack(vols, -1)
    r = placenta_contraction_check(asl_source_4d=series, placenta_mask=np.ones((8, 8, 2), bool),
                                   tr_s=6.0, cfg=CFG)
    assert r.verdict is Verdict.INFO             # always INFO
    assert r.metric["graded"] is False
    occ = r.metric["occupancy_per_volume"]
    assert occ[4] < 0.1 and occ[0] > 0.6, occ    # NOT 0.5 everywhere
    assert r.metric["candidate_event_volumes"] == [4]
    assert r.metric["candidate_event_times_s"][0] == pytest.approx(24.0)


def test_p6_4_does_not_surface_noise_as_a_contraction():
    """Occupancy is a proportion over a few hundred voxels, so its sampling error
    alone crosses the 10% line every few volumes. A candidate must also be
    outside the series' own robust variability, or a calm series reports phantom
    events."""
    mask = np.ones((8, 8, 2), bool)
    for seed in range(6):
        rng = np.random.default_rng(seed)
        calm = np.stack([np.full((8, 8, 2), 10.0) + rng.normal(0, 0.2, (8, 8, 2))
                         for _ in range(8)], -1)
        r = placenta_contraction_check(asl_source_4d=calm, placenta_mask=mask, cfg=CFG)
        assert r.metric["candidate_event_volumes"] == [], f"seed {seed}"

        withev = [np.full((8, 8, 2), 10.0) + rng.normal(0, 0.2, (8, 8, 2)) for _ in range(8)]
        withev[4] = np.full((8, 8, 2), 2.0) + rng.normal(0, 0.2, (8, 8, 2))
        r2 = placenta_contraction_check(asl_source_4d=np.stack(withev, -1),
                                        placenta_mask=mask, cfg=CFG)
        assert r2.metric["candidate_event_volumes"] == [4], f"seed {seed}"


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def _inputs(c):
    rng = np.random.default_rng(9)
    dm4 = np.stack([c.perfusion / 50 + rng.normal(0, 0.3, c.perfusion.shape) for _ in range(16)], -1)
    return dict(perfusion_map=c.perfusion, placenta_mask=c.placenta_mask, m0=c.m0,
                delta_m_4d=dm4, asl_source_4d=dm4, declared_units="mL/100g/min",
                quantified=True,
                constants={"lambda": 0.9, "alpha": 0.767, "t1_blood_ms": 1650},
                field_strength_T=3, labelling_scheme="VSASL",
                scheme_params={"cutoff_velocity_cm_s": 1.6, "post_labeling_delay_s": 1.6},
                gestational_age_wk=30, maternal_position="lateral",
                mask_source="manual, single rater", roi_definition="whole placenta",
                m0_labelled=False, m0_background_suppressed=False,
                asl_background_suppressed=True, normalisation_mode="scalar",
                registration_model="non-rigid (DSVR)", tr_s=6.0, m0_tr_s=8.0,
                context={"cohort_comparable": True})


def test_end_to_end_clean_passes_and_heterogeneity_is_not_penalised():
    """The borderline phantom is strongly heterogeneous, which is physiological -
    it must NOT be graded down for that."""
    clean_rep = run_qc(_inputs(synthetic_placenta_case(quality="clean", seed=0)), cfg=CFG)
    hetero_rep = run_qc(_inputs(synthetic_placenta_case(quality="borderline", seed=0)), cfg=CFG)
    assert clean_rep.overall is Verdict.PASS
    assert hetero_rep.overall is Verdict.PASS


def test_end_to_end_broken_subtraction_is_flagged():
    rep = run_qc(_inputs(synthetic_placenta_case(quality="garbage", seed=0)), cfg=CFG)
    assert rep.overall in (Verdict.WARN, Verdict.FAIL)
    flagged = {r.check for r in rep.results if r.verdict in (Verdict.WARN, Verdict.FAIL)}
    assert "p2.2.implausible_values" in flagged


def test_every_check_degrades_without_inputs_and_none_crashes():
    rep = run_qc({}, cfg=CFG)
    assert len(rep.results) == 15
    assert not [r for r in rep.results if "check error" in r.reason]
