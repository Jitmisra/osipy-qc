"""Known-answer tests for the 19 kidney checks.

Two things these tests protect, beyond "the code runs":

* **The consensus rules.** Nery 2020 R10.1 requires per-kidney cortical
  reporting and R10.2 declares medullary values unreliable. Several checks are
  deliberately weaker than they could be BECAUSE of those rules, and a future
  change that "improves" them by grading the medulla or pooling the kidneys
  would be a regression against the consensus, not an upgrade.
* **The never-FAIL contracts.** Where the design says a check can never FAIL,
  that is asserted here, because an uncalibrated threshold escalating to FAIL is
  the specific failure mode this project exists to avoid.

Anything traceable to real data is marked REAL-DATA and names the dataset.
"""

from dataclasses import replace as _replace

import numpy as np
import pytest

from osipy_qc.core import Verdict
from osipy_qc.core.config import QCConfig
from osipy_qc.core.registry import all_checks
from osipy_qc.checks.kidney import (_bs_state, _resolve_axes, _slice_axis,
                                    cortical_rbf_check,
                                    cortico_medullary_ratio_check,
                                    kidney_breathing_strategy_check,
                                    kidney_data_type_check,
                                    kidney_displacement_check,
                                    kidney_m0_clean_check,
                                    kidney_m0_present_check, kidney_m0_tr_check,
                                    kidney_mask_integrity_check,
                                    kidney_metadata_check,
                                    kidney_outlier_rate_check,
                                    kidney_registration_check,
                                    kidney_slice_coverage_check,
                                    kidney_swap_check,
                                    left_right_consistency_check,
                                    renal_implausible_check, renal_pws_check,
                                    renal_qei_check, renal_tsnr_check)
from osipy_qc.report import run_qc
from osipy_qc.synth import synthetic_kidney_case

CFG = QCConfig(organ="kidney")


@pytest.fixture(scope="module")
def clean():
    return synthetic_kidney_case(quality="clean", seed=0)


def _masks(c, kind="cortex"):
    return getattr(c, f"{kind}_masks")


# --------------------------------------------------------------------------- #
# registry / routing
# --------------------------------------------------------------------------- #
def test_all_19_kidney_checks_are_registered():
    names = list(all_checks("kidney"))
    assert len(names) == 19, names
    assert all(n.startswith("k") for n in names)


def test_a_kidney_run_never_includes_a_brain_check():
    """A kidney graded against a grey-matter spatial prior is not a wrong number,
    it is a meaningless one."""
    rep = run_qc({}, cfg=CFG)
    assert {r.check for r in rep.results} == set(all_checks("kidney"))
    assert not [r for r in rep.results if r.check[0].isdigit()]


# --------------------------------------------------------------------------- #
# K1 - the empty quality-index slot
# --------------------------------------------------------------------------- #
def test_k1_is_unconditionally_na_with_a_null_score():
    """N/A regardless of inputs, and the score is null - a 0.0 would read as
    'worst possible quality' rather than 'does not exist'."""
    for kwargs in ({}, {"rbf_map": np.ones((4, 4, 4))}):
        r = renal_qei_check(**kwargs)
        assert r.verdict is Verdict.NA
        assert r.metric["renal_qei"] is None
        assert len(r.metric["blockers"]) == 4


# --------------------------------------------------------------------------- #
# K2 - noise, signal magnitude, distribution
# --------------------------------------------------------------------------- #
def test_k2_1_tsnr_is_the_mean_of_voxelwise_ratios_not_the_ratio_of_means():
    """The two differ whenever the ROI is heterogeneous, which a kidney always
    is. Built so the difference is unmissable: two voxels, opposite behaviour."""
    series = np.zeros((2, 1, 1, 4))
    series[0, 0, 0, :] = [10.0, 10.0, 10.0, 10.0]      # mean 10, sd 0 -> excluded (sd<=0)
    series[1, 0, 0, :] = [1.0, 3.0, 1.0, 3.0]          # mean 2, sd(ddof=1) = 1.1547
    mask = {"left": np.ones((2, 1, 1), bool)}
    r = renal_tsnr_check(delta_m_4d=series, kidney_masks=mask, cfg=CFG)
    assert r.verdict is Verdict.INFO                    # report-only, always
    expected = 2.0 / np.std([1.0, 3.0, 1.0, 3.0], ddof=1)
    assert r.metric["per_kidney"]["left"]["tsnr"] == pytest.approx(expected)
    assert r.metric["per_kidney"]["left"]["n_voxels"] == 1   # the sd=0 voxel dropped


def test_k2_1_never_grades():
    """No renal tSNR cutoff exists and published values span ~7x."""
    c = synthetic_kidney_case(quality="garbage", seed=0)
    rng = np.random.default_rng(0)
    s = np.stack([c.rbf + rng.normal(0, 50, c.rbf.shape) for _ in range(6)], -1)
    r = renal_tsnr_check(delta_m_4d=s, kidney_masks=c.kidney_masks, cfg=CFG)
    assert r.verdict is Verdict.INFO
    assert "not graded" in r.reason


def test_k2_1_needs_three_repetitions():
    s = np.ones((2, 2, 2, 2))
    r = renal_tsnr_check(delta_m_4d=s, kidney_masks={"left": np.ones((2, 2, 2), bool)}, cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN


def test_k2_2_pws_is_a_ratio_of_roi_means(clean):
    """REAL-DATA anchor: the phantom is built to give exactly 3.00% cortical PWS,
    which is where garciaruiz2025 measures it (2.95 +/- 0.56% at 1.5 T)."""
    r = renal_pws_check(delta_m=clean.delta_m, m0=clean.m0,
                        kidney_masks=clean.kidney_masks, cortex_masks=clean.cortex_masks,
                        pld_or_ti_s=1.4, cfg=CFG)
    assert r.verdict is Verdict.PASS
    assert r.metric["per_kidney"]["left"]["pws_cortex_pct"] == pytest.approx(3.0, abs=0.15)


def test_k2_2_without_a_pld_reports_but_does_not_grade(clean):
    """PWS falls ~3x between PLD 0.5 s and 1.5 s in the same subjects, so an
    ungated band is not a defensible test."""
    r = renal_pws_check(delta_m=clean.delta_m, m0=clean.m0,
                        kidney_masks=clean.kidney_masks, cortex_masks=clean.cortex_masks,
                        cfg=CFG)
    assert r.verdict is Verdict.INFO
    assert "cannot be applied honestly" in r.reason


def test_k2_2_never_fails_even_on_a_broken_subtraction():
    c = synthetic_kidney_case(quality="garbage", seed=0)
    r = renal_pws_check(delta_m=c.delta_m, m0=c.m0, kidney_masks=c.kidney_masks,
                        cortex_masks=c.cortex_masks, pld_or_ti_s=1.4, cfg=CFG)
    assert r.verdict is Verdict.WARN            # never FAIL
    assert "subtraction-sign" in r.reason


def test_k2_2_is_gated_on_a_shared_grid(clean):
    r = renal_pws_check(delta_m=clean.delta_m, m0=clean.m0,
                        kidney_masks=clean.kidney_masks, grids_match=False, cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN
    assert "interpolation" in r.reason


def test_k2_3_denominator_is_every_roi_voxel_including_nonfinite():
    """NaN < 0 is False, so non-finite voxels dilute the fraction rather than
    inflating it - the conservative direction."""
    rbf = np.array([-1.0, 1.0, np.nan, 1.0]).reshape(4, 1, 1)
    m = {"left": np.ones((4, 1, 1), bool)}
    r = renal_implausible_check(rbf_map=rbf, kidney_masks=m, units="mL/100g/min",
                                cfg=QCConfig(organ="kidney", kidney_min_roi_voxels=1))
    assert r.metric["per_kidney"]["left"]["neg_frac"] == pytest.approx(0.25)   # 1 of 4, not 1 of 3


def test_k2_3_boundaries_are_exact():
    """PASS is '< 5%'; 5% itself is WARN. FAIL is '> 20%'; 20% itself is WARN."""
    cfg = QCConfig(organ="kidney", kidney_min_roi_voxels=1)

    def frac(neg, total):
        v = np.array([-1.0] * neg + [1.0] * (total - neg)).reshape(total, 1, 1)
        return renal_implausible_check(rbf_map=v, kidney_masks={"left": np.ones((total, 1, 1), bool)},
                                       units="mL/100g/min", cfg=cfg)
    assert frac(4, 100).verdict is Verdict.PASS      # 4%
    assert frac(5, 100).verdict is Verdict.WARN      # exactly 5%
    assert frac(20, 100).verdict is Verdict.WARN     # exactly 20%
    assert frac(21, 100).verdict is Verdict.FAIL     # > 20%


def test_k2_3_requires_declared_units():
    r = renal_implausible_check(rbf_map=np.ones((4, 4, 4)),
                                kidney_masks={"left": np.ones((4, 4, 4), bool)}, cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN
    assert "units" in r.reason


# --------------------------------------------------------------------------- #
# K3 - level and contrast
# --------------------------------------------------------------------------- #
def test_k3_1_good_outcome_is_info_not_pass(clean):
    """A PASS would assert 'this perfusion level is normal', which no evidence
    supports - the published 139-427 spread is study-level cohort MEANS whose
    healthy and patient ranges overlap almost entirely."""
    r = cortical_rbf_check(rbf_map=clean.rbf, cortex_masks=clean.cortex_masks,
                           units="mL/100g/min", cfg=CFG)
    assert r.verdict is Verdict.INFO
    assert r.metric["per_kidney"]["left"]["cortex_mean"] == pytest.approx(300, abs=5)
    assert "no normality claim" in r.reason


def test_k3_1_warns_outside_the_sanity_bound_and_never_fails():
    c = synthetic_kidney_case(quality="garbage", seed=0)
    r = cortical_rbf_check(rbf_map=c.rbf, cortex_masks=c.cortex_masks,
                           units="mL/100g/min", cfg=CFG)
    assert r.verdict is Verdict.WARN            # negative cortical mean, still not FAIL
    assert "sanity bound" in r.reason


def test_k3_1_with_only_whole_kidney_masks_reports_but_refuses_to_grade(clean):
    """R10.1 asks for cortex; grading the whole-kidney mean under the same name
    would be the more damaging error."""
    r = cortical_rbf_check(rbf_map=clean.rbf, kidney_masks=clean.kidney_masks,
                           units="mL/100g/min", cfg=CFG)
    assert r.verdict is Verdict.NA
    assert r.metric["graded"] is False
    assert "whole-kidney" in r.reason


def test_k3_2_cmr_is_an_integrity_flag_and_never_fails(clean):
    """R10.2 (89% agreement) forbids reading the medullary half as perfusion."""
    r = cortico_medullary_ratio_check(rbf_map=clean.rbf, cortex_masks=clean.cortex_masks,
                                      medulla_masks=clean.medulla_masks, cfg=CFG)
    assert r.verdict is Verdict.INFO
    assert r.metric["per_kidney"]["left"]["cmr"] == pytest.approx(2.5, abs=0.1)
    assert "no perfusion claim" in r.reason

    swapped = cortico_medullary_ratio_check(rbf_map=clean.rbf,
                                            cortex_masks=clean.medulla_masks,
                                            medulla_masks=clean.cortex_masks, cfg=CFG)
    assert swapped.verdict is Verdict.WARN      # never FAIL
    assert "swapped" in swapped.reason


def test_k3_2_without_a_medulla_mask_is_na_not_a_gap(clean):
    r = cortico_medullary_ratio_check(rbf_map=clean.rbf, cortex_masks=clean.cortex_masks,
                                      cfg=CFG)
    assert r.verdict is Verdict.NA
    assert "consensus default" in r.reason


def test_k3_3_asymmetry_and_the_single_kidney_case(clean):
    r = left_right_consistency_check(rbf_map=clean.rbf, cortex_masks=clean.cortex_masks,
                                     cfg=CFG)
    assert r.verdict is Verdict.PASS            # phantom sides differ by 3%
    assert r.metric["asymmetry_pct"] < 20

    one = left_right_consistency_check(rbf_map=clean.rbf,
                                       cortex_masks={"left": clean.cortex_masks["left"]},
                                       cfg=CFG)
    assert one.verdict is Verdict.NA            # transplant/nephrectomy is ordinary


def test_k3_3_never_fails_on_a_lopsided_pair():
    c = synthetic_kidney_case(quality="borderline", seed=0)   # left 69 vs right 300
    r = left_right_consistency_check(rbf_map=c.rbf, cortex_masks=c.cortex_masks, cfg=CFG)
    assert r.verdict is Verdict.WARN
    assert "may be one-sided disease" in r.reason


# --------------------------------------------------------------------------- #
# K4 - masks, registration, coverage
# --------------------------------------------------------------------------- #
def test_k4_1_passes_a_clean_pair_and_fails_the_impossible(clean):
    assert kidney_mask_integrity_check(kidney_masks=clean.kidney_masks,
                                       cortex_masks=clean.cortex_masks,
                                       cfg=CFG).verdict is Verdict.PASS
    empty = kidney_mask_integrity_check(kidney_masks={"left": np.zeros((4, 4, 4), bool)}, cfg=CFG)
    assert empty.verdict is Verdict.FAIL and "empty" in empty.reason
    overlap = kidney_mask_integrity_check(cortex_masks=clean.cortex_masks,
                                          medulla_masks=clean.cortex_masks, cfg=CFG)
    assert overlap.verdict is Verdict.FAIL and "overlap" in overlap.reason


def test_k4_1_does_not_flag_the_medulla_for_being_multi_component():
    """REAL-DATA (renaldro / iBEAt): a real medulla mask is ~8 separate renal
    pyramids per kidney, so the single-dominant-object rule that is right for a
    kidney and its cortex is an anatomical false positive here."""
    pyramids = np.zeros((20, 20, 20), bool)
    for i in range(4):                          # four disjoint blobs
        pyramids[2 + 4 * i:4 + 4 * i, 2:4, 2:4] = True
    r = kidney_mask_integrity_check(medulla_masks={"left": pyramids}, cfg=CFG)
    assert r.verdict is Verdict.PASS
    assert r.metric["per_mask"]["medulla_left"]["n_components"] == 4

    # the same shape as a CORTEX mask is a real finding
    r2 = kidney_mask_integrity_check(cortex_masks={"left": pyramids}, cfg=CFG)
    assert r2.verdict is Verdict.WARN and "fragmented" in r2.reason


def test_k4_1_flags_an_organ_clipped_by_the_field_of_view():
    clipped = np.zeros((8, 8, 8), bool)
    clipped[0:3, 2:5, 2:5] = True               # touches face 0
    r = kidney_mask_integrity_check(kidney_masks={"left": clipped}, cfg=CFG)
    assert r.verdict is Verdict.WARN and "field of view" in r.reason


def test_k4_2_fails_only_on_a_geometry_mismatch_with_no_transform():
    """The ONLY FAIL in this check. An earlier version also FAILed on a Dice
    below an uncalibrated cut-off - an engineering default driving a hard
    failure, which is exactly what this project forbids."""
    r = kidney_registration_check(rbf_map=np.ones((4, 4, 4)), m0=np.ones((8, 8, 8)), cfg=CFG)
    assert r.verdict is Verdict.FAIL
    assert r.metric["shapes_match"] is False
    # a transform rescues it
    ok = kidney_registration_check(rbf_map=np.ones((4, 4, 4)), m0=np.ones((8, 8, 8)),
                                   transforms=[object()], cfg=CFG)
    assert ok.verdict is not Verdict.FAIL


def test_k4_2_compares_affines_not_only_matrix_sizes():
    """The doc's stated reason: two images can share 96x96x5 on DIFFERENT voxel
    sizes, so a matrix-size test alone is necessary but not sufficient."""
    same_shape = dict(rbf_map=np.ones((8, 8, 4)), m0=np.ones((8, 8, 4)), cfg=CFG)
    r = kidney_registration_check(affine=np.diag([2., 2., 8., 1.]),
                                  m0_affine=np.diag([2., 2., 4., 1.]), **same_shape)
    assert r.verdict is Verdict.FAIL
    assert r.metric["affine_max_diff_mm"] == pytest.approx(4.0)
    # within the 0.01 mm float round-trip tolerance -> not a mismatch
    close = kidney_registration_check(affine=np.diag([2., 2., 8., 1.]),
                                      m0_affine=np.diag([2., 2., 8.005, 1.]), **same_shape)
    assert close.verdict is not Verdict.FAIL
    assert close.metric["affines_match"] is True


def test_k4_2_flags_a_single_global_transform_for_both_kidneys():
    """R8.1: the kidneys move independently, so one rigid transform cannot serve
    both."""
    A = np.diag([2., 2., 8., 1.])
    common = dict(rbf_map=np.ones((8, 8, 4)), m0=np.ones((8, 8, 4)),
                  affine=A, m0_affine=A, cfg=CFG)
    assert kidney_registration_check(registration_scope="global", **common).verdict is Verdict.WARN
    assert kidney_registration_check(registration_scope="per_kidney", **common).verdict is Verdict.PASS
    # no provenance and nothing to measure -> INFO, not a silent PASS
    assert kidney_registration_check(**common).verdict is Verdict.INFO


def test_k4_3_counts_usable_slices(clean):
    r = kidney_slice_coverage_check(rbf_map=clean.rbf, kidney_masks=clean.kidney_masks, cfg=CFG)
    assert r.verdict is Verdict.PASS
    assert r.metric["per_kidney"]["left"]["usable_fraction"] == pytest.approx(1.0)

    holed = clean.rbf.copy()
    holed[:, :, :12] = 0.0                      # kill most slices
    r2 = kidney_slice_coverage_check(rbf_map=holed, kidney_masks=clean.kidney_masks, cfg=CFG)
    assert r2.verdict in (Verdict.WARN, Verdict.FAIL)


def test_k4_3_is_na_for_a_single_slice_readout(clean):
    r = kidney_slice_coverage_check(rbf_map=clean.rbf, kidney_masks=clean.kidney_masks,
                                    readout="2D single-slice", cfg=CFG)
    assert r.verdict is Verdict.NA


# --------------------------------------------------------------------------- #
# K5 - metadata, routing, control/label order
# --------------------------------------------------------------------------- #
def test_k5_1_never_fails_and_names_the_consequence_of_each_gap():
    r = kidney_metadata_check(sidecar={"ArterialSpinLabelingType": "pCASL"})
    assert r.verdict is Verdict.WARN            # never FAIL - renal ASL has no BIDS
    assert "pld_or_ti" in r.metric["missing"]
    assert "K2.2" in r.metric["consequences"]["pld_or_ti"]


def test_k5_2_classifies_masks_before_images():
    """A renal dataset ships cortex_label.nii.gz and kidney_mask.nii.gz, whose
    names carry no modality token. Reading those as images is how a mask ends up
    graded as a perfusion map."""
    r = kidney_data_type_check(files=[{"name": "cortex_label.nii.gz"},
                                      {"name": "kidney_mask_left.nii.gz"},
                                      {"name": "ASL_RBF.nii.gz"}])
    assert r.verdict is Verdict.INFO
    assert r.metric["roles"]["cortex_mask"] == ["cortex_label.nii.gz"]
    assert r.metric["roles"]["rbf_map"] == ["ASL_RBF.nii.gz"]


def test_k5_3_counts_pairs_rather_than_pooling_means():
    """A breathing kidney moves between the halves of a pair, so a pooled mean
    can be dominated by a few badly-displaced volumes and report the wrong sign
    with a confident number."""
    rng = np.random.default_rng(0)
    good = np.stack([(110.0 if t % 2 == 0 else 100.0) + rng.normal(0, 0.5, (4, 4, 2))
                     for t in range(12)], -1)
    m = {"left": np.ones((4, 4, 2), bool)}
    r = kidney_swap_check(asl_4d=good, kidney_masks=m, cfg=CFG)
    assert r.verdict is Verdict.PASS
    assert r.metric["fraction_control_brighter"] == pytest.approx(1.0)

    swapped = good[..., ::-1].copy()             # label-first
    r2 = kidney_swap_check(asl_4d=swapped, kidney_masks=m, cfg=CFG)
    assert r2.verdict is Verdict.FAIL


def test_k5_3_uses_aslcontext_when_supplied():
    """REAL-DATA (Brumer heiDATA renal phantom): that dataset is label-first
    (ImageOrder ['label','control']), and told the order the check passes while
    untold it correctly reports the ordering as wrong."""
    rng = np.random.default_rng(1)
    label_first = np.stack([(100.0 if t % 2 == 0 else 110.0) + rng.normal(0, 0.5, (4, 4, 2))
                            for t in range(12)], -1)
    m = {"left": np.ones((4, 4, 2), bool)}
    told = kidney_swap_check(asl_4d=label_first, kidney_masks=m,
                             aslcontext_rows=["label", "control"] * 6, cfg=CFG)
    assert told.verdict is Verdict.PASS
    assert told.metric["assumption"] == "volume order from aslcontext.tsv"
    assert kidney_swap_check(asl_4d=label_first, kidney_masks=m, cfg=CFG).verdict is Verdict.FAIL


def test_k5_3_is_na_under_background_suppression():
    r = kidney_swap_check(asl_4d=np.ones((4, 4, 2, 8)), kidney_masks={"left": np.ones((4, 4, 2), bool)},
                          background_suppression=True, cfg=CFG)
    assert r.verdict is Verdict.NA


# --------------------------------------------------------------------------- #
# K6 - M0
# --------------------------------------------------------------------------- #
def test_k6_1_is_stricter_than_the_brain_for_a_quantified_map():
    """The brain WARNs on an absent M0 in every case. Here, an absent M0 under a
    map presented as quantified means the units are not the units it claims."""
    assert kidney_m0_present_check(m0_type="separate").verdict is Verdict.PASS
    assert kidney_m0_present_check(m0_type="absent", rbf_supplied=True).verdict is Verdict.FAIL
    assert kidney_m0_present_check(m0_type="absent").verdict is Verdict.WARN
    assert kidney_m0_present_check().verdict is Verdict.UNKNOWN


def test_k6_2_reads_a_background_suppression_pulse_count():
    """REAL-DATA (Brumer): the sidecar carries "BackgroundSuppression": 2, an
    integer pulse count where BIDS types a boolean."""
    assert _bs_state(2) is True and _bs_state(0) is False
    assert _bs_state(True) is True and _bs_state(None) is None
    assert _bs_state("yes") is None              # not interpreted, never guessed
    r = kidney_m0_clean_check(m0_background_suppression=2, m0_labelling_applied=False)
    assert r.verdict is Verdict.FAIL
    assert r.metric["m0_background_suppression_read_as"] is True


def test_k6_2_missing_flags_beat_a_matching_readout():
    """A matching readout cannot license a PASS on an M0 whose state nobody knows."""
    r = kidney_m0_clean_check(m0_background_suppression=False, m0_readout="2D", asl_readout="2D")
    assert r.verdict is Verdict.UNKNOWN
    assert "labelling" in r.reason


def test_k6_3_never_fails_and_reports_both_compartments():
    """Cortex and medulla have materially different T1, so one correction factor
    cannot serve both."""
    ok = kidney_m0_tr_check(m0_tr_s=6.0, cfg=CFG)
    assert ok.verdict is Verdict.PASS
    short = kidney_m0_tr_check(m0_tr_s=2.0, field_T=3, cfg=CFG)
    assert short.verdict is Verdict.WARN         # never FAIL - it is correctable
    f = short.metric["correction_factors"]
    assert f["cortex"] != f["medulla"] and short.metric["factor_spread"] > 0
    # The T1 defaults are the wolf2018 published range midpoints, not invented:
    # 3 T cortex 1124-1406 -> 1265 ms, medulla 1388-1685 -> 1537 ms.
    assert short.metric["t1_ms_used"] == {"cortex": 1265.0, "medulla": 1537.0}
    assert "wolf2018" in short.metric["t1_source"]
    # hand-check the physics: 1 / (1 - exp(-TR/T1)) with TR in s and T1 in ms
    assert f["cortex"] == pytest.approx(1 / (1 - np.exp(-2000.0 / 1265.0)))

    # and reproduce the design doc's own worked example: TR 3.0 s gives x1.103
    # (cortex) to x1.165 (medulla) at 3 T
    doc = kidney_m0_tr_check(m0_tr_s=3.0, field_T=3, cfg=CFG)
    assert doc.metric["correction_factors"]["cortex"] == pytest.approx(1.103, abs=0.001)
    assert doc.metric["correction_factors"]["medulla"] == pytest.approx(1.165, abs=0.002)


# --------------------------------------------------------------------------- #
# K7 - respiratory motion
# --------------------------------------------------------------------------- #
def test_k7_1_resolves_anatomical_axes_from_the_affine():
    """Renal acquisitions are routinely coronal or oblique, so the array's third
    axis is not reliably cranio-caudal - and CC carries the respiratory
    excursion (~6.5x the RL motion)."""
    axes = _resolve_axes(np.diag([2.0, 2.0, 8.0, 1.0]))
    assert axes == {"RL": 0, "AP": 1, "CC": 2}
    # a coronal-style affine that maps world z onto array axis 1
    swapped = _resolve_axes(np.array([[2., 0, 0, 0], [0, 0, 2., 0], [0, 2., 0, 0], [0, 0, 0, 1]]))
    assert swapped["CC"] == 1


def test_k7_1_slice_axis_is_not_guessed_from_isotropic_voxels():
    """argmax over equal voxel sizes returns axis 0 by tie-break, and a
    through-plane share measured along an arbitrary axis is meaningless."""
    axis, why = _slice_axis((3.0, 3.0, 3.0))
    assert axis is None and "near-isotropic" in why
    axis2, why2 = _slice_axis((2.0, 2.0, 8.0))
    assert axis2 == 2 and "anisotropy" in why2
    assert _slice_axis((3.0, 3.0, 3.0), supplied=1)[0] == 1


def test_k7_1_still_organ_passes_and_moving_organ_warns(clean):
    rng = np.random.default_rng(3)
    aff = np.diag([2.0, 2.0, 8.0, 1.0])
    still = np.stack([clean.delta_m + rng.normal(0, 3, clean.delta_m.shape) for _ in range(10)], -1)
    r = kidney_displacement_check(delta_m_4d=still, kidney_masks=clean.kidney_masks,
                                  voxel_mm=(2.0, 2.0, 8.0), affine=aff, cfg=CFG)
    assert r.verdict is Verdict.PASS
    # the through-plane FRACTION is gated: direction is undefined without motion
    assert np.isnan(r.metric["per_kidney"]["left"]["through_plane_fraction"])

    moving = np.stack([np.roll(clean.delta_m + rng.normal(0, 3, clean.delta_m.shape),
                               int(round(4 * np.sin(t))), axis=2) for t in range(10)], -1)
    r2 = kidney_displacement_check(delta_m_4d=moving, kidney_masks=clean.kidney_masks,
                                   voxel_mm=(2.0, 2.0, 8.0), affine=aff, cfg=CFG)
    assert r2.verdict is Verdict.WARN           # never FAIL on motion severity
    assert r2.metric["per_kidney"]["left"]["per_axis"]["CC"]["median_vox"] > 0.5


def test_k7_1_needs_an_affine():
    r = kidney_displacement_check(delta_m_4d=np.ones((4, 4, 4, 6)),
                                  kidney_masks={"left": np.ones((4, 4, 4), bool)}, cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN and "affine" in r.reason


def test_k7_2_names_the_rule_that_produced_the_count():
    """Four published variants exist and they disagree, so a report must never
    hide which one was applied."""
    rng = np.random.default_rng(4)
    series = rng.normal(10.0, 1.0, (6, 6, 3, 12))
    series[..., 5] += 50.0                       # one grossly deviant repetition
    m = {"left": np.ones((6, 6, 3), bool)}
    r = kidney_outlier_rate_check(delta_m_4d=series, kidney_masks=m, cfg=CFG)
    assert r.metric["rule"] == "harteveld_2sd_20pct"
    assert 5 in r.metric["per_kidney"]["left"]["rejected_indices"]
    assert "the count is the signal, not the firing" in r.metric["calibration_note"]


def test_k7_2_fails_when_too_few_pairs_survive():
    """The FAIL is definitional: fewer than two surviving pairs leaves nothing
    to average. Reached here through unusable (all non-finite) repetitions,
    which is how it happens in practice."""
    rng = np.random.default_rng(5)
    series = rng.normal(10.0, 1.0, (4, 4, 2, 6))
    series[..., :5] = np.nan                     # only one repetition is usable
    r = kidney_outlier_rate_check(delta_m_4d=series, kidney_masks={"left": np.ones((4, 4, 2), bool)},
                                  cfg=CFG)
    assert r.verdict is Verdict.FAIL
    assert "nothing left to average" in r.reason


def test_k7_2_rule_is_relative_and_cannot_flag_a_uniformly_bad_series():
    """A property of the PUBLISHED rule, documented rather than worked around.

    The +/-2 SD test compares each repetition against the temporal SD of the
    series itself. If most repetitions are wild, the SD they produce is wild too,
    and nothing exceeds it - so a uniformly corrupted series is reported as clean
    by this check alone. That is why the design pairs it with K2.1 (tSNR) and
    K2.3 (implausible values), which are absolute rather than relative, and why
    a PASS here is never on its own evidence that the acquisition was stable."""
    rng = np.random.default_rng(6)
    ramp = rng.normal(10.0, 0.01, (4, 4, 2, 6))
    for t in range(6):
        ramp[..., t] += 100.0 * t                # every repetition wildly different
    r = kidney_outlier_rate_check(delta_m_4d=ramp, kidney_masks={"left": np.ones((4, 4, 2), bool)},
                                  cfg=CFG)
    assert r.verdict is Verdict.PASS
    assert r.metric["per_kidney"]["left"]["n_rejected"] == 0


def test_k7_2_is_na_for_pre_subtracted_data():
    r = kidney_outlier_rate_check(delta_m_4d=np.ones((4, 4, 2, 8)),
                                  kidney_masks={"left": np.ones((4, 4, 2), bool)},
                                  structure="pre-subtracted deltaM", cfg=CFG)
    assert r.verdict is Verdict.NA


def test_k7_3_flags_breath_hold_without_rejecting_it():
    assert kidney_breathing_strategy_check(breathing_strategy="free breathing").verdict is Verdict.PASS
    bh = kidney_breathing_strategy_check(breathing_strategy="breath-hold")
    assert bh.verdict is Verdict.WARN and "not rejected" in bh.reason
    assert kidney_breathing_strategy_check().verdict is Verdict.UNKNOWN


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def _full_inputs(c):
    rng = np.random.default_rng(7)
    dm4 = np.stack([c.delta_m + rng.normal(0, 3, c.delta_m.shape) for _ in range(12)], -1)
    return dict(rbf_map=c.rbf, m0=c.m0, delta_m=c.delta_m, delta_m_4d=dm4,
                kidney_masks=c.kidney_masks, cortex_masks=c.cortex_masks,
                medulla_masks=c.medulla_masks, units="mL/100g/min", pld_or_ti_s=1.4,
                voxel_mm=(2.0, 2.0, 8.0), affine=np.diag([2.0, 2.0, 8.0, 1.0]),
                m0_type="separate", m0_tr_s=6.0, m0_background_suppression=False,
                m0_labelling_applied=False, breathing_strategy="free breathing",
                readout="3D", sidecar={"ArterialSpinLabelingType": "pCASL",
                                       "MagneticFieldStrength": 3, "PostLabelingDelay": 1.4},
                context={"labelling": "pCASL", "field_strength_t": 3, "readout": "3D"},
                files=[{"name": "ASL_RBF.nii.gz"}])


def test_end_to_end_quality_ordering():
    """A clean phantom must not be worse than a garbage one, and the garbage one
    must fail on the sign - the only physically-licensed FAIL in the module."""
    reports = {q: run_qc(_full_inputs(synthetic_kidney_case(quality=q, seed=0)), cfg=CFG)
               for q in ("clean", "borderline", "garbage")}
    assert reports["garbage"].overall is Verdict.FAIL
    assert reports["clean"].overall in (Verdict.PASS, Verdict.INFO, Verdict.WARN)
    bad = {r.check for r in reports["garbage"].results if r.verdict is Verdict.FAIL}
    assert bad == {"k2.3.implausible_values"}

    # the borderline case must flag its lopsided left kidney in both places
    b = {r.check: r for r in reports["borderline"].results}
    assert b["k3.3.left_right"].verdict is Verdict.WARN
    assert b["k3.2.cmr"].verdict is Verdict.WARN


def test_every_check_reports_without_inputs_and_none_crashes():
    """A check must degrade, never raise: run_qc's blanket except would turn a
    raise into 'check error: ...', a stack-trace fragment where a reader needs
    an instruction."""
    rep = run_qc({}, cfg=CFG)
    assert len(rep.results) == 19
    assert not [r for r in rep.results if "check error" in r.reason]
    assert all(r.verdict in (Verdict.UNKNOWN, Verdict.NA) for r in rep.results)


def test_k4_3_resolves_the_slice_axis_instead_of_assuming_axis_2():
    """Renal acquisitions are routinely coronal or oblique, so array axis 2 is
    not reliably the slice direction - counting "slices" along the wrong axis
    counts something that is not a slice."""
    rbf = np.full((6, 20, 20), 300.0)
    mask = np.ones((6, 20, 20), bool)
    rbf[2] = 0.0                                 # one dead slice, along axis 0
    r = kidney_slice_coverage_check(rbf_map=rbf, kidney_masks={"left": mask},
                                    voxel_mm=(8.0, 2.0, 2.0), cfg=CFG)
    assert r.metric["slice_axis"] == 0
    assert r.metric["per_kidney"]["left"]["usable_fraction"] == pytest.approx(5 / 6)
    # near-isotropic voxels carry no evidence of a slice direction; the fallback
    # is stated in the metric rather than hidden
    iso = kidney_slice_coverage_check(rbf_map=rbf, kidney_masks={"left": mask},
                                      voxel_mm=(2.0, 2.0, 2.0), cfg=CFG)
    assert iso.metric["slice_axis"] == 2
    assert "could not be inferred" in iso.metric["slice_axis_source"]


def test_k4_3_zero_usable_slices_is_unknown_not_fail():
    """The spec: "FAIL never on the fraction. Fewer than one usable slice per
    kidney is instead reported as UNKNOWN, since no ROI statistic exists to
    grade." The cut-points are uncalibrated, so they may not carry a FAIL."""
    dead = np.zeros((4, 12, 12))
    r = kidney_slice_coverage_check(rbf_map=dead, kidney_masks={"left": np.ones((4, 12, 12), bool)},
                                    voxel_mm=(8.0, 2.0, 2.0), cfg=CFG)
    assert r.verdict is Verdict.UNKNOWN
    assert "no ROI statistic" in r.reason


def test_k2_3_never_calls_a_minority_of_negative_voxels_a_majority():
    """The FAIL reason said "a majority-negative map ... is not a perfusion map"
    for ANY negative fraction over the 20% line, so a map that was 25% negative
    was told it was majority-negative - contradicted by the metric printed in the
    same sentence.

    The two FAILs are licensed differently and must read differently: above half
    the SIGN decides and no calibration is involved, so that one is not
    provisional and --no-strict leaves it alone; between 20% and half only the
    uncalibrated threshold decides, so that one stays provisional.
    """
    m = np.ones((10, 10, 10))

    def grade(frac, strict=True):
        vol = np.full((10, 10, 10), 200.0)
        vol.reshape(-1)[:int(frac * vol.size)] = -50.0
        return all_checks()["k2.3.implausible_values"]["fn"](
            rbf_map=vol, kidney_masks={"left": m, "right": m},
            cortex_masks={"left": m, "right": m}, units="mL/100g/min",
            cfg=_replace(CFG, strict=strict))

    minority = grade(0.25)
    assert minority.verdict is Verdict.FAIL
    assert minority.provisional
    assert "majority" not in minority.reason.lower()
    assert "most of the map" not in minority.reason
    assert grade(0.25, strict=False).verdict is Verdict.WARN   # provisional: demotable

    majority = grade(0.62)
    assert majority.verdict is Verdict.FAIL
    assert not majority.provisional
    assert "most of the map is negative" in majority.reason
    # licensed by the sign, so turning strict off must NOT demote it
    assert grade(0.62, strict=False).verdict is Verdict.FAIL
