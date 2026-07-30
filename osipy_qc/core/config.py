"""
All tunable thresholds in one place, each carrying its PROVENANCE.

Why provenance matters
----------------------
Most of this field has no published PASS/FAIL cutoffs. ASLPrep, ExploreASL,
MRIQC and fMRIPrep all report QC metrics and ship *zero* verdict logic. The QEI
cut-off (~0.5, Dolui 2024) is essentially the only threshold in the ASL QC
corpus with a real published derivation.

So every threshold below is tagged with where it actually came from:

    PUBLISHED       a peer-reviewed paper states this number for this purpose.
    IMPLEMENTATION  a reference implementation uses it (code, not a paper).
    UNCALIBRATED    our engineering default. NOT yet calibrated against rater
                    labels. These must not drive a FAIL on their own.

`THRESHOLD_PROVENANCE` below is the machine-readable source of truth, and is
what the report prints next to each number. If you cannot cite it, say so here
rather than in a docstring.

Population profiles
-------------------
CBF norms move enormously with age (a newborn's GM ~16 vs an adult's ~58
mL/100g/min), so the adult band would condemn every neonatal scan. v1.0 ships
two calibrated profiles: `adult` (the brain target) and `neonate` (the mentor's
neonatal domain). Other age groups are planned once their bands are confirmed
with the mentors. `for_population()` returns the tuned config; every number
traces to a citation recorded in THRESHOLD_PROVENANCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class Provenance(str, Enum):
    """Where a threshold's value actually came from."""

    PUBLISHED = "published"
    IMPLEMENTATION = "implementation"
    UNCALIBRATED = "uncalibrated"


@dataclass
class QCConfig:
    # ---- context -----------------------------------------------------------
    population: str = "adult"       # see for_population(): "adult" | "neonate"
    organ: str = "brain"            # brain | kidney (kidney profile is a stub, see ORGANS)
    strict: bool = True             # False -> demote uncalibrated FAILs to WARN (clinical cohorts)

    # ---- QEI (Module 1) ----------------------------------------------------
    qei_warn: float = 0.50          # PUBLISHED - Dolui 2024
    qei_pass: float = 0.55          # UNCALIBRATED - our margin above the published cut-off
    tissue_thresh: float = 0.7      # IMPLEMENTATION - ASLPrep code (paper says 0.9; conflict)
    smooth_fwhm_mm: float = 5.0     # PUBLISHED - Dolui 2024

    # QEI curve constants: the live ASLPrep compute_qei values, not the paper's
    # rounded ones. ASLPrep's own comment says they "differ slightly from those
    # in the paper, but match the actual values used in the original QEI
    # implementation". NOTE qei_c: the paper's Fig.2 prints exp(-0.1*x^0.9);
    # 0.054 rounds to 0.05, NOT 0.1 - a ~1.85x gap, unlike every other constant
    # which rounds cleanly. Open question for the QEI author.
    qei_a: float = 3.0126
    qei_b: float = 2.4419
    qei_c: float = 0.054
    qei_d: float = 0.9272
    qei_e: float = 2.8478
    qei_f: float = 0.5196

    # ---- Noise & distribution (Module 2) -----------------------------------
    # ExploreASL's own tiers: CBF-contrast < 0.67 < vascular-contrast < 1.0 <
    # artifactual-contrast. ExploreASL KEEPS the 0.67-1.0 band (it is excluded
    # only from CBF statistics); >1.0 is the exclusion line. Earlier versions of
    # this file FAILed at 0.67, which inverted ExploreASL's meaning.
    scov_vascular: float = 0.67     # IMPLEMENTATION - above this -> vascular tier (WARN)
    scov_artifact: float = 1.0      # IMPLEMENTATION - above this -> artifactual tier (FAIL)

    # ---- CBF level & contrast (Module 3) -----------------------------------
    gm_cbf_lo: float = 40.0         # PUBLISHED - White Paper p.17 ("40-100 can be normal")
    gm_cbf_hi: float = 100.0        # PUBLISHED - White Paper p.17
    gm_cbf_fail_lo: float = 10.0    # UNCALIBRATED
    gm_cbf_fail_hi: float = 150.0   # UNCALIBRATED
    wm_cbf_lo: float = 15.8         # PUBLISHED - Wu 2013 measured WM range (see provenance)
    wm_cbf_hi: float = 27.5         # PUBLISHED - Wu 2013
    wm_cbf_fail_lo: float = 5.0     # UNCALIBRATED
    wm_cbf_fail_hi: float = 50.0    # UNCALIBRATED
    ratio_pass: float = 1.5         # UNCALIBRATED - only ratio_min has a source
    ratio_min: float = 1.0          # PUBLISHED - ASLPrep excludes GM/WM ratio < 1
    neg_gm_warn: float = 0.10       # UNCALIBRATED
    neg_gm_fail: float = 0.20       # UNCALIBRATED

    # Whole-brain plausibility (3.5) - the ONLY magnitude bounds available when
    # no tissue maps were supplied. Deliberately absurd rather than normal: see
    # the long comment on check 3.5 in checks/cbf_level.py. These do NOT encode
    # "what healthy CBF looks like" - 3.1 does that, from published GM bands.
    # They encode "this cannot be a quantified CBF map at all", which is the only
    # claim a self-derived brain mask can support.
    brain_cbf_absurd_lo: float = 0.0    # UNCALIBRATED - a negative brain mean is non-physical
    brain_cbf_absurd_hi: float = 300.0  # UNCALIBRATED - ~3x the top of the published GM band
    # The percentile brain_mask_fallback thresholds at. Exposed because it moves
    # the reported mean substantially (41 -> 60 on synthetic data across 25 -> 75),
    # which is precisely why 3.5 does not grade that mean against a normal band.
    brain_mask_percentile: float = 50.0  # UNCALIBRATED

    # Deep-GM vs cortical-GM (neonatal). In newborns CBF is HIGHER in deep grey
    # matter than cortical grey matter - the reverse of the adult pattern - so a
    # low ratio here is itself a red flag for a neonate.
    deep_gm_ratio_lo: float = 1.3   # UNCALIBRATED - derived from Miranda 2006 (1.88 term, 2.05 preterm)
    deep_gm_ratio_hi: float = 2.6   # UNCALIBRATED - derived from Miranda 2006

    # ---- Co-registration & coverage (Module 4) -----------------------------
    dice_pass: float = 0.9          # UNCALIBRATED
    dice_warn: float = 0.7          # UNCALIBRATED (0.7 exists only for segmentation, not registration)
    # Fraction of the tissue mask that must actually be covered by non-zero CBF.
    # Guards the "cerebellum outside the ASL FOV" failure: zero voxels inside the
    # ROI silently drag mean GM CBF down and corrupt the GM/WM ratio.
    coverage_warn: float = 0.90     # UNCALIBRATED
    coverage_fail: float = 0.75     # UNCALIBRATED

    # ---- M0 (Module 6) -----------------------------------------------------
    m0_tr_min_s: float = 5.0        # PUBLISHED - White Paper p.15
    t1_tissue_s: float = 1.4        # UNCALIBRATED - matches no published value; see provenance

    # ---- Motion (Module 7) -------------------------------------------------
    # Both numbers are published, but they grade DIFFERENT statistics and were
    # previously wired to the wrong ones:
    #   1.0 mm is ASLPrep's subject-exclusion rule on MEAN framewise displacement.
    #   0.5 mm is Power 2012's PER-FRAME censoring threshold (not a mean).
    fd_mean_fail_mm: float = 1.0    # PUBLISHED - Adebimpe 2022 (mean FD > 1 mm -> exclude)
    fd_frame_censor_mm: float = 0.5  # PUBLISHED - Power 2012 (per-frame censoring)
    fd_censor_frac_warn: float = 0.20  # UNCALIBRATED - share of frames over the censor line
    head_radius_mm: float = 50.0    # PUBLISHED - Power 2012 (rotation -> mm sphere)

    # ---- Quantification parameters (user-supplied) -------------------------
    # These are NOT QC thresholds - they are acquisition facts the QC layer must
    # be told, because they are absent from NIfTI headers and they change the CBF
    # a pipeline produces. None = unknown; checks that need them report UNKNOWN
    # rather than silently assuming a default.
    labeling_efficiency: float | None = None   # alpha (White Paper: 0.85 PCASL, 0.98 PASL)
    label_duration_s: float | None = None      # tau
    post_labeling_delay_s: float | None = None  # PLD
    t1_blood_s: float | None = None            # 1.65 s @3T adult; differs in neonates (hematocrit)


# ---------------------------------------------------------------------------
# Provenance: field name -> (level, citation, what the source actually says)
# ---------------------------------------------------------------------------
THRESHOLD_PROVENANCE: dict[str, tuple[Provenance, str, str]] = {
    "qei_warn": (
        Provenance.PUBLISHED,
        "Dolui S, et al. JMRI 2024;60(6):2497-2508. doi:10.1002/jmri.29308",
        'Discussion: "a cut-off value of 0.5 has worked reliably for a wide variety ASL '
        'protocols in multiple studies." The ROC-derived value in the paper is 0.53.',
    ),
    "qei_pass": (
        Provenance.UNCALIBRATED,
        "NONE",
        "0.55 appears nowhere in the QEI paper. It is our provisional margin above 0.50. "
        "Open question: use the paper's ROC value 0.53, or 0.50?",
    ),
    "tissue_thresh": (
        Provenance.IMPLEMENTATION,
        "Adebimpe A, et al. Nat Methods 2022;19(6):683-686. doi:10.1038/s41592-022-01458-7 "
        "(+ aslprep/utils/confounds.py compute_qei default)",
        "CONFLICT: ASLPrep binarises tissue maps at 0.7; Dolui 2024 p.4 says 0.9. "
        "0.7 appears nowhere in the QEI paper. Only the author can settle this.",
    ),
    "smooth_fwhm_mm": (
        Provenance.PUBLISHED,
        "Dolui S, et al. JMRI 2024;60(6):2497-2508. doi:10.1002/jmri.29308",
        '"our recommendation is to smooth the CBF maps by a 5 mm isotropic Gaussian kernel '
        '... as that was used to derive the QEI parameters and the cut-off value."',
    ),
    "qei_a": (Provenance.IMPLEMENTATION, "aslprep/utils/confounds.py compute_qei",
              "Paper Fig.2 prints 1-exp(-3.0*x^2.4); 3.0126 rounds to 3.0. Consistent."),
    "qei_b": (Provenance.IMPLEMENTATION, "aslprep/utils/confounds.py compute_qei",
              "2.4419 rounds to the paper's 2.4. Consistent."),
    "qei_c": (
        Provenance.IMPLEMENTATION,
        "aslprep/utils/confounds.py compute_qei",
        "DISCREPANCY: the paper's Fig.2 prints exp(-0.1*x^0.9). 0.054 rounds to 0.05, not "
        "0.1 - a ~1.85x gap. Every other constant rounds cleanly. Needs the author's ruling.",
    ),
    "qei_d": (Provenance.IMPLEMENTATION, "aslprep/utils/confounds.py compute_qei",
              "0.9272 rounds to the paper's 0.9. Consistent."),
    "qei_e": (Provenance.IMPLEMENTATION, "aslprep/utils/confounds.py compute_qei",
              "2.8478 rounds to the paper's 2.8. Consistent."),
    "qei_f": (Provenance.IMPLEMENTATION, "aslprep/utils/confounds.py compute_qei",
              "0.5196 rounds to the paper's 0.5. Consistent."),

    "scov_vascular": (
        Provenance.IMPLEMENTATION,
        "ExploreASL xASL_qc_SortBySpatialCoV.m (bins at 0.67 and 1.0); "
        "Mutsaerts H, et al. NeuroImage 2020;219:117031",
        "CAUTION - the citation chain usually given for 0.67 is BROKEN. ExploreASL 2020 "
        "attributes it to Mutsaerts 2018, which contains neither 'spatial CoV' nor 0.67. "
        "Mutsaerts 2017 (the sCoV paper) states only 'mean GM spatial CoV was 56.9 +/- 13.2% "
        "(range 39.3-113.6%)' and gives NO cutoff. 0.67 is literally 2/3, an ExploreASL "
        "implementation default. Cite it as an ExploreASL convention, not as Mutsaerts-derived.",
    ),
    "scov_artifact": (
        Provenance.IMPLEMENTATION,
        "ExploreASL xASL_qc_SortBySpatialCoV.m",
        "ExploreASL's own tiers: CBF-contrast <0.67< vascular <1.0< artifactual. It KEEPS the "
        "0.67-1.0 band; >1.0 is the exclusion line.",
    ),

    "gm_cbf_lo": (
        Provenance.PUBLISHED,
        "Alsop DC, et al. MRM 2015;73(1):102-116, p.17. doi:10.1002/mrm.25197",
        'Verbatim: "As a general rule, gray matter CBF values from 40-100 ml/min/100ml can be normal."',
    ),
    "gm_cbf_hi": (
        Provenance.PUBLISHED,
        "Alsop DC, et al. MRM 2015;73(1):102-116, p.17. doi:10.1002/mrm.25197",
        'Verbatim: "gray matter CBF values from 40-100 ml/min/100ml can be normal."',
    ),
    "gm_cbf_fail_lo": (
        Provenance.UNCALIBRATED,
        "NONE",
        "Invented FAIL bound - a literature sweep found ZERO hits for a '<10 mL/100g/min' GM cutoff. "
        "The 40-100 PASS band is sourced; this is not.",
    ),
    "gm_cbf_fail_hi": (
        Provenance.UNCALIBRATED,
        "NONE (150 is a repurposed reporting number)",
        "150 exists only in ASLPrep code as CBF_THRESH_DEFAULTS = (100, 150, 200) "
        "(aslprep/utils/confounds.py:12-13), which is a CONFIGURABLE REPORTING parameter for "
        "'% of voxels above X' - its tests exercise gt_50/gt_99/gt_20/gt_60. No verdict logic is "
        "attached to 150 anywhere.",
    ),
    "wm_cbf_lo": (
        Provenance.PUBLISHED,
        "Wu W-C, Lin S-C, Wang DJ, Chen K-L, Li Y-D. PLoS One 2013;8(12):e82679. "
        "doi:10.1371/journal.pone.0082679 (PMID 24324822). Corroborated by Clement P, Petr J, "
        "Dijsselhof MBJ, Padrela B, Pasternak M, Dolui S, et al. Front Radiol 2022;2:929533. "
        "doi:10.3389/fradi.2022.929533",
        'Wu 2013 verbatim: "The measured white matter perfusion and perfusion ratio of gray matter '
        'to white matter were 15.8-27.5 ml/100ml/min and 1.8-4.0, respectively, depending on spatial '
        'resolution as well as the amount of deep white matter included." Clement 2022 (Dolui is a '
        'co-author) verbatim: "about 20 mL/100g/min in white matter (WM) in young and healthy adults". '
        "NOTE the ASL White Paper itself states NO WM CBF value - its WM section is entirely "
        "qualitative. NOTE ALSO the unit clash: Wu reports ml/100ml/min, Clement mL/100g/min; "
        "converting needs a density assumption (~1.05 g/mL). Do not silently equate them.",
    ),
    "wm_cbf_hi": (
        Provenance.PUBLISHED,
        "Wu W-C, et al. PLoS One 2013;8(12):e82679. doi:10.1371/journal.pone.0082679",
        'Upper end of Wu 2013\'s measured 15.8-27.5 ml/100ml/min. An earlier "20-25" band was '
        "dropped: it sits at or above the measured central tendency (16.1, 19.5, 20, 21.2 across "
        "Aslan 2011 / Zhang 2014 / Clement 2022 / Juttukonda 2021) and would flag normal scans.",
    ),
    "wm_cbf_fail_lo": (Provenance.UNCALIBRATED, "NONE", "Provisional engineering bound."),
    "wm_cbf_fail_hi": (Provenance.UNCALIBRATED, "NONE", "Provisional engineering bound."),
    "ratio_min": (
        Provenance.PUBLISHED,
        "Adebimpe A, et al. Nat Methods 2022;19(6):683-686. doi:10.1038/s41592-022-01458-7",
        'Verbatim, twice: "this ratio is expected to be greater than 1" and "we excluded '
        'participants with ... a CBF GM to WM ratio of less than 1."',
    ),
    "ratio_pass": (
        Provenance.UNCALIBRATED,
        "NONE for 1.5. Context: Clement P, ... Dolui S, et al. Front Radiol 2022;2:929533; "
        "Wu W-C, et al. PLoS One 2013;8(12):e82679; Zhang K, et al. JCBFM 2014;34(8):1373-1380",
        "No source states 1.5. What IS published is the normal RANGE: Clement 2022 (Dolui co-author) "
        '"CBF in GM is around 2 to 4 times higher than the WM CBF"; Wu 2013 "1.8-4.0"; Zhang 2014 '
        '"GM/WM ratio of CBF was 3.0 for PET and 3.4 for pCASL". '
        "THE LOGIC TRAP: those are POPULATION RANGES, which cannot justify a DECISION BOUNDARY. "
        "THREE reasons a fixed 1.5 is unsafe: (1) the ratio is SMOOTHING-DEPENDENT - Wu 2013 measured "
        "2.3 unsmoothed, 2.2 at 3 mm, 1.8 at 8 mm, and we smooth at 5 mm FWHM for the QEI; "
        "(2) it declines ~0.79%/year with age (Parkes LM, et al. MRM 2004;51(4):736-743 - relative "
        "rate from the abstract; that paper's absolute CBF values are paywalled and unverified); "
        "(3) smooth GE 3D spiral scans legitimately land at 1.2-1.3.",
    ),
    "neg_gm_warn": (Provenance.UNCALIBRATED, "NONE",
                    "Dolui 2024 uses negative-GM fraction as a CONTINUOUS QEI term, never a cutoff."),
    "neg_gm_fail": (Provenance.UNCALIBRATED, "NONE", "No published cutoff exists."),
    "deep_gm_ratio_lo": (
        Provenance.UNCALIBRATED,
        "Derived from Miranda MJ, Olofsson K, Sidaros K. Pediatr Res 2006;60(3):359-363. "
        "doi:10.1203/01.PDR.0000232785.00965.B3 (PMID 16857776)",
        "Miranda measured basal ganglia+thalami 39 vs cortical GM 19 (preterm at term-equivalent) "
        "and 30 vs 16 (term), both p<0.0001 -> deep/cortical ratios 2.05 and 1.88. The BAND is our "
        "extrapolation from those two points, not a published cutoff.",
    ),
    "deep_gm_ratio_hi": (Provenance.UNCALIBRATED, "Derived from Miranda 2006 (as above)",
                         "Our extrapolation, not a published cutoff."),

    "dice_pass": (Provenance.UNCALIBRATED, "NONE", "No published Dice cutoff for registration QC."),
    "dice_warn": (
        Provenance.UNCALIBRATED,
        "Closest: Zou KH, et al. Acad Radiol 2004;11(2):178-189. doi:10.1016/S1076-6332(03)00671-8",
        "Zou reports 'good overlap ... DSC>0.700' but for SEGMENTATION validation, not "
        "registration QC, and not in ASL. Birn 2023 (Front Neuroimaging 2:1072927) further shows "
        "Dice is non-monotonic with registration quality. Weak basis - treat as a hint.",
    ),
    "coverage_warn": (Provenance.UNCALIBRATED, "NONE",
                      "Our defence against the 'cerebellum outside the ASL FOV' bug."),
    "coverage_fail": (Provenance.UNCALIBRATED, "NONE", "Our engineering default."),

    "m0_tr_min_s": (
        Provenance.PUBLISHED,
        "Alsop DC, et al. MRM 2015;73(1):102-116, p.15. doi:10.1002/mrm.25197",
        'Verbatim: "If TR is less than 5s, the PD image should be multiplied by the factor '
        '(1/(1 - e^-TR/T1,tissue)) ... No labeling or BS should be applied for this scan."',
    ),
    "t1_tissue_s": (
        Provenance.UNCALIBRATED,
        "NONE for 1.4 s specifically",
        "The White Paper says only 'the assumed T1 of gray matter' and its Table 3 gives no GM T1 "
        "(only T1,blood = 1650 ms @3T). ExploreASL uses 1240 ms GM / 800 ms WM; Wansapura 1999 "
        "reports GM T1 ~1331 ms. 1400 ms matches none of these.",
    ),

    "fd_mean_fail_mm": (
        Provenance.PUBLISHED,
        "Adebimpe A, et al. Nat Methods 2022;19(6):683-686. doi:10.1038/s41592-022-01458-7",
        'Verbatim: "we excluded participants with mean frame-wise displacement greater than 1 mm." '
        "Applies to MEAN FD (this is how we now wire it).",
    ),
    "fd_frame_censor_mm": (
        Provenance.PUBLISHED,
        "Power JD, et al. NeuroImage 2012;59(3):2142-2154. doi:10.1016/j.neuroimage.2011.10.018",
        'Verbatim: "values of 0.5 for framewise displacement ... were chosen to represent values '
        'well above the norm found in still subjects." NOTE Power chose it by eyeballing plots, and '
        "it is a PER-FRAME censoring threshold, not a subject-level mean cutoff.",
    ),
    "fd_censor_frac_warn": (Provenance.UNCALIBRATED, "NONE",
                            "Our default for 'how many censored frames is too many'."),
    "head_radius_mm": (
        Provenance.PUBLISHED,
        "Power JD, et al. NeuroImage 2012;59(3):2142-2154. doi:10.1016/j.neuroimage.2011.10.018",
        'Verbatim: "converted from degrees to millimeters by calculating displacement on the '
        'surface of a sphere of radius 50 mm." This is a DEFINITION, not a threshold.',
    ),
}


def provenance_of(field: str) -> tuple[Provenance, str, str]:
    """Provenance record for a config field. Unregistered fields are UNCALIBRATED."""
    return THRESHOLD_PROVENANCE.get(
        field, (Provenance.UNCALIBRATED, "NONE", "No provenance recorded.")
    )


def uncalibrated_fields() -> list[str]:
    """Every threshold we cannot cite. These must never drive a FAIL alone."""
    return sorted(k for k, (lvl, _, _) in THRESHOLD_PROVENANCE.items()
                  if lvl is Provenance.UNCALIBRATED)


# ---------------------------------------------------------------------------
# Population profiles
# ---------------------------------------------------------------------------
# CBF norms per supported population. v1.0 scope is deliberately tight: `adult`
# is the brain target, and `neonate` is the mentor's neonatal domain. Other age
# groups (infant / child / adolescent / elderly) are PLANNED but not shipped -
# their bands were engineering extrapolations, not yet calibrated with the
# mentors, so exposing them would over-state validated support. They live only in
# THRESHOLD_PROVENANCE.md's "planned" section until confirmed.
#
# Sources (verified against the papers):
#   adult    Alsop DC, et al. MRM 2015;73(1):102-116 (ASL White Paper), p.17 -
#            "gray matter CBF from 40-100 ml/min/100ml can be normal"; WM
#            15.8-27.5 (Wu W-C, et al. PLoS One 2013;8(12):e82679).
#   neonate  Miranda MJ, Olofsson K, Sidaros K. Pediatr Res 2006;60(3):359-363.
#            doi:10.1203/01.PDR.0000232785.00965.B3 - term: cortical GM 16, deep
#            GM 30, WM 10 mL/100g/min; preterm@TEA: 19 / 39 / 15. One 'neonate'
#            band spans both term and preterm.
#
# The BANDS are our engineering widening of those measured means; the MEANS are
# published, the CUTOFFS are not (see THRESHOLD_PROVENANCE.md).
POPULATIONS: dict[str, dict] = {
    "adult": dict(),  # the dataclass defaults (White Paper 40-100)
    "neonate": dict(
        gm_cbf_lo=8.0, gm_cbf_hi=32.0, gm_cbf_fail_lo=2.0, gm_cbf_fail_hi=60.0,
        wm_cbf_lo=5.0, wm_cbf_hi=24.0, wm_cbf_fail_lo=1.0, wm_cbf_fail_hi=40.0,
        ratio_pass=1.1, ratio_min=0.8,
    ),
}

# Organ profiles. Kidney is a deliberate STUB: the checks that transfer are
# marked, and QEI is explicitly excluded because its spatial template
# (spCBF = 2.5*GM + 1*WM) is a brain tissue model with no kidney equivalent.
ORGANS: dict[str, dict] = {
    "brain": {
        "skip_checks": (),
        "note": "Default. All 17 checks apply.",
    },
    "kidney": {
        # QEI needs a GM/WM template -> structurally inapplicable.
        # Cortex/medulla contrast replaces the GM/WM ratio; not yet implemented.
        "skip_checks": ("1.qei", "3.2.gm_wm_ratio", "3.4.deep_gm_ratio"),
        "note": (
            "STUB. Renal ASL consensus: Nery F, et al. MAGMA 2020;33(1):141-161. "
            "doi:10.1007/s10334-019-00800-z. Blocker: no consensus-endorsed automated "
            "cortex/medulla segmentation exists, so the cortex/medulla ratio cannot yet "
            "be computed unsupervised."
        ),
    },
}


def for_population(name: str, **overrides) -> QCConfig:
    """Config tuned to a population, e.g. for_population('neonate').

    Raises on an unknown name rather than silently returning adult bands - a
    silent adult default on a neonate would fail every scan.
    """
    if name not in POPULATIONS:
        raise ValueError(
            f"unknown population {name!r}; known: {sorted(POPULATIONS)}"
        )
    return replace(QCConfig(population=name), **{**POPULATIONS[name], **overrides})


def for_organ(name: str, **overrides) -> QCConfig:
    """Config for an organ. Checks listed in ORGANS[name]['skip_checks'] are the
    ones the caller should not run (run_qc's `checks=` argument)."""
    if name not in ORGANS:
        raise ValueError(f"unknown organ {name!r}; known: {sorted(ORGANS)}")
    return replace(QCConfig(organ=name), **overrides)


def skipped_for_organ(name: str) -> tuple[str, ...]:
    """Check names that do not apply to this organ."""
    return tuple(ORGANS.get(name, {}).get("skip_checks", ()))
