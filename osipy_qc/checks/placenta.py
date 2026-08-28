"""
Placenta QC — the 15 checks of PLACENTA_QC_DESIGN.md, Streams A and B.

Read this before changing a threshold
-------------------------------------
The placenta has less published ground beneath it than any organ this package
touches. Taso 2023 (ISMRM Perfusion Study Group) is the nearest thing to a
reference document and the placenta appears there with neither Recommendations
nor summarised practice — where the kidney at least has 59 consensus statements
(with no quality thresholds), the placenta has no consensus document at all.

What follows from that:

1. **Units are a gate, not a formality.** P2.1 runs first and everything numeric
   depends on it. Placental perfusion is published in mL/100g/min, in %-of-M0,
   and in arbitrary units, and the same physiology produces wildly different
   numbers in each. A magnitude bound applied to an undeclared map is a bound on
   an unknown quantity.

2. **Heterogeneity is physiology, not a defect.** The placenta is genuinely
   non-uniform — cotyledons and septa are real structures — so P2.3 reports
   within-placenta variation as INFO and never grades it. This is the opposite
   of the brain's spatial-CoV check, and copying that check across would
   condemn healthy placentas.

3. **Masks come from outside and are graded, not trusted.** Human inter-rater
   Dice on placental segmentation is about 0.68, against >0.90 for brain GM/WM.
   The mask is therefore a declared input with recorded provenance, and P3.1
   grades the mask itself.

4. **Gestational age is mandatory context.** Placental perfusion changes across
   gestation, so a verdict without GA is not interpretable — which is why its
   absence is a definitional FAIL in P4.2 rather than a missing-metadata WARN.

Ethics note: placental images are frequently unshareable. Everything here runs
locally and the report is written to be self-explanatory, so a site that cannot
send us data can still run the tool and send us only the report.
"""

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict
from ..utils.mathops import pearson
from ..utils.roi import (as_mask, component_sizes, largest_component_fraction,
                         local_ssim, roi_stats, roi_values)

ORGAN = "placenta"

# Local-SSIM box half-width in voxels. The design specifies a 20 mm kernel; at
# the 3-4 mm in-plane resolution placental ASL is acquired at, that is a box of
# roughly 7 voxels across.
_SSIM_RADIUS = 3

# Unit families the checks fold synonyms into. Two spellings of the same
# quantity must not be treated as two different quantities.
_UNIT_FAMILIES: dict[str, tuple[str, ...]] = {
    "per_mass": ("ml/100g/min", "ml/min/100g", "ml/100 g/min", "ml/min/100 g",
                 "ml100gmin", "mlper100gpermin"),
    "percent_m0": ("%m0", "percent_m0", "%_of_m0", "fraction_m0", "pct_m0"),
    "arbitrary": ("a.u.", "au", "arbitrary", "arb"),
}


def _unit_family(units) -> str | None:
    """Fold a declared unit string into its family, or None if undeclared."""
    if not isinstance(units, str) or not units.strip():
        return None
    u = units.strip().lower().replace(" ", "")
    for family, spellings in _UNIT_FAMILIES.items():
        if any(u == s.replace(" ", "") for s in spellings):
            return family
    return "other"


def _grid_error(volume, mask) -> str | None:
    """Actionable message when the mask is not on the image grid."""
    if mask is None:
        return None
    a, m = np.asarray(volume).shape[:3], np.asarray(mask).shape[:3]
    if a != m:
        return (f"the placenta mask is {tuple(m)} but the image is {tuple(a)} - resample the "
                "mask into the image grid before grading (this tool deliberately does not "
                "resample: interpolating a mask changes which voxels are called placenta)")
    return None


def _holes_fraction(mask) -> float:
    """Share of the mask's bounding-box interior that is enclosed background.

    A hole is background that the mask surrounds. Measured by flood-filling the
    background inward from the volume faces: whatever background the fill cannot
    reach is enclosed. Uses the same 6-connected propagation as the component
    labeller, so no scipy and no separate connectivity convention.
    """
    m = as_mask(mask)
    if m.ndim != 3 or not m.any():
        return float("nan")
    outside = np.zeros(m.shape, dtype=bool)
    # seed from every face, then propagate through background only
    outside[0], outside[-1] = True, True
    outside[:, 0], outside[:, -1] = True, True
    outside[:, :, 0], outside[:, :, -1] = True, True
    outside &= ~m
    while True:
        grown = outside.copy()
        for axis in range(3):
            for shift in (1, -1):
                rolled = np.roll(outside, shift, axis=axis)
                idx: list = [slice(None)] * 3
                idx[axis] = 0 if shift == 1 else -1
                rolled = rolled.copy()
                rolled[tuple(idx)] = False
                grown |= rolled
        grown &= ~m
        if np.array_equal(grown, outside):
            break
        outside = grown
    enclosed = int(np.count_nonzero(~m & ~outside))
    return float(enclosed / (int(m.sum()) + enclosed)) if (m.sum() + enclosed) else 0.0


# --------------------------------------------------------------------------- #
# Module P1 — the quality-index slot, deliberately empty
# --------------------------------------------------------------------------- #
@register_qc_check("p1.1.placental_qei", stream="B", required=False, organ=ORGAN)
def placental_qei_check(**_) -> CheckResult:
    """ALWAYS N/A in v1. No placental quality index exists.

    The brain QEI is built on a grey/white spatial prior (spCBF = 2.5*GM + 1*WM)
    and its constants were fitted against expert ratings. The placenta has no
    tissue-class prior to build such a template from, and no rated dataset to fit
    against. Registered so the absence is visible in every report rather than
    being an invisible hole in the check list.
    """
    return CheckResult(
        "p1.1.placental_qei", Verdict.NA,
        metric={"placental_qei": None,
                "blockers": ["no_tissue_class_prior",
                             "mask_boundary_poorly_reproducible",
                             "no_rated_dataset"],
                "components_shipped_separately": ["P2.2.negative_fraction",
                                                  "P2.3.segment_cov",
                                                  "P6.1.rejected_fraction",
                                                  "P6.2.temporal_sd"]},
        reason="no placental quality index exists; the computable components ship as "
               "P2.2, P2.3, P6.1 and P6.2")


# --------------------------------------------------------------------------- #
# Module P2 — units, validity, distribution
# --------------------------------------------------------------------------- #
@register_qc_check("p2.1.units_declaration", stream="B", required=True, organ=ORGAN)
def placenta_units_check(perfusion_map=None, declared_units=None, quantified=None,
                         constants=None, physiological_bound_requested=None,
                         **_) -> CheckResult:
    """THE GATE. Runs before anything numeric and decides what the map's numbers
    even mean.

    A map in arbitrary units or %-of-M0 is perfectly legal and useful; what is
    not legal is grading it against a physiological bound. So an undeclared map
    is only a FAIL when a physiological claim is actually being made of it
    (`quantified=True`); otherwise it is a WARN that constrains what downstream
    checks are allowed to do.
    """
    if perfusion_map is None:
        return CheckResult("p2.1.units_declaration", Verdict.UNKNOWN, reason="no map supplied")
    family = _unit_family(declared_units)
    needed = ("lambda", "alpha", "t1_blood_ms")
    have = {k: (constants or {}).get(k) for k in needed} if isinstance(constants, dict) else {}
    missing = [k for k in needed if have.get(k) in (None, "")]
    metric = {"declared_units": declared_units, "units_family": family,
              "quantified": bool(quantified), "constants": have,
              "missing_constants": missing}

    # The spec's FAIL is "declared_units is None AND the caller requested a
    # physiological-units metric or supplied a physiological bound" - it is about
    # a request that cannot be honoured, not about the map itself. Presenting a
    # map as `quantified` IS such a request (it asserts physiological units), and
    # an explicit bound is the other form of it.
    physiological_request = bool(quantified) or bool(physiological_bound_requested)
    metric["physiological_request"] = physiological_request
    if family is None:
        if physiological_request:
            return CheckResult("p2.1.units_declaration", Verdict.FAIL, metric=metric,
                               reason="a physiological-units result was requested but the map's "
                                      "units are not declared - the request cannot be honoured, "
                                      "since no bound applies to an unknown quantity")
        return CheckResult("p2.1.units_declaration", Verdict.WARN, metric=metric,
                           reason="units not declared - magnitude checks will report values "
                                  "without grading them")
    if quantified and missing:
        return CheckResult("p2.1.units_declaration", Verdict.WARN, metric=metric,
                           reason=f"units declared ({declared_units}) but the quantification "
                                  f"constants {missing} are missing - the map cannot be "
                                  "reproduced or compared")
    return CheckResult("p2.1.units_declaration", Verdict.PASS, metric=metric,
                       reason=f"units declared as {declared_units} ({family})"
                              + ("" if not quantified else " with full quantification constants"))


@register_qc_check("p2.2.implausible_values", stream="B", required=True, organ=ORGAN)
def placenta_implausible_check(perfusion_map=None, placenta_mask=None, declared_units=None,
                               cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Negative, non-finite and upper-outlier fractions inside the placenta.

    The upper fence is data-driven (P75 + 3*IQR) rather than a fixed ceiling,
    because no published placental perfusion ceiling exists and the plausible
    magnitude depends entirely on the declared units. A Tukey fence widened from
    the usual 1.5 to 3.0 asks "is this voxel extreme even for THIS placenta",
    which is answerable without a literature bound.
    """
    if perfusion_map is None:
        return CheckResult("p2.2.implausible_values", Verdict.UNKNOWN, reason="no map supplied")
    if placenta_mask is None:
        return CheckResult("p2.2.implausible_values", Verdict.UNKNOWN, reason="needs a placenta mask")
    bad = _grid_error(perfusion_map, placenta_mask)
    if bad:
        return CheckResult("p2.2.implausible_values", Verdict.UNKNOWN, reason=bad)

    perf = np.asarray(perfusion_map, dtype=float)
    m = as_mask(placenta_mask)
    all_vals = perf[m]
    n_all = int(all_vals.size)
    if n_all < cfg.placenta_min_roi_voxels:
        return CheckResult("p2.2.implausible_values", Verdict.UNKNOWN,
                           metric={"n_voxels": n_all},
                           reason=f"only {n_all} voxels in the mask; at least "
                                  f"{cfg.placenta_min_roi_voxels} are needed")
    finite = all_vals[np.isfinite(all_vals)]
    nonfinite_frac = float(1.0 - finite.size / n_all)
    # Count the non-finite voxels, then EXCLUDE them: the negative and
    # upper-outlier fractions are shares OF THE MEASURED VOXELS. Leaving them in
    # the denominator makes both fractions shrink as the map gets emptier, so a
    # half-absent map with every measured voxel negative would report 50%
    # negative instead of 100% - the fraction would fall exactly when the data
    # got worse. `nonfinite_fraction` carries the absence separately.
    neg_frac = (float(np.count_nonzero(finite < 0) / finite.size)
                if finite.size else float("nan"))

    upper_frac = float("nan")
    fence = float("nan")
    if finite.size:
        p25, p75 = np.percentile(finite, [25, 75])
        fence = float(p75 + cfg.placenta_iqr_multiplier * (p75 - p25))
        upper_frac = float(np.count_nonzero(finite > fence) / finite.size)

    metric = {"n_voxels": n_all, "n_finite_voxels": int(finite.size),
              "fraction_denominator": "finite in-mask voxels",
              "negative_fraction": neg_frac,
              "nonfinite_fraction": nonfinite_frac, "upper_outlier_fraction": upper_frac,
              "upper_fence": fence, "fence_rule": f"P75 + {cfg.placenta_iqr_multiplier}*IQR",
              "units_family": _unit_family(declared_units),
              "thresholds": {"negative_warn": cfg.placenta_neg_frac_warn,
                             "upper_warn": cfg.placenta_upper_frac_warn,
                             "nonfinite_fail": cfg.placenta_nonfinite_fail}}
    shown = (f"{neg_frac:.1%} negative, {nonfinite_frac:.1%} non-finite, "
             f"{upper_frac:.1%} above the fence" if np.isfinite(upper_frac)
             else f"{neg_frac:.1%} negative, {nonfinite_frac:.1%} non-finite")

    if nonfinite_frac > cfg.placenta_nonfinite_fail:
        return CheckResult("p2.2.implausible_values", Verdict.FAIL, metric=metric,
                           reason=f"{shown} - the map is mostly absent inside its own mask")
    if (np.isfinite(neg_frac) and neg_frac > cfg.placenta_neg_frac_warn) or (
            np.isfinite(upper_frac) and upper_frac > cfg.placenta_upper_frac_warn):
        return CheckResult("p2.2.implausible_values", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"{shown} - above the implausible-value lines")
    return CheckResult("p2.2.implausible_values", Verdict.PASS, metric=metric,
                       reason=f"{shown} - within the implausible-value lines")


@register_qc_check("p2.3.segment_cov", stream="B", required=True, organ=ORGAN)
def placenta_segment_cov_check(perfusion_map=None, placenta_mask=None, context=None,
                               cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Within-placenta heterogeneity. Report-only: emits INFO, never a verdict.

    This is the deliberate opposite of the brain's spatial-CoV check. Cotyledons
    and septa make a healthy placenta genuinely non-uniform, and published
    healthy segment CoV is 0.58 +/- 0.10 - a level that would be alarming in
    grey matter. Grading it would condemn normal physiology, so the number is
    reported with its reference beside it and no threshold is applied.
    """
    if perfusion_map is None or placenta_mask is None:
        return CheckResult("p2.3.segment_cov", Verdict.UNKNOWN, reason="needs a map and a mask")
    bad = _grid_error(perfusion_map, placenta_mask)
    if bad:
        return CheckResult("p2.3.segment_cov", Verdict.UNKNOWN, reason=bad)

    perf = np.asarray(perfusion_map, dtype=float)
    m = as_mask(placenta_mask)
    k = cfg.placenta_segment_size
    means = []
    nz = perf.shape[2] if perf.ndim == 3 else 1
    for z in range(nz):
        sl, ms = (perf[:, :, z], m[:, :, z]) if perf.ndim == 3 else (perf, m)
        for i in range(0, sl.shape[0] - k + 1, k):
            for j in range(0, sl.shape[1] - k + 1, k):
                seg_m = ms[i:i + k, j:j + k]
                if not seg_m.all():          # only COMPLETE segments count
                    continue
                vals = sl[i:i + k, j:j + k][np.isfinite(sl[i:i + k, j:j + k])]
                if vals.size == k * k:
                    means.append(float(vals.mean()))

    if len(means) < cfg.placenta_min_segments:
        return CheckResult("p2.3.segment_cov", Verdict.UNKNOWN,
                           metric={"n_segments": len(means)},
                           reason=f"only {len(means)} complete {k}x{k} segments; at least "
                                  f"{cfg.placenta_min_segments} are needed")
    arr = np.asarray(means)
    mu = float(arr.mean())
    if not (mu > 0):
        # A CoV about a non-positive mean is not a dispersion measure. Saying so
        # is more useful than printing "nan", and it points at the real finding.
        return CheckResult("p2.3.segment_cov", Verdict.UNKNOWN,
                           metric={"n_segments": len(means), "mean_of_segment_means": mu},
                           reason=f"the mean of the segment means is {mu:.1f} - a coefficient of "
                                  "variation about a non-positive mean is not a dispersion "
                                  "measure (see P2.2 for the negative fraction)")
    scov = float(arr.std(ddof=1) / mu)
    metric = {"segment_cov": scov, "n_segments": len(means), "segment_size": [k, k],
              "mean_of_segment_means": mu,
              "healthy_reference": {"mean": 0.58, "sd": 0.10},
              "context": context if isinstance(context, dict) else {},
              "graded": False,
              "provenance": "reported only - placental heterogeneity is physiological "
                            "(cotyledons and septa), so no threshold is applied"}
    return CheckResult("p2.3.segment_cov", Verdict.INFO, metric=metric,
                       reason=f"segment CoV {scov:.2f} over {len(means)} segments "
                              "(healthy reference 0.58 +/- 0.10) - reported, not graded")


# --------------------------------------------------------------------------- #
# Module P3 — mask integrity and slab coverage
# --------------------------------------------------------------------------- #
@register_qc_check("p3.1.mask_integrity", stream="B", required=True, organ=ORGAN)
def placenta_mask_integrity_check(placenta_mask=None, perfusion_map=None, mask_source=None,
                                  roi_definition=None, cfg: QCConfig = QCConfig(),
                                  **_) -> CheckResult:
    """Is the placenta mask one sane object on the right grid, and where did it
    come from?

    Provenance is recorded rather than judged. Human inter-rater Dice on
    placental segmentation is about 0.68 - two competent experts disagree on a
    third of the boundary - so the tool cannot verify a mask it is handed. What
    it can do is state who drew it and how, so a reader knows what the numbers
    rest on.
    """
    if placenta_mask is None:
        return CheckResult("p3.1.mask_integrity", Verdict.UNKNOWN, reason="no mask supplied")
    m = as_mask(placenta_mask)
    n = int(m.sum())
    metric = {"n_voxels": n, "mask_source": mask_source or "not stated",
              "roi_definition": roi_definition or "not stated"}
    if perfusion_map is not None:
        bad = _grid_error(perfusion_map, placenta_mask)
        metric["grid_matches"] = bad is None
        if bad:
            # physically impossible to grade: the mask does not describe this image
            return CheckResult("p3.1.mask_integrity", Verdict.FAIL, metric=metric, reason=bad)
    if n == 0:
        return CheckResult("p3.1.mask_integrity", Verdict.FAIL, metric=metric,
                           reason="the placenta mask is empty")

    sizes = component_sizes(m)
    frac = largest_component_fraction(m)
    holes = _holes_fraction(m)
    metric.update({"n_components": len(sizes), "largest_component_fraction": frac,
                   "holes_fraction": holes,
                   "component_floor": cfg.placenta_mask_component_frac,
                   "holes_warn": cfg.placenta_holes_frac_warn})
    problems = []
    if np.isfinite(frac) and frac < cfg.placenta_mask_component_frac:
        problems.append(f"{len(sizes)} disconnected pieces ({frac:.0%} in the largest)")
    if np.isfinite(holes) and holes > cfg.placenta_holes_frac_warn:
        problems.append(f"{holes:.1%} enclosed holes")
    if mask_source in (None, ""):
        problems.append("mask provenance not stated")
    elif ("perfusion" in str(mask_source).lower()
          and "whole" in str(roi_definition or "").lower()):
        # Drawing a whole-placenta ROI on the perfusion map itself inverts the
        # literature's convention (the mask is drawn on the M0 / a structural
        # image) and risks circularity: the mask is then defined by the very
        # signal it is used to measure, so a dropout region is excluded from the
        # ROI rather than reported by it.
        problems.append("the whole-placenta mask was drawn on the perfusion map itself, which "
                        "inverts the literature's convention and risks circularity - the mask is "
                        "defined by the signal it is then used to measure")
    if problems:
        return CheckResult("p3.1.mask_integrity", Verdict.WARN, metric=metric,
                           reason=f"{n} voxels: " + "; ".join(problems))
    return CheckResult("p3.1.mask_integrity", Verdict.PASS, metric=metric,
                       reason=f"{n} voxels, one connected object, {holes:.1%} holes, "
                              f"drawn by {mask_source}")


@register_qc_check("p3.2.slab_coverage", stream="B", required=True, organ=ORGAN)
def placenta_slab_coverage_check(placenta_mask=None, anatomical_mask=None,
                                 perfusion_map=None, cfg: QCConfig = QCConfig(),
                                 **_) -> CheckResult:
    """Does the imaging slab actually contain the whole placenta?

    Never FAILs. A clipped slab is a real limitation - published placental ASL
    slabs are ~57 mm or 8 slices, and a placenta routinely exceeds that - but it
    is a limitation of the acquisition, not evidence that the data is wrong. The
    reader needs to know the reported mean describes only the part that fitted.
    """
    if placenta_mask is None:
        return CheckResult("p3.2.slab_coverage", Verdict.UNKNOWN, reason="no mask supplied")
    m = as_mask(placenta_mask)
    if not m.any():
        return CheckResult("p3.2.slab_coverage", Verdict.UNKNOWN, reason="the mask is empty")

    n = int(m.sum())
    faces = int(m[:, :, 0].sum() + m[:, :, -1].sum()) if m.ndim == 3 and m.shape[2] > 1 else 0
    edge_frac = float(faces / n)
    metric = {"n_voxels": n, "voxels_on_slab_faces": faces,
              "edge_slice_occupied": bool(faces > 0),
              "edge_voxel_fraction": edge_frac,
              "edge_warn": cfg.placenta_edge_voxel_frac,
              # explicitly null rather than absent: a reader must be able to see
              # that coverage was NOT measured, not merely that it is missing
              "covered_fraction_vs_anatomical": None}
    covered = float("nan")
    if anatomical_mask is not None:
        a = as_mask(anatomical_mask)
        if a.shape == m.shape and a.any():
            covered = float(np.count_nonzero(m & a) / a.sum())
            metric["covered_fraction_vs_anatomical"] = covered
            metric["covered_warn"] = cfg.placenta_covered_frac

    problems = []
    if edge_frac >= cfg.placenta_edge_voxel_frac:
        problems.append(f"{edge_frac:.1%} of the mask sits on the first or last slice, so the "
                        "placenta is probably cut off by the slab")
    if np.isfinite(covered) and covered < cfg.placenta_covered_frac:
        problems.append(f"the slab covers only {covered:.0%} of the anatomical placenta")
    if problems:
        return CheckResult("p3.2.slab_coverage", Verdict.WARN, metric=metric, provisional=True,
                           reason="; ".join(problems))
    return CheckResult("p3.2.slab_coverage", Verdict.PASS, metric=metric,
                       reason=f"{edge_frac:.1%} of the mask on a slab face - the placenta appears "
                              "to fit inside the imaged volume")


# --------------------------------------------------------------------------- #
# Module P4 — labelling scheme and gestational context
# --------------------------------------------------------------------------- #
@register_qc_check("p4.1.labelling_scheme", stream="A", required=True, organ=ORGAN)
def placenta_labelling_check(labelling_scheme=None, scheme_params=None, asl_4d=None,
                             **_) -> CheckResult:
    """Which labelling scheme, and therefore WHICH CIRCULATION was measured?

    This is definitional rather than statistical, and it is why an undeclared
    scheme is a FAIL. The placenta carries two circulations, and the scheme
    decides which one the numbers describe: pCASL labelling the maternal
    descending aorta measures maternal supply only, while VSASL is contributed
    to by both. Two maps with the same units and similar values can therefore be
    measurements of different things, and nothing downstream can recover which.
    """
    if labelling_scheme in (None, "") and scheme_params in (None, {}) and asl_4d is None:
        return CheckResult("p4.1.labelling_scheme", Verdict.UNKNOWN, reason="nothing supplied")
    if labelling_scheme in (None, ""):
        return CheckResult("p4.1.labelling_scheme", Verdict.FAIL,
                           metric={"labelling_scheme": None},
                           reason="no labelling scheme declared - the placenta has two "
                                  "circulations and the scheme decides which one was measured, "
                                  "so the map's meaning is undefined")

    scheme = str(labelling_scheme).strip().lower()
    params = scheme_params if isinstance(scheme_params, dict) else {}
    # Scheme -> compartment, from the ISMRM review's own table. FAIR belongs with
    # VSASL, NOT with pCASL: pCASL labels the maternal descending aorta "to
    # selectively label maternal placental perfusion", while both VSASL and FAIR
    # are contributed to by maternal AND fetal flow. Grouping FAIR with pCASL
    # would tell a reader the map describes only the maternal circulation when it
    # does not - the exact confusion this check exists to prevent.
    if "vsasl" in scheme or "velocity" in scheme:
        critical, compartment = ("cutoff_velocity_cm_s", "post_labeling_delay_s"), "maternal_and_fetal"
    elif "fair" in scheme:
        critical, compartment = ("inversion_slab_thickness_mm",), "maternal_and_fetal"
    elif "pcasl" in scheme or "casl" in scheme:
        critical, compartment = ("labelling_plane_position",), "maternal"
    elif "pasl" in scheme:
        critical, compartment = ("inversion_slab_thickness_mm",), "maternal_and_fetal"
    else:
        critical, compartment = (), "unknown"
    missing = [k for k in critical if params.get(k) in (None, "")]
    metric = {"labelling_scheme": labelling_scheme, "measured_compartment": compartment,
              "scheme_params": params, "critical_params": list(critical),
              "missing_params": missing}
    if missing:
        return CheckResult("p4.1.labelling_scheme", Verdict.WARN, metric=metric,
                           reason=f"{labelling_scheme} declared ({compartment}), but the "
                                  f"scheme-critical parameters {missing} are missing - the "
                                  "measurement is not reproducible")
    return CheckResult("p4.1.labelling_scheme", Verdict.PASS, metric=metric,
                       reason=f"{labelling_scheme}, measuring {compartment} circulation, with "
                              "its scheme-critical parameters recorded")


@register_qc_check("p4.2.ga_context", stream="A", required=True, organ=ORGAN)
def placenta_ga_context_check(gestational_age_wk=None, maternal_position=None,
                              maternal_bmi=None, field_strength_T=None,
                              placental_location=None, cfg: QCConfig = QCConfig(),
                              **_) -> CheckResult:
    """Gestational age and the maternal/scanner context.

    GA is not optional metadata here. Placental perfusion changes systematically
    across gestation, so a perfusion value without a GA cannot be interpreted at
    all - hence a definitional FAIL rather than a missing-field WARN. Maternal
    position matters for a similar reason (lateral 207 +/- 39 vs supine
    171 +/- 3x in the same study), so it is recorded and its absence noted.
    """
    context = {"maternal_position": maternal_position, "maternal_bmi": maternal_bmi,
               "field_strength_T": field_strength_T, "placental_location": placental_location}
    if gestational_age_wk is None and all(v is None for v in context.values()):
        return CheckResult("p4.2.ga_context", Verdict.UNKNOWN, reason="no metadata at all")
    metric = {"gestational_age_wk": gestational_age_wk, **context,
              "studied_range_wk": [cfg.placenta_ga_min_wk, cfg.placenta_ga_max_wk],
              "magnitude_band_applied": False}
    if gestational_age_wk is None:
        return CheckResult("p4.2.ga_context", Verdict.FAIL, metric=metric,
                           reason="gestational age absent - placental perfusion changes across "
                                  "gestation, so the value cannot be interpreted")
    try:
        ga = float(gestational_age_wk)
    except (TypeError, ValueError):
        return CheckResult("p4.2.ga_context", Verdict.UNKNOWN, metric=metric,
                           reason=f"gestational age {gestational_age_wk!r} is not a number")

    problems = []
    if not (cfg.placenta_ga_min_wk <= ga <= cfg.placenta_ga_max_wk):
        problems.append(f"GA {ga:g} wk is outside the studied {cfg.placenta_ga_min_wk:g}-"
                        f"{cfg.placenta_ga_max_wk:g} wk range")
    if maternal_position in (None, ""):
        problems.append("maternal position not recorded (supine and lateral differ materially)")
    if problems:
        return CheckResult("p4.2.ga_context", Verdict.WARN, metric=metric,
                           reason="; ".join(problems))
    return CheckResult("p4.2.ga_context", Verdict.PASS, metric=metric,
                       reason=f"GA {ga:g} wk, {maternal_position} position"
                              + (f", {field_strength_T} T" if field_strength_T else ""))


# --------------------------------------------------------------------------- #
# Module P5 — M0 and quantification
# --------------------------------------------------------------------------- #
@register_qc_check("p5.1.m0_state", stream="A", required=True, organ=ORGAN)
def placenta_m0_state_check(m0=None, m0_labelled=None, m0_background_suppressed=None,
                            asl_background_suppressed=None, quantified=None,
                            perfusion_map=None, m0_tr_s=None, **_) -> CheckResult:
    """Is there an M0, is it unlabelled, and was it background-suppressed?

    The FAILs here are all definitional: claiming quantified perfusion with no
    M0 means the calibration denominator is unknown; a labelled or
    background-suppressed M0 measures less signal than the tissue actually has,
    so every perfusion value derived from it is inflated.

    Note the asymmetry that catches people out - background suppression must be
    OFF for the M0 but is expected ON for the ASL pairs. An absent-BS ASL series
    is therefore only a WARN, and there is no published placental rule making it
    an error.
    """
    if m0 is None:
        if quantified:
            return CheckResult("p5.1.m0_state", Verdict.FAIL,
                               metric={"m0_present": False, "quantified": True},
                               reason="quantified perfusion is claimed but no M0 was supplied - "
                                      "the calibration denominator is unknown")
        return CheckResult("p5.1.m0_state", Verdict.WARN,
                           metric={"m0_present": False, "quantified": bool(quantified)},
                           reason="no M0 - a perfusion-weighted image can be inspected but not "
                                  "quantified")
    metric = {"m0_present": True, "m0_labelled": m0_labelled,
              "m0_background_suppressed": m0_background_suppressed,
              "asl_background_suppressed": asl_background_suppressed,
              "quantified": bool(quantified)}
    if perfusion_map is not None:
        same = np.asarray(m0).shape[:3] == np.asarray(perfusion_map).shape[:3]
        metric["grid_matches"] = same
        if not same:
            return CheckResult("p5.1.m0_state", Verdict.FAIL, metric=metric,
                               reason=f"M0 {np.asarray(m0).shape[:3]} and the perfusion map "
                                      f"{np.asarray(perfusion_map).shape[:3]} are on different "
                                      "grids - they cannot divide")
    if m0_labelled:
        return CheckResult("p5.1.m0_state", Verdict.FAIL, metric=metric,
                           reason="the M0 carries labelling - it is not a calibration image")
    if m0_background_suppressed:
        if quantified:
            return CheckResult("p5.1.m0_state", Verdict.FAIL, metric=metric,
                               reason="the M0 was background-suppressed - the tissue signal it "
                                      "exists to measure has been crushed, so perfusion is "
                                      "over-estimated")
        # Not a failure when no quantification is claimed: the M0 is then a
        # reference image rather than a calibration denominator. But it is never
        # a PASS, and the earlier version fell through to one whose reason said
        # "not background-suppressed" about a background-suppressed M0.
        return CheckResult("p5.1.m0_state", Verdict.WARN, metric=metric,
                           reason="the M0 was background-suppressed - usable as a reference "
                                  "image, but it cannot serve as a calibration denominator")
    if m0_labelled is None or m0_background_suppressed is None:
        return CheckResult("p5.1.m0_state", Verdict.UNKNOWN, metric=metric,
                           reason="the M0's labelling and background-suppression state are not "
                                  "recorded")
    metric["m0_tr_s"] = m0_tr_s
    soft = []
    if asl_background_suppressed is False:
        soft.append("the ASL pairs were not background-suppressed - this may be deliberate; "
                    "there is no published placental rule requiring it")
    if m0_tr_s is None:
        soft.append("the M0's repetition time is not recorded, so incomplete relaxation cannot "
                    "be ruled out")
    if soft:
        return CheckResult("p5.1.m0_state", Verdict.WARN, metric=metric,
                           reason="M0 is clean, but " + "; ".join(soft))
    return CheckResult("p5.1.m0_state", Verdict.PASS, metric=metric,
                       reason="M0 present, unlabelled and not background-suppressed")


@register_qc_check("p5.2.m0_heterogeneity", stream="A", required=True, organ=ORGAN)
def placenta_m0_heterogeneity_check(m0=None, placenta_mask=None, normalisation_mode=None,
                                    gestational_age_wk=None, perfusion_map=None,
                                    segment_cov=None, cfg: QCConfig = QCConfig(),
                                    **_) -> CheckResult:
    """How structured is the M0 inside the placenta, and does the normalisation
    strategy cope with it?

    Never FAILs. The placental M0 really is spatially structured - that is
    physiology plus coil sensitivity, not a defect. What matters is the
    interaction with normalisation: dividing voxel-wise by a structured M0
    imprints that structure onto the perfusion map, where it will later be read
    as perfusion heterogeneity. A scalar reference sidesteps the problem.
    """
    if m0 is None or placenta_mask is None:
        return CheckResult("p5.2.m0_heterogeneity", Verdict.UNKNOWN, reason="needs an M0 and a mask")
    bad = _grid_error(m0, placenta_mask)
    if bad:
        return CheckResult("p5.2.m0_heterogeneity", Verdict.UNKNOWN, reason=bad)

    vals = roi_values(m0, placenta_mask)
    if vals.size < cfg.placenta_min_roi_voxels:
        return CheckResult("p5.2.m0_heterogeneity", Verdict.UNKNOWN,
                           metric={"n_voxels": int(vals.size)},
                           reason=f"only {vals.size} M0 voxels inside the mask")
    mu = float(vals.mean())
    m0_cov = float(vals.std(ddof=1) / mu) if mu > 0 else float("nan")
    scalar_ref = float(np.percentile(vals, 80))
    mode = (normalisation_mode or "not stated").strip().lower()
    # A voxel-wise divide propagates M0 structure into the perfusion map; a
    # scalar divide cannot. How much structure counts as "high" is SELF-
    # REFERENCED: no published placental M0-heterogeneity bound exists, so the
    # M0's structure is compared against the perfusion map's OWN structure. If
    # the M0 varies as much as the perfusion does, a voxel-wise divide is
    # imprinting a comparable amount of M0 pattern onto the result. An absolute
    # cut-off here would be an invented number pretending to be a measurement.
    ref_cov = float("nan")
    ref_source = None
    if perfusion_map is not None and _grid_error(perfusion_map, placenta_mask) is None:
        pv = roi_values(perfusion_map, placenta_mask)
        if pv.size and pv.mean() > 0:
            ref_cov = float(pv.std(ddof=1) / pv.mean())
            ref_source = "perfusion map, in mask"
    elif isinstance(segment_cov, (int, float)) and np.isfinite(segment_cov):
        ref_cov, ref_source = float(segment_cov), "P2.3 segment CoV"
    if np.isfinite(ref_cov):
        high_structure = np.isfinite(m0_cov) and m0_cov >= ref_cov
    else:
        # Nothing to self-reference against. Rather than invent an absolute
        # bound, the risk is left undetermined and the strategy alone is judged.
        high_structure = False
    metric = {"m0_in_mask_cov": m0_cov, "scalar_reference_p80": scalar_ref,
              "m0_median": float(np.median(vals)),
              "normalisation_mode": normalisation_mode or "not stated",
              "comparison_cov": ref_cov, "comparison_source": ref_source or "none available",
              "structure_risk": ("high" if high_structure
                                 else ("low" if np.isfinite(ref_cov) else "undetermined")),
              "gestational_age_wk": gestational_age_wk, "n_voxels": int(vals.size)}
    if "voxel" in mode and high_structure:
        return CheckResult("p5.2.m0_heterogeneity", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"M0 CoV {m0_cov:.2f} inside the placenta is at or above the "
                                  f"perfusion map's own {ref_cov:.2f}, and normalisation is "
                                  "voxel-wise - that structure will be imprinted on the map and "
                                  "later read as perfusion heterogeneity")
    if mode in ("not stated", ""):
        return CheckResult("p5.2.m0_heterogeneity", Verdict.WARN, metric=metric,
                           reason=f"M0 CoV {m0_cov:.2f} inside the placenta, but the "
                                  "normalisation strategy was not stated")
    return CheckResult("p5.2.m0_heterogeneity", Verdict.PASS, metric=metric,
                       reason=f"M0 CoV {m0_cov:.2f} inside the placenta with {mode} "
                              "normalisation")


@register_qc_check("p5.3.quant_constants", stream="A", required=True, organ=ORGAN)
def placenta_quant_constants_check(constants=None, field_strength_T=None,
                                   labelling_scheme=None, quantified=None,
                                   **_) -> CheckResult:
    """Are the quantification constants recorded, and is T1-blood consistent
    with the field strength?

    Never FAILs. The published placental ranges (lambda 0.9-1.0, alpha
    0.6-0.767) are context, not bounds - the field is too young to say a value
    outside them is wrong. What IS checkable is internal consistency: a T1-blood
    of 1650 ms at 1.5 T is a transcription error whatever the true placental
    value turns out to be.
    """
    cons = constants if isinstance(constants, dict) else {}
    if not cons and not quantified:
        return CheckResult("p5.3.quant_constants", Verdict.UNKNOWN,
                           reason="no constants recorded and no quantification claimed")
    needed = ("lambda", "alpha", "t1_blood_ms")
    missing = [k for k in needed if cons.get(k) in (None, "")]
    metric = {"constants": {k: cons.get(k) for k in needed}, "missing": missing,
              "field_strength_T": field_strength_T, "labelling_scheme": labelling_scheme,
              "published_context": {"lambda": [0.9, 1.0], "alpha": [0.6, 0.767],
                                    "t1_blood_ms": {"3T": 1650, "1.5T": 1350}}}
    problems = [f"missing {missing}"] if missing else []

    t1 = cons.get("t1_blood_ms")
    if isinstance(t1, (int, float)) and field_strength_T:
        expected = 1650.0 if float(field_strength_T) >= 2.5 else 1350.0
        metric["t1_blood_expected_ms"] = expected
        if abs(float(t1) - expected) > 200:
            problems.append(f"T1-blood {t1:g} ms is not consistent with "
                            f"{field_strength_T} T (expected near {expected:g} ms)")
    if problems:
        return CheckResult("p5.3.quant_constants", Verdict.WARN, metric=metric,
                           reason="; ".join(problems))
    return CheckResult("p5.3.quant_constants", Verdict.PASS, metric=metric,
                       reason="lambda, alpha and T1-blood recorded and internally consistent")


# --------------------------------------------------------------------------- #
# Module P6 — motion, the dominant placental problem
# --------------------------------------------------------------------------- #
@register_qc_check("p6.1.pair_outliers", stream="A", required=True, organ=ORGAN)
def placenta_pair_outliers_check(delta_m_4d=None, placenta_mask=None, labelling_scheme=None,
                                 cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Per-pair subtraction outlier rejection — the one implementable published
    rule for placental ASL.

    Fewer than four surviving pairs is a definitional FAIL: an average over
    three subtractions is not an average, and every downstream number would be
    dominated by whichever pairs happened to survive.
    """
    if delta_m_4d is None or placenta_mask is None:
        return CheckResult("p6.1.pair_outliers", Verdict.UNKNOWN,
                           reason="needs the 4D subtraction series and a placenta mask")
    arr = np.asarray(delta_m_4d, dtype=float)
    if arr.ndim != 4 or arr.shape[3] < 2:
        return CheckResult("p6.1.pair_outliers", Verdict.UNKNOWN,
                           reason=f"expected a 4D series with repetitions, got {arr.shape}")
    bad = _grid_error(arr[..., 0], placenta_mask)
    if bad:
        return CheckResult("p6.1.pair_outliers", Verdict.UNKNOWN, reason=bad)

    m = as_mask(placenta_mask)
    series = arr[m]                                    # (n_vox, T)
    # see the note in checks/kidney.py: a sample SD needs two finite points, so
    # it is only asked for where it is defined
    n_finite = np.isfinite(series).sum(axis=1, keepdims=True)
    mu = np.full((series.shape[0], 1), np.nan)
    sd = np.full((series.shape[0], 1), np.nan)
    enough = (n_finite >= 2).ravel()
    if enough.any():
        mu[enough] = np.nanmean(series[enough], axis=1, keepdims=True)
        sd[enough] = np.nanstd(series[enough], axis=1, ddof=1, keepdims=True)
    valid = np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    rejected, fracs, signs = [], [], []
    for p in range(series.shape[1]):
        col = series[:, p:p + 1]
        good = valid & np.isfinite(col)
        if not good.any():
            rejected.append(p); fracs.append(float("nan")); signs.append("unmeasurable"); continue
        resid = (col[good] - mu[good]) / sd[good]
        dev = np.abs(resid) > cfg.placenta_outlier_sd
        frac = float(dev.sum() / dev.size)
        fracs.append(frac)
        if frac > cfg.placenta_outlier_voxel_frac:
            rejected.append(p)
            # Keep the SIGN. In VSASL a HIGH outlier is the signature of moving
            # tissue being labelled as if it were blood - a physically different
            # failure from a low outlier, and one a reader can act on. Discarding
            # it (which taking the absolute value alone does) makes the
            # spurious-labelling warning unreachable.
            signs.append("high" if float(np.mean(resid[dev])) > 0 else "low")

    n_pairs = series.shape[1]
    surviving = n_pairs - len(rejected)
    rejected_frac = float(len(rejected) / n_pairs)
    n_high = signs.count("high")
    n_low = signs.count("low")
    metric = {"n_pairs": n_pairs, "n_rejected": len(rejected), "surviving_pairs": surviving,
              "rejected_fraction": rejected_frac, "rejected_indices": rejected,
              "n_rejected_high": n_high, "n_rejected_low": n_low,
              "rejection_signs": signs,
              "rule": f"+/-{cfg.placenta_outlier_sd} SD on more than "
                      f"{cfg.placenta_outlier_voxel_frac:.0%} of placental voxels",
              "deviating_voxel_fraction_per_pair": [round(f, 4) if np.isfinite(f) else None
                                                    for f in fracs],
              "thresholds": {"warn_fraction": cfg.placenta_rejected_frac_warn,
                             "severe_fraction": cfg.placenta_rejected_frac_severe,
                             "min_surviving": cfg.placenta_min_surviving_pairs,
                             "good_surviving": cfg.placenta_good_surviving_pairs}}
    if surviving < cfg.placenta_min_surviving_pairs:
        return CheckResult("p6.1.pair_outliers", Verdict.FAIL, metric=metric,
                           reason=f"only {surviving} of {n_pairs} pairs survive - fewer than "
                                  f"{cfg.placenta_min_surviving_pairs} is not an average")
    problems = []
    if surviving < cfg.placenta_good_surviving_pairs:
        problems.append(f"only {surviving} pairs survive")
    if rejected_frac > cfg.placenta_rejected_frac_severe:
        problems.append(f"{rejected_frac:.1%} of pairs rejected, well above the "
                        f"{cfg.placenta_rejected_frac_warn:.1%} line")
    elif rejected_frac > cfg.placenta_rejected_frac_warn:
        problems.append(f"{rejected_frac:.1%} of pairs rejected")
    # Only for a DECLARED VSASL acquisition. The physical argument is
    # scheme-specific - in VSASL a high outlier means moving tissue labelled as
    # if it were blood - so raising it on an unknown scheme would be asserting a
    # mechanism that may not apply.
    scheme = str(labelling_scheme or "").lower()
    if n_high and "vsasl" in scheme:
        problems.append(f"{n_high} rejected for EXCESS signal - in VSASL that is the signature of "
                        "moving tissue being labelled as if it were blood, not simple motion")
    if problems:
        return CheckResult("p6.1.pair_outliers", Verdict.WARN, metric=metric, provisional=True,
                           reason="; ".join(problems))
    return CheckResult("p6.1.pair_outliers", Verdict.PASS, metric=metric,
                       reason=f"{surviving} of {n_pairs} pairs survive "
                              f"({rejected_frac:.1%} rejected)")


@register_qc_check("p6.2.temporal_sd", stream="A", required=True, organ=ORGAN)
def placenta_temporal_sd_check(delta_m_4d=None, asl_source_4d=None, placenta_mask=None,
                               surviving_pairs=None, labelling_scheme=None,
                               field_strength_T=None, context=None,
                               cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Temporal stability of the placental signal after motion correction.

    The statistic is a VOXELWISE ratio averaged in the mask - per-voxel SD over
    the retained SOURCE volumes divided by that voxel's own mean, then averaged
    across the placenta - not the spread of whole-ROI means. The two are
    different quantities: an ROI mean is stable while individual voxels swing
    wildly, so the ROI version reports calm on a scan that is not.

    Graded only against a cohort-comparable reference (6.7 +/- 3.1%), and
    comparability is DERIVED, not taken on trust: the reference cohort is VSASL
    at 3 T, which is the only setting it was measured in. Anything else is
    reported as INFO, because an incomparable threshold is worse than none.
    """
    source = asl_source_4d if asl_source_4d is not None else delta_m_4d
    if source is None or placenta_mask is None:
        return CheckResult("p6.2.temporal_sd", Verdict.UNKNOWN, reason="needs a 4D series and a mask")
    arr = np.asarray(source, dtype=float)
    if arr.ndim != 4:
        return CheckResult("p6.2.temporal_sd", Verdict.UNKNOWN,
                           reason=f"expected a 4D series, got {arr.shape}")
    bad = _grid_error(arr[..., 0], placenta_mask)
    if bad:
        return CheckResult("p6.2.temporal_sd", Verdict.UNKNOWN, reason=bad)
    keep = [t for t in range(arr.shape[3]) if np.isfinite(arr[..., t]).any()]
    if len(keep) < 3:
        return CheckResult("p6.2.temporal_sd", Verdict.UNKNOWN,
                           metric={"n_usable_volumes": len(keep)},
                           reason=f"only {len(keep)} usable volumes; at least 3 are needed")

    m = as_mask(placenta_mask)

    def _voxelwise_ratio(series, invert=False):
        """Mean over mask voxels of (SD/mean), or (mean/SD) when inverted."""
        vals = series[m]                            # (n_vox, T)
        n_fin = np.isfinite(vals).sum(axis=1)
        ok = n_fin >= 2
        if not ok.any():
            return float("nan"), 0
        mu = np.nanmean(vals[ok], axis=1)
        sd = np.nanstd(vals[ok], axis=1, ddof=1)
        good = np.isfinite(mu) & np.isfinite(sd) & (np.abs(mu) > 0) & (sd > 0)
        if not good.any():
            return float("nan"), 0
        ratio = (np.abs(mu[good]) / sd[good]) if invert else (sd[good] / np.abs(mu[good]))
        return float(ratio.mean()), int(good.sum())

    tsd, n_vox = _voxelwise_ratio(arr[..., keep])
    tsd_pct = float(tsd * 100.0) if np.isfinite(tsd) else float("nan")

    # tSNR of the SUBTRACTION series is a separate, complementary number
    tsnr_dm = float("nan")
    if delta_m_4d is not None:
        dm = np.asarray(delta_m_4d, dtype=float)
        if dm.ndim == 4 and dm.shape[:3] == m.shape:
            dkeep = [t for t in range(dm.shape[3]) if np.isfinite(dm[..., t]).any()]
            if len(dkeep) >= 2:
                tsnr_dm, _ = _voxelwise_ratio(dm[..., dkeep], invert=True)

    ctx = context if isinstance(context, dict) else {}
    scheme = str(labelling_scheme or ctx.get("labelling_scheme") or "").lower()
    field = field_strength_T if field_strength_T is not None else ctx.get("field_strength_T")
    comparable = bool("vsasl" in scheme and field is not None and abs(float(field) - 3.0) < 0.3)
    metric = {"normalised_tsd_pct": tsd_pct,
              "definition": "voxelwise SD over source images / voxelwise mean, averaged in "
                            "mask, as percent",
              "tsnr_delta_m": tsnr_dm, "n_voxels": n_vox,
              "n_volumes_used": len(keep), "surviving_pairs": surviving_pairs,
              "cohort_comparable": comparable,
              "comparability_rule": "the reference cohort is VSASL at 3 T; anything else is "
                                    "reported without a verdict",
              "labelling_scheme": labelling_scheme, "field_strength_T": field,
              "reference": {"mean_pct": 6.7, "sd_pct": 3.1},
              "warn_pct": cfg.placenta_tsd_warn_pct}
    if not np.isfinite(tsd_pct):
        return CheckResult("p6.2.temporal_sd", Verdict.UNKNOWN, metric=metric,
                           reason="no placental voxel has a usable temporal mean and SD")
    if comparable and tsd_pct > cfg.placenta_tsd_warn_pct:
        return CheckResult("p6.2.temporal_sd", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"voxelwise temporal SD {tsd_pct:.1f}% of the mean, above the "
                                  f"{cfg.placenta_tsd_warn_pct}% line for the comparable VSASL "
                                  "3 T cohort (reference 6.7 +/- 3.1%)")
    tail = ("" if comparable else " - not a VSASL 3 T acquisition, so the reference does not "
                                 "apply and no verdict is given")
    return CheckResult("p6.2.temporal_sd", Verdict.INFO, metric=metric,
                       reason=f"voxelwise temporal SD {tsd_pct:.1f}% of the mean over "
                              f"{len(keep)} volumes (reference 6.7 +/- 3.1%){tail}")


@register_qc_check("p6.3.registration_residual", stream="A", required=True, organ=ORGAN)
def placenta_registration_check(asl_source_4d=None, delta_m_4d=None, placenta_mask=None,
                                registration_model=None, reference_volume=None,
                                cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Was a deformable registration used, and how much residual deformation is
    left?

    A rigid model is flagged because the placenta is not rigid - it deforms with
    maternal breathing and uterine activity, so a rigid transform cannot in
    principle remove the motion. Residual similarity to the reference volume is
    measured with normalised cross-correlation inside the mask.
    """
    series_in = asl_source_4d if asl_source_4d is not None else delta_m_4d
    if series_in is None or placenta_mask is None:
        return CheckResult("p6.3.registration_residual", Verdict.UNKNOWN,
                           reason="needs a 4D series and a mask")
    arr = np.asarray(series_in, dtype=float)
    if arr.ndim != 4 or arr.shape[3] < 3:
        return CheckResult("p6.3.registration_residual", Verdict.UNKNOWN,
                           reason="fewer than 3 volumes")
    bad = _grid_error(arr[..., 0], placenta_mask)
    if bad:
        return CheckResult("p6.3.registration_residual", Verdict.FAIL, reason=bad)

    m = as_mask(placenta_mask)
    # `reference_volume` is documented as an INDEX into the series; an array is
    # accepted too, because a caller with a separately-reconstructed reference
    # has nowhere else to put it.
    ref_source = "median volume"
    if isinstance(reference_volume, (int, np.integer)) and not isinstance(reference_volume, bool):
        idx = int(reference_volume)
        if not (0 <= idx < arr.shape[3]):
            return CheckResult("p6.3.registration_residual", Verdict.FAIL,
                               metric={"reference_volume": idx, "n_volumes": int(arr.shape[3])},
                               reason=f"reference volume index {idx} is outside the series "
                                      f"(0-{arr.shape[3] - 1}) - no reference is resolvable")
        ref_vol = arr[..., idx]
        ref_source = f"volume {idx}"
    elif reference_volume is not None:
        ref_vol = np.asarray(reference_volume, dtype=float)
        if ref_vol.shape != arr.shape[:3]:
            return CheckResult("p6.3.registration_residual", Verdict.FAIL,
                               reason=f"reference volume {ref_vol.shape} does not match the "
                                      f"series {arr.shape[:3]}")
        ref_source = "supplied array"
    else:
        ref_vol = np.nanmedian(arr, axis=3)    # the median volume is a robust reference
    ref = ref_vol[m]

    # Is NCC informative here at all? A correlation between two volumes measures
    # how well their SPATIAL STRUCTURE lines up, so it needs structure to exist.
    # A placenta that is genuinely uniform gives a low NCC no matter how perfect
    # the registration is - the correlation is then between two noise fields.
    # Measured as the ratio of the reference's spatial spread to the typical
    # voxel-wise temporal spread (the noise), both inside the mask.
    noise = float(np.nanmedian(np.nanstd(arr[m], axis=1, ddof=1)))
    structure = float(np.nanstd(ref[np.isfinite(ref)])) if np.isfinite(ref).any() else float("nan")
    cnr = float(structure / noise) if noise > 0 and np.isfinite(structure) else float("nan")
    informative = np.isfinite(cnr) and cnr >= 1.0
    nccs = []
    for t in range(arr.shape[3]):
        v = arr[..., t][m]
        good = np.isfinite(v) & np.isfinite(ref)
        nccs.append(pearson(v[good], ref[good]) if good.sum() > 2 else float("nan"))
    # Local SSIM against the same reference: NCC is global and can stay high
    # while a region is locally deformed, which is exactly what a contraction
    # does. The in-mask MINIMUM is taken, so one badly-deformed region is not
    # averaged away by the rest of the placenta.
    ssims = []
    for t in range(arr.shape[3]):
        smap = local_ssim(arr[..., t], ref_vol, radius=_SSIM_RADIUS)[m]
        smap = smap[np.isfinite(smap)]
        ssims.append(float(smap.min()) if smap.size else float("nan"))
    ssims = np.asarray(ssims, dtype=float)
    ssim_finite = ssims[np.isfinite(ssims)]

    nccs = np.asarray(nccs, dtype=float)
    finite = nccs[np.isfinite(nccs)]
    if finite.size == 0:
        return CheckResult("p6.3.registration_residual", Verdict.UNKNOWN,
                           reason="no volume could be compared with the reference")
    below = float(np.count_nonzero(finite < cfg.placenta_ncc_pass) / finite.size)
    below_ssim = (float(np.count_nonzero(ssim_finite < cfg.placenta_ssim_pass) / ssim_finite.size)
                  if ssim_finite.size else float("nan"))
    model = (registration_model or "not declared").strip().lower()
    non_rigid = any(t in model for t in ("non-rigid", "nonrigid", "deform", "elastic",
                                         "bspline", "b-spline", "svr", "dsvr", "affine+"))
    metric = {"median_ncc": float(np.median(finite)), "min_ncc": float(finite.min()),
              "fraction_below_ncc": below, "ncc_pass": cfg.placenta_ncc_pass,
              "registration_model": registration_model or "not declared",
              "model_is_non_rigid": non_rigid, "n_volumes": int(finite.size),
              "reference": ref_source,
              "min_local_ssim": float(ssim_finite.min()) if ssim_finite.size else float("nan"),
              "median_local_ssim": float(np.median(ssim_finite)) if ssim_finite.size else float("nan"),
              "fraction_below_ssim": below_ssim, "ssim_pass": cfg.placenta_ssim_pass,
              "ssim_kernel_voxels": 2 * _SSIM_RADIUS + 1,
              "spatial_contrast_to_noise": cnr, "ncc_informative": bool(informative)}
    if not informative:
        # The model question is still worth answering; the NCC number is not.
        note = (f"spatial contrast-to-noise {cnr:.2f} inside the placenta - too little "
                "structure for correlation to say anything about registration")
        if not non_rigid:
            return CheckResult("p6.3.registration_residual", Verdict.WARN, metric=metric,
                               reason=f"registration model is "
                                      f"'{metric['registration_model']}', which cannot remove "
                                      f"placental deformation; {note}, so the residual was not "
                                      "graded")
        return CheckResult("p6.3.registration_residual", Verdict.INFO, metric=metric,
                           reason=f"non-rigid registration declared; {note}, so the residual "
                                  "was not graded")
    if not non_rigid:
        return CheckResult("p6.3.registration_residual", Verdict.WARN, metric=metric,
                           reason=f"registration model is '{metric['registration_model']}' - the "
                                  "placenta deforms, so a rigid or undeclared model cannot in "
                                  f"principle remove its motion (median NCC "
                                  f"{metric['median_ncc']:.2f})")
    bad = []
    if below >= cfg.placenta_bad_volume_frac:
        bad.append(f"{below:.0%} of volumes below NCC {cfg.placenta_ncc_pass}")
    if np.isfinite(below_ssim) and below_ssim >= cfg.placenta_bad_volume_frac:
        bad.append(f"{below_ssim:.0%} of volumes below local SSIM {cfg.placenta_ssim_pass}")
    if bad:
        return CheckResult("p6.3.registration_residual", Verdict.WARN, metric=metric,
                           provisional=True,
                           reason=f"residual deformation against the reference: {'; '.join(bad)} "
                                  f"(median NCC {metric['median_ncc']:.2f}, min local SSIM "
                                  f"{metric['min_local_ssim']:.2f})")
    return CheckResult("p6.3.registration_residual", Verdict.PASS, metric=metric,
                       reason=f"non-rigid registration, median NCC {metric['median_ncc']:.2f} and "
                              f"min local SSIM {metric['min_local_ssim']:.2f}, "
                              f"{below:.0%} of volumes below the NCC line")


@register_qc_check("p6.4.contraction_events", stream="A", required=True, organ=ORGAN)
def placenta_contraction_check(asl_source_4d=None, delta_m_4d=None, placenta_mask=None,
                               tr_s=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Candidate uterine contraction / bulk-deformation events. Report-only: INFO.

    Contractions are common (reported in 60% or more of scans) and they transiently
    squeeze the placenta, so a perfusion average taken across one mixes two
    physiological states. Candidate events are surfaced for a human to look at,
    never graded: the tool cannot distinguish a contraction from a large breath,
    and calling one a defect would be a claim it cannot support.
    """
    series_in = asl_source_4d if asl_source_4d is not None else delta_m_4d
    if series_in is None or placenta_mask is None:
        return CheckResult("p6.4.contraction_events", Verdict.UNKNOWN,
                           reason="needs a 4D series and a mask")
    arr = np.asarray(series_in, dtype=float)
    if arr.ndim != 4 or arr.shape[3] < cfg.placenta_min_volumes_contraction:
        return CheckResult("p6.4.contraction_events", Verdict.UNKNOWN,
                           reason=f"fewer than {cfg.placenta_min_volumes_contraction} volumes")
    bad = _grid_error(arr[..., 0], placenta_mask)
    if bad:
        return CheckResult("p6.4.contraction_events", Verdict.UNKNOWN, reason=bad)

    m = as_mask(placenta_mask)
    # A contraction thickens and shrinks the placental cross-section, so the
    # share of the mask still carrying placental signal drops.
    #
    # The reference level has to come from OUTSIDE the volume being measured,
    # and it has to sit LOW. Two earlier versions were wrong in instructive ways:
    # each volume's own median makes the statistic 0.5 for every volume by
    # definition, and the series median puts every normal volume at ~0.5 too, so
    # ordinary noise crosses the 10% line. A low percentile of the whole series
    # instead puts a normal volume near 0.75 and a collapsed one near 0, which
    # separates the event from the noise instead of competing with it.
    all_vals = arr[m]
    finite_all = all_vals[np.isfinite(all_vals)]
    if finite_all.size == 0:
        return CheckResult("p6.4.contraction_events", Verdict.UNKNOWN,
                           reason="no finite placental signal in the series")
    level = float(np.percentile(finite_all, 25))
    occupancy = []
    for t in range(arr.shape[3]):
        v = arr[..., t][m]
        v = v[np.isfinite(v)]
        occupancy.append(float(np.count_nonzero(v > level) / v.size) if v.size else np.nan)
    occ = np.asarray(occupancy, dtype=float)
    finite = occ[np.isfinite(occ)]
    if finite.size < cfg.placenta_min_volumes_contraction:
        return CheckResult("p6.4.contraction_events", Verdict.UNKNOWN,
                           reason="too few volumes with finite placental signal")
    baseline = float(np.median(finite))
    # The design's rule is "a drop of more than 10% below baseline". On its own
    # that surfaces noise: occupancy is a proportion over a few hundred mask
    # voxels, so its sampling error alone crosses 10% every few volumes and a
    # perfectly calm series reports phantom "events". A candidate must therefore
    # ALSO be outside the series' own variability, measured robustly (MAD, scaled
    # to a normal SD) so one real event cannot inflate the floor that is supposed
    # to catch it. Both conditions, never either.
    mad = float(np.median(np.abs(finite - baseline)))
    noise_floor = 3.0 * 1.4826 * mad
    drops = [int(t) for t, v in enumerate(occ)
             if np.isfinite(v) and baseline > 0
             and (baseline - v) / baseline > cfg.placenta_contraction_drop
             and (baseline - v) > noise_floor]
    metric = {"occupancy_per_volume": [round(float(v), 4) if np.isfinite(v) else None for v in occ],
              "occupancy_definition": "share of masked voxels above the 25th percentile of the "
                                      "whole series inside the mask",
              "reference_level": level, "baseline_occupancy": baseline,
              "noise_floor": noise_floor, "candidate_event_volumes": drops,
              "n_candidate_events": len(drops),
              "drop_threshold": cfg.placenta_contraction_drop, "tr_s": tr_s,
              "prevalence_context": "contractions are reported in 60% or more of placental scans",
              "graded": False}
    if tr_s and drops:
        metric["candidate_event_times_s"] = [round(t * float(tr_s), 1) for t in drops]
    if drops:
        return CheckResult("p6.4.contraction_events", Verdict.INFO, metric=metric,
                           reason=f"{len(drops)} candidate contraction/deformation event(s) at "
                                  f"volume(s) {drops[:6]}{'...' if len(drops) > 6 else ''} - "
                                  "surfaced for review, not graded")
    return CheckResult("p6.4.contraction_events", Verdict.INFO, metric=metric,
                       reason="no candidate contraction events detected - reported, not graded")
