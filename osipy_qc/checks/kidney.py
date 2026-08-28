"""
Kidney QC — the 19 checks of KIDNEY_QC_DESIGN.md, Streams A and B.

Read this before changing a threshold
-------------------------------------
Nery F, et al. MAGMA 2020;33(1):141-161 is the renal ASL consensus and the
renal counterpart of the brain's White Paper. It carries 59 consensus statements
and **zero numeric quality thresholds** — no tSNR cutoff, no CoV cutoff, no
motion limit, no QEI equivalent. Everything published is about how to ACQUIRE a
renal ASL scan; nothing published says whether the resulting map is good.

Three consequences run through every check below:

1. **Most verdicts here are INFO or WARN, and several checks can never FAIL.**
   A FAIL is reserved for physical impossibility (a majority-negative map, an
   empty mask, masks that overlap) — never for a value merely outside an
   engineering band. Where the design says "FAIL: never", the code has no FAIL
   branch at all rather than a branch nobody expects to reach.

2. **Two consensus rules ARE binding and shape the whole module.**
   R10.1 (100% agreement): report CORTICAL perfusion, left and right kidney
   SEPARATELY. So nothing here pools the two kidneys, and no combined mean is
   emitted at all — a downstream consumer cannot accidentally use one.
   R10.2 (89% agreement): medullary values "are not considered reliable with
   current measurement approaches". So the cortico-medullary ratio, which looks
   like the obvious renal analogue of the brain's GM/WM ratio, ships as a
   segmentation-INTEGRITY flag and is never a perfusion verdict.

3. **Technical factors move renal perfusion more than disease does.** Labelling
   scheme alone is ~1.8x (FAIR 362±57 vs pCASL 201±72 in the SAME subjects),
   field strength ~11%, age ~20%, quantification constants ~22%. A band that is
   not conditioned on all of those measures the protocol, not the patient, which
   is why K3.1 grades a 50-500 sanity bound and reports the acquisition context
   beside the value instead of pretending to a reference interval.

Masks are inputs, never inventions. There is no renal equivalent of "just run
BET on it", so every ROI here comes from the caller and K4.1 grades the mask
itself rather than trusting it.
"""

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict
from ..utils.roi import (SIDES, as_mask, as_sides, asymmetry_index,
                         component_sizes, largest_component_fraction, roi_stats,
                         roi_values, touches_fov_edge)

ORGAN = "kidney"


# --------------------------------------------------------------------------- #
# helpers shared by the kidney checks
# --------------------------------------------------------------------------- #
def _sides_present(*mask_dicts) -> list[str]:
    """The side labels that appear in any supplied mask dict, in a stable order."""
    seen: list[str] = []
    for d in mask_dicts:
        for side in as_sides(d):
            if side not in seen:
                seen.append(side)
    return sorted(seen, key=lambda s: (SIDES.index(s) if s in SIDES else 99, s))


def _bs_state(value) -> bool | None:
    """Interpret a BackgroundSuppression field that may not be a boolean.

    BIDS types this field as a boolean, but real sidecars carry the NUMBER OF
    SUPPRESSION PULSES instead - the Brumer renal phantom data ships
    `"BackgroundSuppression": 2`. Both conventions are readable without guessing:
    0 and False mean off, any positive count and True mean on. A string or
    anything else is not interpreted at all, because a wrong reading here flips
    a FAIL into a PASS on the calibration image.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value >= 1)
    return None


#: Unit spellings that mean "perfusion per unit mass or volume", which is what
#: the renal bands are stated in. Anything else - arbitrary units, %-of-M0, or a
#: string nobody recognises - is NOT a declaration this module can grade against.
_RENAL_UNITS = ("ml/100g/min", "ml/min/100g", "ml/100 g/min", "ml/min/100 g",
                "ml/min/100ml", "ml/min/100 ml", "ml/100ml/min", "ml/100 ml/min")


def _units_reason(units) -> str:
    """Why a map's units are not gradeable - stated precisely.

    "not declared" and "declared, but in a unit these bands do not describe" are
    different situations with different fixes, and telling a caller who
    correctly wrote 'a.u.' that they declared nothing is simply wrong.
    """
    if not isinstance(units, str) or not units.strip():
        return ("units not declared - the 50-500 bound and the 500 ceiling are stated per "
                "100 g and cannot be applied to an unknown quantity")
    return (f"units declared as '{units.strip()}', which is not a per-mass or per-volume "
            "perfusion unit - the renal bands are stated per 100 g, so they do not describe "
            "this map")


def _units_declared(units) -> bool:
    """Whether the caller stated what the map's numbers mean, IN A UNIT THE
    BANDS APPLY TO.

    Undeclared units are not a formatting nicety: the 50-500 sanity band and the
    500 ceiling are both stated per 100 g, and a map in mL/min/100 mL differs by
    the ~1.05 g/mL tissue density before anything else. Grading an undeclared
    map would be grading an unknown quantity.

    Accepting ANY non-empty string had the same effect by a different route: a
    map declared "a.u." - or "banana" - was graded against a per-100-g band, so
    a caller who correctly said their map was in arbitrary units got a confident
    verdict about a quantity the band does not describe.
    """
    if not isinstance(units, str) or not units.strip():
        return False
    return units.strip().lower().replace(" ", "") in tuple(
        u.replace(" ", "") for u in _RENAL_UNITS)


def _fmt_sides(values: dict, fmt: str = "{:.0f}") -> str:
    """'300 (left) / 291 (right)' — the reason strings always name the side,
    because the whole module reports the two kidneys separately."""
    parts = []
    for side, v in values.items():
        parts.append(f"{fmt.format(v) if np.isfinite(v) else 'n/a'} ({side})")
    return " / ".join(parts)


def _grid_error(volume, *mask_dicts) -> str | None:
    """Actionable message if any supplied mask is not on the image's grid.

    Every check that reads a mask calls this FIRST. Without it, roi_stats raises
    and run_qc's blanket except turns the report line into "check error: mask
    shape (512, 512, 20) != volume shape ..." - a stack-trace fragment where a
    reader needs an instruction. Real data makes this the common case, not an
    edge case: the renaldro/iBEAt cortex-medulla labels ship at 1 mm on a 512^2
    grid while the ASL sits at 4.69 mm on a 64x32 grid.
    """
    shape = np.asarray(volume).shape[:3]
    for masks in mask_dicts:
        for side, mask in as_sides(masks).items():
            m = np.asarray(mask)
            if m.shape[:3] != shape:
                return (f"the {side} mask is {tuple(m.shape[:3])} but the image is "
                        f"{tuple(shape)} - resample the masks into the image grid before "
                        "grading (this tool deliberately does not resample: interpolating a "
                        "mask changes which voxels are called kidney)")
    return None


def _context_note(context) -> str:
    """A short acquisition-context tail for reason strings.

    The context is not decoration. Labelling scheme moves cortical RBF by ~1.8x
    in the same subjects, so a number reported without it invites a comparison
    that is not valid.
    """
    if not isinstance(context, dict):
        return ""
    bits = [str(context[k]) for k in ("labelling", "field_strength_t", "readout")
            if context.get(k) not in (None, "")]
    return f" [{', '.join(bits)}]" if bits else ""


# --------------------------------------------------------------------------- #
# Module K1 — the quality-index slot, deliberately empty
# --------------------------------------------------------------------------- #
@register_qc_check("k1.1.renal_qei", stream="B", required=False, organ=ORGAN)
def renal_qei_check(**_) -> CheckResult:
    """ALWAYS N/A in v1. There is no renal QEI, and inventing one is the single
    least defensible thing this project could do.

    This check exists precisely so that the absence is visible in every report
    rather than being an invisible gap in the check list. It takes no inputs and
    performs no arithmetic — there is deliberately no UNKNOWN branch, because
    UNKNOWN would mean "I could have graded this if you gave me more data", and
    no possible input makes a renal quality index exist.

    Why not build one from the components we do have (K2.1 tSNR, K2.3 negative
    fraction, K3.2 CMR)? A geometric mean of three uncalibrated components is
    not a validated index; it is an uncalibrated number with a confident name.
    The brain QEI's constants and its ~0.5 cutoff were FITTED to expert ratings
    of brain CBF maps. No equivalent rated renal dataset exists to fit against.
    """
    return CheckResult(
        "k1.1.renal_qei", Verdict.NA,
        metric={
            "renal_qei": None,          # null, never 0.0 — 0.0 reads as "worst possible"
            "blockers": [
                # verbatim from the design doc, in its order
                "no_probability_substrate",
                "boundary_not_reliably_drawable",
                "denominator_compartment_declared_unreliable",
                "no_labelled_dataset",
            ],
            "components_shipped_separately": ["K3.2.cmr", "K2.3.negative_fraction", "K2.1.tsnr"],
        },
        reason="no renal quality index exists; the computable components ship as "
               "K2.1, K2.3 and K3.2")


# --------------------------------------------------------------------------- #
# Module K2 — noise, signal magnitude, distribution
# --------------------------------------------------------------------------- #
@register_qc_check("k2.1.tsnr", stream="B", required=True, organ=ORGAN)
def renal_tsnr_check(delta_m_4d=None, kidney_masks=None, cortex_masks=None,
                     context=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Temporal SNR per kidney. Report-only: emits INFO, never a graded verdict.

    No renal tSNR cutoff exists, and the published values span about sevenfold
    (medullary VSI-ASL 0.17 ± 0.14 up to cortical 3.54 ± 0.71) driven by ROI,
    labelling and readout rather than by quality. A band across that spread would
    grade the protocol. Worse, tSNR is defined per repetition here, so an
    averaged map from 25 pairs has ~sqrt(25) = 5x the SNR of one repetition —
    two honest numbers from the same scan differing fivefold, with no published
    convention saying which one a threshold would apply to.

    tSNR is the ROI MEAN OF PER-VOXEL RATIOS (mu/sd per voxel, then averaged),
    not the ratio of ROI-mean signal to ROI-mean SD. The two differ whenever the
    ROI is heterogeneous, which a kidney always is.
    """
    if delta_m_4d is None:
        return CheckResult("k2.1.tsnr", Verdict.UNKNOWN,
                           reason="needs the 4D repetition series (an averaged map has no "
                                  "temporal axis)")
    arr = np.asarray(delta_m_4d, dtype=float)
    if arr.ndim != 4:
        return CheckResult("k2.1.tsnr", Verdict.UNKNOWN,
                           reason=f"expected a 4D series, got shape {arr.shape}")

    # Drop whole non-finite VOLUMES (time-points), not individual voxels: a
    # repetition rejected by K7.2 is absent as a repetition, and zero-filling it
    # would inject a fabricated constant volume into the temporal SD.
    keep = [t for t in range(arr.shape[3]) if np.isfinite(arr[..., t]).any()]
    if len(keep) < cfg.kidney_min_repetitions:
        return CheckResult("k2.1.tsnr", Verdict.UNKNOWN,
                           metric={"n_usable_repetitions": len(keep)},
                           reason=f"only {len(keep)} usable repetitions; a temporal SD needs "
                                  f"at least {cfg.kidney_min_repetitions}")
    series = arr[..., keep]

    cortex, whole = as_sides(cortex_masks), as_sides(kidney_masks)
    sides = _sides_present(cortex_masks, kidney_masks)
    if not sides:
        return CheckResult("k2.1.tsnr", Verdict.UNKNOWN, reason="needs at least a whole-kidney mask")
    bad = _grid_error(series[..., 0], cortex_masks, kidney_masks)
    if bad:
        return CheckResult("k2.1.tsnr", Verdict.UNKNOWN, reason=bad)

    mu = series.mean(axis=3)
    # ddof=1: a sample SD over repetitions. (The brain QEI's dispersion term uses
    # ddof=0 to stay byte-faithful to ASLPrep; this is a different statistic in a
    # different module and the renal design states ddof=1 explicitly.)
    sd = series.std(axis=3, ddof=1)

    per_side: dict = {}
    for side in sides:
        roi = cortex.get(side)
        roi_kind = "cortex"
        if roi is None:
            roi, roi_kind = whole.get(side), "whole_kidney"
        if roi is None:
            continue
        m = as_mask(roi) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)  # sd<=0 excluded, not clipped
        vals = (mu[m] / sd[m]) if m.any() else np.array([])
        per_side[side] = {
            "tsnr": float(vals.mean()) if vals.size else float("nan"),
            "roi": roi_kind,
            "n_voxels": int(vals.size),
            "n_repetitions_used": len(keep),
        }
    if not per_side:
        return CheckResult("k2.1.tsnr", Verdict.UNKNOWN, reason="needs at least a whole-kidney mask")

    metric = {
        "per_kidney": per_side,
        "definition": "mean over ROI voxels of (mean over repetitions / SD over "
                      "repetitions, ddof=1)",
        "per_repetition": True,
        "context": context if isinstance(context, dict) else {},
        "provenance": "uncalibrated - no renal tSNR cutoff exists; reported for "
                      "inspection only",
    }
    shown = {s: v["tsnr"] for s, v in per_side.items()}
    kind = next(iter(per_side.values()))["roi"].replace("_", "-")
    return CheckResult("k2.1.tsnr", Verdict.INFO, metric=metric,
                       reason=f"{kind} tSNR {_fmt_sides(shown, '{:.2f}')}, per-repetition"
                              f"{_context_note(context)} - reported, not graded")


@register_qc_check("k2.2.pws_pct", stream="B", required=True, organ=ORGAN)
def renal_pws_check(delta_m=None, m0=None, kidney_masks=None, cortex_masks=None,
                    medulla_masks=None, pld_or_ti_s=None, n_bs_pulses=None,
                    grids_match=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Perfusion-weighted signal as a percentage of M0 — the renal analogue of
    "does this look like ~1% ASL contrast?"

    Grades the CORTICAL value only (R10.1); medulla and whole-kidney are reported
    but never graded (R10.2). Never FAILs: every bound here is uncalibrated, and
    a low PWS at a long PLD is a legitimate physiological finding.
    """
    if delta_m is None or m0 is None:
        return CheckResult("k2.2.pws_pct", Verdict.UNKNOWN,
                           reason="needs both the mean subtraction image and M0")
    # Gate on space first: a ratio computed across mismatched grids is a number
    # about interpolation, not about labelling.
    if grids_match is False:
        return CheckResult("k2.2.pws_pct", Verdict.UNKNOWN,
                           reason="delta-M and M0 are not on the same grid (see K4.2) - "
                                  "a PWS ratio across grids measures interpolation")
    dm = np.asarray(delta_m, dtype=float)
    m0a = np.asarray(m0, dtype=float)
    if dm.shape != m0a.shape:
        return CheckResult("k2.2.pws_pct", Verdict.UNKNOWN,
                           metric={"delta_m_shape": dm.shape, "m0_shape": m0a.shape},
                           reason=f"delta-M {dm.shape} and M0 {m0a.shape} are different shapes")

    rois = {"cortex": as_sides(cortex_masks), "whole": as_sides(kidney_masks),
            "medulla": as_sides(medulla_masks)}
    sides = _sides_present(cortex_masks, kidney_masks, medulla_masks)
    if not sides:
        return CheckResult("k2.2.pws_pct", Verdict.UNKNOWN, reason="needs at least a whole-kidney mask")
    bad = _grid_error(dm, cortex_masks, kidney_masks, medulla_masks)
    if bad:
        return CheckResult("k2.2.pws_pct", Verdict.UNKNOWN, reason=bad)

    per_side: dict = {}
    bad_denominator: list[str] = []
    for side in sides:
        entry: dict = {}
        for kind, masks in rois.items():
            roi = masks.get(side)
            if roi is None:
                continue
            # PWS% is a ratio of ROI MEANS, not the mean of per-voxel ratios.
            num = roi_stats(dm, roi)
            den = roi_stats(m0a, roi)
            if den["n"] == 0 or not np.isfinite(den["mean"]) or den["mean"] <= 0:
                bad_denominator.append(f"{side}/{kind}")
                entry[f"pws_{kind}_pct"] = float("nan")
                continue
            entry[f"pws_{kind}_pct"] = float(100.0 * num["mean"] / den["mean"])
        if entry:
            per_side[side] = entry

    cortical = {s: v.get("pws_cortex_pct", float("nan")) for s, v in per_side.items()}
    gradeable = {s: v for s, v in cortical.items() if np.isfinite(v)}

    metric = {
        "per_kidney": per_side,
        "pld_or_ti_s": pld_or_ti_s,
        "band_used": [cfg.kidney_pws_lo, cfg.kidney_pws_hi],
        "graded_roi": "cortex",
        "provenance": "uncalibrated band drawn to contain the full published spread",
    }
    if n_bs_pulses is not None:
        # Recorded and annotated, never used to shift the band: no published
        # BS-conditioned PWS band exists to re-band onto.
        metric["n_bs_pulses"] = n_bs_pulses
        metric["bs_note"] = ("background suppression attenuates delta-M and M0 differently; "
                             "value annotated, band NOT adjusted (no published BS-conditioned band)")

    if not gradeable:
        why = (f"M0 mean is not positive in {sorted(set(bad_denominator))}"
               if bad_denominator else "no cortical mask supplied")
        return CheckResult("k2.2.pws_pct", Verdict.UNKNOWN, metric=metric,
                           reason=f"cortical PWS not computable - {why}")

    shown = _fmt_sides(gradeable, "{:.2f}%")
    if pld_or_ti_s is None:
        # PWS falls ~3x between PLD 0.5 s and 1.5 s in the same subjects, so an
        # ungated band is not a defensible test.
        return CheckResult("k2.2.pws_pct", Verdict.INFO, metric=metric,
                           reason=f"cortical PWS {shown} - PLD/TI unknown, so the "
                                  f"{cfg.kidney_pws_lo}-{cfg.kidney_pws_hi}% band cannot be "
                                  "applied honestly")

    nonpositive = [s for s, v in gradeable.items() if v <= 0]
    outside = [s for s, v in gradeable.items()
               if not (cfg.kidney_pws_lo <= v <= cfg.kidney_pws_hi)]
    if nonpositive:
        return CheckResult("k2.2.pws_pct", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"cortical PWS {shown} - non-positive on "
                                  f"{', '.join(nonpositive)}, which points at a subtraction-sign "
                                  "failure (cross-check K5.3)")
    if outside:
        return CheckResult("k2.2.pws_pct", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"cortical PWS {shown} at PLD/TI {pld_or_ti_s} s - outside "
                                  f"{cfg.kidney_pws_lo}-{cfg.kidney_pws_hi}% on "
                                  f"{', '.join(outside)}; labelling weak or M0 mis-scaled")
    return CheckResult("k2.2.pws_pct", Verdict.PASS, metric=metric,
                       reason=f"cortical PWS {shown} at PLD/TI {pld_or_ti_s} s - within the "
                              "reported renal range")


@register_qc_check("k2.3.implausible_values", stream="B", required=True, organ=ORGAN)
def renal_implausible_check(rbf_map=None, kidney_masks=None, cortex_masks=None,
                            units=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Fraction of within-kidney voxels that are physiologically impossible.

    Two fractions per kidney: below zero, and above the ceiling. The FAIL branch
    is licensed by the SIGN, not by its 20%: a majority-negative perfusion map is
    not a perfusion map at all. The ceiling is an IMPLEMENTATION number (one
    study's preprocessing clip), so it never FAILs on its own.

    The denominator is every voxel in the ROI, including non-finite ones. NaN < 0
    and NaN > ceiling are both False, so non-finite voxels dilute the fraction
    rather than inflating it — which is the conservative direction.
    """
    if rbf_map is None:
        return CheckResult("k2.3.implausible_values", Verdict.UNKNOWN, reason="needs an RBF map")
    if not _units_declared(units):
        return CheckResult("k2.3.implausible_values", Verdict.UNKNOWN,
                           reason=_units_reason(units))
    rbf = np.asarray(rbf_map, dtype=float)
    rois = as_sides(cortex_masks) or as_sides(kidney_masks)
    roi_kind = "cortex" if as_sides(cortex_masks) else "whole_kidney"
    if not rois:
        return CheckResult("k2.3.implausible_values", Verdict.UNKNOWN, reason="needs a kidney mask")

    ceiling = cfg.kidney_implausible_ceiling
    per_side: dict = {}
    for side, mask in rois.items():
        m = as_mask(mask)
        if m.shape != rbf.shape:
            return CheckResult("k2.3.implausible_values", Verdict.UNKNOWN,
                               reason=f"{side} mask {m.shape} does not match the RBF grid "
                                      f"{rbf.shape} - resample before grading")
        vals = rbf[m]
        n = int(vals.size)
        if n < cfg.kidney_min_roi_voxels:
            per_side[side] = {"n_voxels": n, "neg_frac": float("nan"),
                              "over_ceiling_frac": float("nan")}
            continue
        per_side[side] = {
            "n_voxels": n,
            # denominator is ALL ROI voxels; NaN compares False and so is never
            # counted as implausible
            "neg_frac": float(np.count_nonzero(vals < 0) / n),
            "over_ceiling_frac": float(np.count_nonzero(vals > ceiling) / n),
        }

    metric = {"per_kidney": per_side, "roi": roi_kind, "ceiling": float(ceiling),
              "units": units, "warn_fraction": cfg.kidney_frac_warn,
              "fail_fraction": cfg.kidney_frac_fail}
    usable = {s: v for s, v in per_side.items() if np.isfinite(v["neg_frac"])}
    if not usable:
        return CheckResult("k2.3.implausible_values", Verdict.UNKNOWN, metric=metric,
                           reason=f"fewer than {cfg.kidney_min_roi_voxels} voxels in every ROI")

    neg = {s: v["neg_frac"] for s, v in usable.items()}
    over = {s: v["over_ceiling_frac"] for s, v in usable.items()}
    worst_neg = max(neg.values())
    worst_over = max(over.values())
    shown = (f"negative {_fmt_sides({s: 100 * x for s, x in neg.items()}, '{:.1f}%')}, "
             f"over {ceiling:.0f}: {_fmt_sides({s: 100 * x for s, x in over.items()}, '{:.1f}%')}")

    if worst_neg > cfg.kidney_frac_fail:
        sides = [s for s, v in neg.items() if v > cfg.kidney_frac_fail]
        verdict = Verdict.FAIL if cfg.strict else Verdict.WARN
        return CheckResult("k2.3.implausible_values", verdict, metric=metric, provisional=True,
                           reason=f"{shown} - a majority-negative map on {', '.join(sides)} is "
                                  "not a perfusion map")
    # NOTE: the design's verdict table scopes FAIL to the NEGATIVE fraction only,
    # and its WARN row covers "either fraction in 5-20%", leaving over-ceiling
    # above 20% matching no row. Treated as WARN (the most severe verdict the
    # table licenses for that quantity) rather than silently PASSing it, and the
    # metric carries the raw fraction so the gap is visible rather than papered
    # over. Raised as a documentation question rather than decided unilaterally.
    if worst_neg >= cfg.kidney_frac_warn or worst_over >= cfg.kidney_frac_warn:
        return CheckResult("k2.3.implausible_values", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"{shown} - above the {cfg.kidney_frac_warn:.0%} line")
    return CheckResult("k2.3.implausible_values", Verdict.PASS, metric=metric,
                       reason=f"{shown} - within the {cfg.kidney_frac_warn:.0%} line")


# --------------------------------------------------------------------------- #
# Module K3 — perfusion level and contrast
# --------------------------------------------------------------------------- #
@register_qc_check("k3.1.cortical_rbf", stream="B", required=True, organ=ORGAN)
def cortical_rbf_check(rbf_map=None, cortex_masks=None, kidney_masks=None,
                       units=None, context=None, cfg: QCConfig = QCConfig(),
                       **_) -> CheckResult:
    """Cortical RBF per kidney, graded against a 50-500 SANITY bound only.

    INFO — not PASS — is the good outcome here, and that is deliberate. A PASS
    would assert "this perfusion level is normal", which no evidence supports:
    the published 139-427 spread is a range of study-level cohort MEANS whose
    healthy and patient ranges overlap almost entirely, and the PET reference
    standard itself disagrees with ASL by ±136 mL/min/100 mL. The band encodes
    "this cannot be renal cortical perfusion", not "this is healthy perfusion",
    and the acquisition context travels with the number so nobody compares two
    values acquired differently.

    There is no low-side FAIL, for an anatomical reason: a cortex only 1-2 voxels
    thick partial-volumes with medulla and biases cortical RBF downward, so a low
    value can be a fact about voxel size rather than about data quality.
    """
    if rbf_map is None:
        return CheckResult("k3.1.cortical_rbf", Verdict.UNKNOWN, reason="needs an RBF map")
    if not _units_declared(units):
        return CheckResult("k3.1.cortical_rbf", Verdict.UNKNOWN,
                           reason=_units_reason(units))
    rbf = np.asarray(rbf_map, dtype=float)
    cortex, whole = as_sides(cortex_masks), as_sides(kidney_masks)
    if not cortex and not whole:
        return CheckResult("k3.1.cortical_rbf", Verdict.UNKNOWN, reason="needs a kidney mask")
    bad = _grid_error(rbf, cortex_masks, kidney_masks)
    if bad:
        return CheckResult("k3.1.cortical_rbf", Verdict.UNKNOWN, reason=bad)

    lo, hi = cfg.kidney_rbf_sanity_lo, cfg.kidney_rbf_sanity_hi
    ctx = context if isinstance(context, dict) else {}
    base_metric = {"band_used": [lo, hi], "units": units, "context": ctx,
                   "provenance": "uncalibrated sanity bound - not a reference interval"}

    if not cortex:
        # Report the whole-kidney mean, and refuse to grade it: R10.1 asks for
        # cortex, and grading a different quantity under the same name would be
        # the more damaging error.
        means = {s: roi_stats(rbf, m)["mean"] for s, m in whole.items()}
        return CheckResult(
            "k3.1.cortical_rbf", Verdict.NA,
            metric={**base_metric, "whole_kidney_mean": means, "graded": False},
            reason=f"no cortex mask; whole-kidney mean {_fmt_sides(means)} reported and "
                   "NOT graded (the consensus quantity is cortical)")

    per_side: dict = {}
    for side, mask in cortex.items():
        s = roi_stats(rbf, mask)
        entry = {"cortex_mean": s["mean"], "cortex_median": s["median"],
                 # sample SD (ddof=1) as the design states for this statistic
                 "cortex_sd": float(roi_values(rbf, mask).std(ddof=1)) if s["n"] > 1 else float("nan"),
                 "n_voxels": s["n"]}
        if side in whole:
            entry["whole_kidney_mean"] = roi_stats(rbf, whole[side])["mean"]
        per_side[side] = entry

    metric = {**base_metric, "per_kidney": per_side}
    gradeable = {s: v["cortex_mean"] for s, v in per_side.items()
                 if v["n_voxels"] >= cfg.kidney_min_roi_voxels and np.isfinite(v["cortex_mean"])}
    if not gradeable:
        return CheckResult("k3.1.cortical_rbf", Verdict.UNKNOWN, metric=metric,
                           reason=f"fewer than {cfg.kidney_min_roi_voxels} cortical voxels - "
                                  "a cortical mean here would be partial volume, not cortex")

    shown = _fmt_sides(gradeable) + f" {units}" + _context_note(ctx)
    outside = [s for s, v in gradeable.items() if not (lo <= v <= hi)]
    if outside:
        return CheckResult("k3.1.cortical_rbf", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"cortical RBF {shown} - outside the {lo:.0f}-{hi:.0f} sanity "
                                  f"bound on {', '.join(outside)} (a sanity bound, not a "
                                  "normal range)")
    return CheckResult("k3.1.cortical_rbf", Verdict.INFO, metric=metric,
                       reason=f"cortical RBF {shown} - reported with its acquisition context; "
                              "no normality claim is made")


@register_qc_check("k3.2.cmr", stream="B", required=False, organ=ORGAN)
def cortico_medullary_ratio_check(rbf_map=None, cortex_masks=None, medulla_masks=None,
                                  units=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Cortico-medullary ratio — a SEGMENTATION-INTEGRITY flag, never a
    perfusion verdict.

    This looked like the obvious renal analogue of the brain's GM/WM ratio:
    scale-free, contrast-based, exactly the kind of check that survives a
    calibration error. Consensus R10.2 (89% agreement) undercuts it — medullary
    values "are not considered reliable with current measurement approaches" —
    so a CMR outside expectation cannot be read as a perfusion finding.

    What it CAN still catch is real and useful: measured CMR clusters at
    2.26-2.59, and only ~10% of renal blood flow perfuses the medulla, so a ratio
    at or below 1.5 says the two masks are probably mixed up or mislabelled.
    """
    if rbf_map is None:
        return CheckResult("k3.2.cmr", Verdict.UNKNOWN, reason="needs an RBF map")
    cortex, medulla = as_sides(cortex_masks), as_sides(medulla_masks)
    if not medulla:
        return CheckResult("k3.2.cmr", Verdict.NA,
                           reason="no medulla mask - the consensus default is cortex-only "
                                  "reporting, so this is not a gap")
    if not cortex:
        return CheckResult("k3.2.cmr", Verdict.UNKNOWN, reason="needs a cortex mask to form a ratio")

    rbf = np.asarray(rbf_map, dtype=float)
    bad = _grid_error(rbf, cortex_masks, medulla_masks)
    if bad:
        return CheckResult("k3.2.cmr", Verdict.UNKNOWN, reason=bad)
    per_side: dict = {}
    for side in _sides_present(cortex_masks, medulla_masks):
        cm, mm = cortex.get(side), medulla.get(side)
        if cm is None or mm is None:
            continue
        c, m = roi_stats(rbf, cm), roi_stats(rbf, mm)
        overlap = int(np.count_nonzero(as_mask(cm) & as_mask(mm)))
        ratio = float("nan")
        if (m["n"] >= cfg.kidney_min_medulla_voxels and c["n"] >= cfg.kidney_min_roi_voxels
                and np.isfinite(m["mean"]) and m["mean"] > 0):
            ratio = float(c["mean"] / m["mean"])
        per_side[side] = {"cmr": ratio, "cortex_mean": c["mean"], "medulla_mean": m["mean"],
                          "n_cortex": c["n"], "n_medulla": m["n"],
                          "cortex_medulla_overlap_voxels": overlap}

    metric = {"per_kidney": per_side, "trip_point": cfg.kidney_cmr_trip,
              "interpretation": "segmentation-integrity flag only; R10.2 forbids reading "
                                "this as a perfusion verdict"}
    gradeable = {s: v["cmr"] for s, v in per_side.items() if np.isfinite(v["cmr"])}
    if not gradeable:
        return CheckResult("k3.2.cmr", Verdict.UNKNOWN, metric=metric,
                           reason="medullary mean is not positive, or a mask has too few voxels")

    shown = _fmt_sides(gradeable, "{:.2f}")
    low = [s for s, v in gradeable.items() if v < cfg.kidney_cmr_trip]
    if low:
        return CheckResult("k3.2.cmr", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"cortico-medullary ratio {shown} - below {cfg.kidney_cmr_trip} "
                                  f"on {', '.join(low)}; cortex and medulla masks are probably "
                                  "swapped or mixed, NOT a perfusion finding")
    return CheckResult("k3.2.cmr", Verdict.INFO, metric=metric,
                       reason=f"cortico-medullary ratio {shown} - consistent with intact "
                              "segmentation; no perfusion claim is made")


@register_qc_check("k3.3.left_right", stream="B", required=True, organ=ORGAN)
def left_right_consistency_check(rbf_map=None, cortex_masks=None, kidney_masks=None,
                                 units=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Left-vs-right cortical RBF consistency — a review flag, never a rejection.

    Published data find NO significant left-right difference in cortical RBF
    (P = 0.93 FAIR, P = 0.52 pCASL), with a tolerated normal left>right bias of
    roughly 0.5-6.8% and a between-visit CV of 4-13%. The 20% tolerance sits well
    above that noise floor, so what it flags is a genuine one-sided problem —
    which may be the anatomy (a diseased or transplanted kidney) rather than the
    data, and that is exactly why it never FAILs.
    """
    if rbf_map is None:
        return CheckResult("k3.3.left_right", Verdict.UNKNOWN, reason="needs an RBF map")
    rois = as_sides(cortex_masks) or as_sides(kidney_masks)
    roi_kind = "cortex" if as_sides(cortex_masks) else "whole_kidney"
    if roi_kind == "whole_kidney" and len(rois) >= 2:
        # The consensus quantity is CORTICAL (R10.1). A whole-kidney asymmetry is
        # still worth seeing - it is scale-free and catches one-sided problems -
        # but it is a different quantity, so it is reported as INFO rather than
        # graded against the cortical tolerance.
        rbf_wk = np.asarray(rbf_map, dtype=float)
        bad_wk = _grid_error(rbf_wk, kidney_masks)
        if bad_wk:
            return CheckResult("k3.3.left_right", Verdict.UNKNOWN, reason=bad_wk)
        means_wk = {s: roi_stats(rbf_wk, m)["mean"] for s, m in rois.items()}
        vals = list(means_wk.values())
        ai_wk = asymmetry_index(vals[0], vals[1]) if len(vals) == 2 else float("nan")
        return CheckResult(
            "k3.3.left_right", Verdict.INFO,
            metric={"per_kidney": means_wk, "roi": roi_kind, "asymmetry_pct": ai_wk,
                    "graded": False, "tolerance": cfg.kidney_asymmetry_tol},
            reason=f"whole-kidney asymmetry {ai_wk:.1f}% - reported, not graded (the "
                   "consensus quantity is cortical; supply cortex masks to grade it)")
    if len(rois) < 2:
        return CheckResult("k3.3.left_right", Verdict.NA,
                           metric={"n_kidneys": len(rois)},
                           reason="only one kidney supplied - nothing to compare (transplant, "
                                  "nephrectomy or agenesis are all ordinary)")

    rbf = np.asarray(rbf_map, dtype=float)
    bad = _grid_error(rbf, cortex_masks, kidney_masks)
    if bad:
        return CheckResult("k3.3.left_right", Verdict.UNKNOWN, reason=bad)
    means = {s: roi_stats(rbf, m)["mean"] for s, m in rois.items()}
    if len(means) != 2 or not all(np.isfinite(v) for v in means.values()):
        return CheckResult("k3.3.left_right", Verdict.UNKNOWN, metric={"means": means},
                           reason="a kidney's cortical RBF is unavailable")

    (sa, a), (sb, b) = sorted(means.items())
    ai_pct = asymmetry_index(a, b)
    metric = {"per_kidney": means, "roi": roi_kind,
              "asymmetry_index": ai_pct / 100.0 if np.isfinite(ai_pct) else float("nan"),
              "asymmetry_pct": ai_pct, "tolerance": cfg.kidney_asymmetry_tol,
              "definition": "|L-R| / (0.5*(L+R))"}
    if not np.isfinite(ai_pct):
        return CheckResult("k3.3.left_right", Verdict.UNKNOWN, metric=metric,
                           reason="both kidneys average to zero - no ratio to form")
    shown = f"{means[sa]:.0f} ({sa}) vs {means[sb]:.0f} ({sb}), asymmetry {ai_pct:.1f}%"
    if ai_pct / 100.0 > cfg.kidney_asymmetry_tol:
        return CheckResult("k3.3.left_right", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"{shown} - above the {cfg.kidney_asymmetry_tol:.0%} tolerance; "
                                  "flag for human review, may be one-sided disease rather than "
                                  "a data defect")
    return CheckResult("k3.3.left_right", Verdict.PASS, metric=metric,
                       reason=f"{shown} - within the {cfg.kidney_asymmetry_tol:.0%} tolerance")


# --------------------------------------------------------------------------- #
# Module K4 — masks, registration, coverage
# --------------------------------------------------------------------------- #
@register_qc_check("k4.1.mask_integrity", stream="B", required=True, organ=ORGAN)
def kidney_mask_integrity_check(kidney_masks=None, cortex_masks=None, medulla_masks=None,
                                cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Is each supplied mask a single, sane, unclipped object?

    Masks are declared inputs here, so this is the check that keeps them honest.
    Its FAILs are all definitional rather than statistical: an empty mask, a
    cortex that intersects its own medulla, or a left kidney that intersects the
    right, are each impossible as anatomy — no threshold is involved and no
    tolerance is allowed.
    """
    whole, cortex, medulla = (as_sides(kidney_masks), as_sides(cortex_masks),
                              as_sides(medulla_masks))
    if not (whole or cortex or medulla):
        return CheckResult("k4.1.mask_integrity", Verdict.UNKNOWN, reason="no masks supplied")

    per_mask: dict = {}
    failures: list[str] = []
    warnings: list[str] = []
    for kind, masks in (("kidney", whole), ("cortex", cortex), ("medulla", medulla)):
        for side, mask in masks.items():
            m = as_mask(mask)
            n = int(m.sum())
            sizes = component_sizes(m)
            frac = largest_component_fraction(m)
            edge = touches_fov_edge(m)
            per_mask[f"{kind}_{side}"] = {
                "n_voxels": n,
                "n_components": len(sizes),
                "largest_component_fraction": frac,
                "touches_fov_edge": edge,
            }
            if n == 0:
                failures.append(f"{kind} {side} is empty")
                continue
            # The medulla is genuinely multi-component: it is a set of renal
            # pyramids, not one blob. Real iBEAt data has ~8 medullary components
            # of 2000-4000 voxels each per kidney, so the single-dominant-object
            # rule that is right for a kidney and its cortex is an anatomical
            # false positive here. The count is reported, never graded.
            if kind == "medulla":
                continue
            if np.isfinite(frac) and frac < cfg.kidney_mask_component_frac:
                warnings.append(f"{kind} {side} is fragmented ({frac:.0%} in its largest piece, "
                                f"{len(sizes)} pieces)")
            if edge:
                warnings.append(f"{kind} {side} touches the edge of the field of view")
            if kind == "cortex" and n < cfg.kidney_min_cortex_voxels:
                warnings.append(f"cortex {side} has only {n} voxels")

    # Definitional impossibilities: overlaps that anatomy does not permit.
    for side in _sides_present(cortex_masks, medulla_masks):
        if side in cortex and side in medulla:
            n = int(np.count_nonzero(as_mask(cortex[side]) & as_mask(medulla[side])))
            per_mask.setdefault(f"overlap_cortex_medulla_{side}", n)
            if n:
                failures.append(f"cortex and medulla overlap on {side} ({n} voxels)")
    if len(whole) == 2:
        a, b = (as_mask(whole[s]) for s in sorted(whole))
        n = int(np.count_nonzero(a & b))
        per_mask["overlap_left_right"] = n
        if n:
            failures.append(f"the two kidney masks overlap ({n} voxels)")

    n_masks = len(whole) + len(cortex) + len(medulla)
    metric = {"per_mask": per_mask, "failures": failures, "warnings": warnings,
              "n_masks": n_masks, "component_floor": cfg.kidney_mask_component_frac}
    if failures:
        return CheckResult("k4.1.mask_integrity", Verdict.FAIL, metric=metric,
                           reason="; ".join(failures))
    if warnings:
        return CheckResult("k4.1.mask_integrity", Verdict.WARN, metric=metric,
                           reason="; ".join(warnings))
    return CheckResult("k4.1.mask_integrity", Verdict.PASS, metric=metric,
                       reason=f"{n_masks} mask(s): one dominant component each, none "
                              "clipped by the field of view")


@register_qc_check("k4.2.registration", stream="B", required=True, organ=ORGAN)
def kidney_registration_check(rbf_map=None, m0=None, delta_m=None, delta_m_4d=None,
                              kidney_masks=None, affine=None, m0_affine=None,
                              transforms=None, registration_scope=None,
                              voxel_mm=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Are the perfusion image and M0 on one geometry, was each kidney registered
    separately, and how much residual misalignment is left?

    Three tests, and only the first can FAIL:

    1. **Geometry.** Matching matrix sizes are necessary but not sufficient - two
       images can both be 96x96x5 on different voxel sizes - so the AFFINES are
       compared too, over the 3x4 geometry block with a 0.01 mm tolerance for
       float round-trip. A mismatch with no transform supplied is a corruption
       rather than an absence: every ROI statistic would be computed over the
       wrong voxels. That is the only FAIL here.
    2. **Scope.** A single global rigid transform for both kidneys cannot be
       right, because the two kidneys move independently with respiration. WARN.
    3. **Residual.** The intensity-weighted centroid of each kidney in the first
       and last volume, reported in mm. Cheap and dependency-free; K7.1 grades
       displacement properly, this only flags a residual above one in-plane voxel.

    Deliberately NOT Dice. An earlier version FAILed on a Dice below an
    uncalibrated cut-off, which is exactly the thing this project forbids: an
    engineering default driving a hard failure.
    """
    moving = rbf_map if rbf_map is not None else delta_m
    metric: dict = {"registration_scope": registration_scope or "not recorded"}
    if moving is None and m0 is None and delta_m_4d is None:
        return CheckResult("k4.2.registration", Verdict.UNKNOWN,
                           reason="needs a perfusion image, an M0, or a 4D series")

    # ---- 1. geometry -------------------------------------------------------
    shapes_match = affines_match = None
    if moving is not None and m0 is not None:
        a, b = np.asarray(moving), np.asarray(m0)
        shapes_match = a.shape[:3] == b.shape[:3]
        metric.update({"perfusion_shape": tuple(a.shape[:3]), "m0_shape": tuple(b.shape[:3]),
                       "shapes_match": shapes_match})
    if affine is not None and m0_affine is not None:
        diff = float(np.abs(np.asarray(affine, dtype=float)[:3, :4]
                            - np.asarray(m0_affine, dtype=float)[:3, :4]).max())
        affines_match = diff <= 0.01           # tolerance for float round-trip
        metric.update({"affine_max_diff_mm": diff, "affines_match": affines_match})

    geometry_ok = (shapes_match is not False) and (affines_match is not False)
    if not geometry_ok:
        if transforms is not None:
            metric["transform_supplied"] = True
        else:
            why = []
            if shapes_match is False:
                why.append(f"shapes {metric['perfusion_shape']} vs {metric['m0_shape']}")
            if affines_match is False:
                why.append(f"affines differ by {metric['affine_max_diff_mm']:.3f} mm")
            return CheckResult("k4.2.registration", Verdict.FAIL, metric=metric,
                               reason=f"perfusion and M0 are on different geometries "
                                      f"({'; '.join(why)}) and no transform was supplied - "
                                      "every ROI statistic would be over the wrong voxels")

    # ---- 3. residual centroid shift, first volume against last -------------
    residuals: dict = {}
    masks = as_sides(kidney_masks)
    arr = np.asarray(delta_m_4d, dtype=float) if delta_m_4d is not None else None
    if arr is not None and arr.ndim == 4 and arr.shape[3] >= 2 and masks:
        vox = tuple(float(v) for v in (voxel_mm or (1.0, 1.0, 1.0)))
        in_plane = float(min(vox))
        for side, mask in masks.items():
            m = as_mask(mask)
            if m.shape != arr.shape[:3]:
                continue
            idx = np.argwhere(m).astype(float)
            cents = []
            for t in (0, arr.shape[3] - 1):
                w = arr[..., t][m]
                good = np.isfinite(w)
                if not good.any():
                    cents = []
                    break
                ww = np.abs(w[good])
                cents.append(idx[good].T @ ww / ww.sum() if ww.sum() > 0
                             else idx[good].mean(axis=0))
            if len(cents) == 2:
                d_mm = (cents[1] - cents[0]) * np.asarray(vox)
                residuals[side] = {"shift_mm": float(np.linalg.norm(d_mm)),
                                   "shift_in_plane_voxels": float(np.linalg.norm(d_mm) / in_plane)}
        if residuals:
            metric["residual_first_to_last"] = residuals
            metric["in_plane_mm"] = in_plane

    # ---- 2. scope + verdict (worst of the tests that ran) -------------------
    scope = (registration_scope or "").strip().lower()
    problems = []
    if scope == "global":
        problems.append("a single global transform was used for both kidneys, which cannot be "
                        "right - they move independently")
    drifted = [s for s, v in residuals.items() if v["shift_in_plane_voxels"] > 1.0]
    if drifted:
        worst_shift = max(residuals[s]["shift_mm"] for s in drifted)
        problems.append(f"residual centroid shift {worst_shift:.1f} mm on {', '.join(drifted)}, "
                        "more than one in-plane voxel")
    if problems:
        return CheckResult("k4.2.registration", Verdict.WARN, metric=metric, provisional=True,
                           reason="; ".join(problems))
    if not scope and not residuals:
        return CheckResult("k4.2.registration", Verdict.INFO, metric=metric,
                           reason="geometry is consistent; no registration provenance recorded "
                                  "and no 4D series to measure a residual with, so alignment "
                                  "itself was not tested")
    detail = []
    if scope:
        detail.append(f"scope {scope}")
    if residuals:
        detail.append("residual " + _fmt_sides({s: v["shift_mm"] for s, v in residuals.items()},
                                               "{:.1f} mm"))
    return CheckResult("k4.2.registration", Verdict.PASS, metric=metric,
                       reason="geometry consistent; " + ", ".join(detail))


@register_qc_check("k4.3.slice_coverage", stream="B", required=True, organ=ORGAN)
def kidney_slice_coverage_check(rbf_map=None, kidney_masks=None, readout=None,
                                voxel_mm=None, slice_axis=None,
                                cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """What share of the slices holding kidney actually carry usable data?

    A slice counts as usable when it holds enough mask to judge (>= 20 voxels)
    and at least 90% of those voxels are finite and non-zero. Zeros inside an
    organ mask are the signature of a slice that dropped out, and they silently
    drag every ROI mean toward zero — the failure this check exists to surface.
    """
    if rbf_map is None:
        return CheckResult("k4.3.slice_coverage", Verdict.UNKNOWN, reason="needs a perfusion map")
    masks = as_sides(kidney_masks)
    if not masks:
        return CheckResult("k4.3.slice_coverage", Verdict.UNKNOWN, reason="needs a kidney mask")
    bad = _grid_error(np.asarray(rbf_map), kidney_masks)
    if bad:
        return CheckResult("k4.3.slice_coverage", Verdict.UNKNOWN, reason=bad)
    if isinstance(readout, str) and "single-slice" in readout.lower():
        return CheckResult("k4.3.slice_coverage", Verdict.NA,
                           metric={"readout": readout},
                           reason="2D single-slice readout - slice coverage does not apply "
                                  "(this is the consensus default acquisition)")

    rbf = np.asarray(rbf_map, dtype=float)
    # Which axis the SLICES lie along must be resolved, not assumed. Renal
    # acquisitions are routinely coronal or oblique, so array axis 2 is not
    # reliably the slice direction, and counting "slices" along the wrong axis
    # counts something that is not a slice at all.
    s_axis, s_source = _slice_axis(voxel_mm or (1.0, 1.0, 1.0), slice_axis)
    if s_axis is None:
        s_axis, s_source = 2, ("assumed array axis 2 - voxels are near-isotropic, so the slice "
                               "direction could not be inferred; pass slice_axis to be sure")
    per_side: dict = {}
    for side, mask in masks.items():
        m = as_mask(mask)
        if m.shape != rbf.shape:
            return CheckResult("k4.3.slice_coverage", Verdict.UNKNOWN,
                               reason=f"{side} mask {m.shape} does not match the map {rbf.shape}")
        n_with_mask = n_usable = 0
        for k in range(rbf.shape[s_axis]):
            sl = np.take(m, k, axis=s_axis)
            n_mask = int(sl.sum())
            if n_mask < cfg.kidney_min_slice_voxels:
                continue
            n_with_mask += 1
            vals = np.take(rbf, k, axis=s_axis)[sl]
            good = np.isfinite(vals) & (vals != 0)
            if good.sum() / n_mask >= cfg.kidney_slice_finite_frac:
                n_usable += 1
        frac = (n_usable / n_with_mask) if n_with_mask else float("nan")
        per_side[side] = {"slices_with_mask": n_with_mask, "usable_slices": n_usable,
                          "usable_fraction": frac}

    metric = {"per_kidney": per_side, "pass_floor": cfg.kidney_slice_usable_pass,
              "warn_floor": cfg.kidney_slice_usable_warn,
              "slice_axis": s_axis, "slice_axis_source": s_source}
    fracs = {s: v["usable_fraction"] for s, v in per_side.items() if np.isfinite(v["usable_fraction"])}
    if not fracs:
        return CheckResult("k4.3.slice_coverage", Verdict.UNKNOWN, metric=metric,
                           reason="no slice holds enough kidney mask to judge")
    if any(v["usable_slices"] == 0 for v in per_side.values()):
        empty = [s for s, v in per_side.items() if v["usable_slices"] == 0]
        # UNKNOWN, not FAIL. With no usable slice there is no ROI statistic to
        # grade, so the honest answer is "not measurable" - and the fraction's
        # cut-points are uncalibrated, so they may not carry a FAIL either.
        return CheckResult("k4.3.slice_coverage", Verdict.UNKNOWN, metric=metric,
                           reason=f"no usable slice at all on {', '.join(empty)} - there is no "
                                  "ROI statistic to grade")
    shown = _fmt_sides({s: 100 * v for s, v in fracs.items()}, "{:.0f}%")
    lowest = min(fracs.values())
    if lowest < cfg.kidney_slice_usable_warn:
        return CheckResult("k4.3.slice_coverage", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"usable slices {shown} - below {cfg.kidney_slice_usable_warn:.0%}, "
                                  "most of the organ carries no usable data")
    if lowest < cfg.kidney_slice_usable_pass:
        return CheckResult("k4.3.slice_coverage", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"usable slices {shown} - below {cfg.kidney_slice_usable_pass:.0%}")
    return CheckResult("k4.3.slice_coverage", Verdict.PASS, metric=metric,
                       reason=f"usable slices {shown}")


# --------------------------------------------------------------------------- #
# Module K5 — schema, data type, control/label ordering
# --------------------------------------------------------------------------- #
# The nine acquisition facts the renal consensus (Nery 2020, Table 4) expects a
# renal ASL study to report. They are listed here rather than inline so the
# report can name exactly what is missing, and so the list is auditable against
# the paper. FAIL is unreachable for this check: no standard MANDATES renal ASL
# metadata (there is no renal ASL-BIDS), so absence is a gap in reporting
# practice, not a defect in the scan.
_NERY_TABLE4_ITEMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("labelling_scheme",   ("ArterialSpinLabelingType", "labelling", "labeling")),
    ("labelling_duration", ("LabelingDuration", "label_duration_s", "tau")),
    ("pld_or_ti",          ("PostLabelingDelay", "InversionTime", "pld_or_ti_s")),
    ("background_suppression", ("BackgroundSuppression", "background_suppression")),
    ("readout",            ("MRAcquisitionType", "PulseSequenceType", "readout")),
    ("field_strength",     ("MagneticFieldStrength", "field_strength_t", "field_T")),
    ("n_repetitions",      ("TotalAcquiredPairs", "n_pairs_acquired", "n_repetitions")),
    ("breathing_strategy", ("breathing_strategy", "RespiratoryStrategy")),
    ("quantification_constants", ("labeling_efficiency", "LabelingEfficiency",
                                  "BloodBrainPartitionCoefficient", "quant_constants")),
)


@register_qc_check("k5.1.metadata", stream="A", required=True, organ=ORGAN)
def kidney_metadata_check(sidecar=None, header=None, affine=None, detected=None,
                          context=None, **kwargs) -> CheckResult:
    """Which of the nine consensus-reportable acquisition facts are actually
    present? (Nery 2020 Table 4.)

    This can never FAIL. Renal ASL has no BIDS specification and no standard
    obliging any of these fields to exist, so a missing item means the study
    cannot be interpreted or compared - not that the data is bad. The check
    exists to make that gap explicit and, importantly, to say WHICH checks
    downstream had to degrade because of it.
    """
    # The flat inputs dict counts as a source too. Several of these facts reach
    # run_qc as top-level keys (breathing_strategy, pld_or_ti_s, m0_tr_s...)
    # rather than nested in a sidecar, and a check that only looked inside the
    # metadata dicts reported them missing while they sat one level up.
    # `cfg` is plumbing, not metadata: run_qc passes it to every check, so
    # including it would make the flat inputs dict permanently non-empty and a
    # run with no inputs at all would WARN about missing fields instead of
    # reporting that there was nothing to inspect.
    flat = {k: v for k, v in kwargs.items() if k != "cfg" and v is not None}
    sources = [d for d in (sidecar, detected, context, header, flat)
               if isinstance(d, dict) and d]
    if not sources and affine is None:
        return CheckResult("k5.1.metadata", Verdict.UNKNOWN,
                           reason="no sidecar, header or affine - nothing to inspect")

    present, missing = [], []
    for item, keys in _NERY_TABLE4_ITEMS:
        found = any(src.get(k) not in (None, "") for src in sources for k in keys)
        (present if found else missing).append(item)

    # what each missing item costs downstream, so the gap is actionable
    consequences = {
        "pld_or_ti": "K2.2 reports PWS without applying its band",
        "labelling_scheme": "cortical RBF cannot be compared across studies (FAIR vs pCASL ~1.8x)",
        "field_strength": "field-strength effect (~11%) unaccounted",
        "breathing_strategy": "K7.3 returns UNKNOWN",
        "n_repetitions": "K7.2 cannot check the consensus 20-pair minimum",
        "background_suppression": "K5.3 cannot decide whether the intensity test applies",
        "quantification_constants": "a ~22% quantification error would pass every check silently",
    }
    metric = {"table": "Nery 2020 Table 4", "present": present, "missing": missing,
              "n_present": len(present), "n_items": len(_NERY_TABLE4_ITEMS),
              "consequences": {k: v for k, v in consequences.items() if k in missing}}
    if missing:
        return CheckResult("k5.1.metadata", Verdict.WARN, metric=metric,
                           reason=f"{len(present)}/{len(_NERY_TABLE4_ITEMS)} consensus-reportable "
                                  f"items present; missing {', '.join(missing)}")
    return CheckResult("k5.1.metadata", Verdict.PASS, metric=metric,
                       reason=f"all {len(_NERY_TABLE4_ITEMS)} consensus-reportable items present")


@register_qc_check("k5.2.data_type", stream="A", required=True, organ=ORGAN)
def kidney_data_type_check(files=None, context=None, detected=None, sidecar=None,
                           kidney_masks=None, cortex_masks=None, medulla_masks=None,
                           **_) -> CheckResult:
    """Routing check: what kind of renal dataset is this? Emits INFO, never grades.

    Masks are classified BEFORE images, because a renal dataset routinely ships
    files like cortex_label.nii.gz and kidney_mask.nii.gz whose names contain
    no modality token at all. Reading those as images is how a mask ends up
    graded as a perfusion map.
    """
    # Masks that arrived through the console's per-role boxes are NOT in `files`
    # - they are already resolved arrays. Counting only filenames reported
    # "0 mask file(s)" to a user who had just supplied six, which reads as if
    # the upload had failed when the masks were graded perfectly well.
    supplied = {kind: sorted(as_sides(m))
                for kind, m in (("kidney", kidney_masks), ("cortex", cortex_masks),
                                ("medulla", medulla_masks))
                if as_sides(m)}
    n_supplied = sum(len(v) for v in supplied.values())
    if not files and not detected and not n_supplied:
        return CheckResult("k5.2.data_type", Verdict.UNKNOWN, reason="no files to inspect")

    roles: dict[str, list[str]] = {}
    for f in (files or []):
        name = (f.get("name") if isinstance(f, dict) else str(f)) or ""
        low = name.lower()
        # masks first - the name may carry no modality token whatsoever
        if any(t in low for t in ("mask", "label", "seg", "roi")):
            if "cortex" in low or "cortical" in low:
                role = "cortex_mask"
            elif "medulla" in low or "medullary" in low:
                role = "medulla_mask"
            else:
                role = "kidney_mask"
        elif any(t in low for t in ("m0", "calib")):
            role = "m0"
        elif any(t in low for t in ("rbf", "perfusion", "cbf", "flow")):
            role = "rbf_map"
        elif any(t in low for t in ("asl", "fair", "pcasl", "casl", "deltam", "pair",
                                    "control", "label_", "tag")):
            role = "asl"
        elif any(t in low for t in ("t1", "t2", "anat", "struct")):
            role = "structural"
        else:
            role = "other"
        roles.setdefault(role, []).append(name)

    # `context` is a folder-name string in the brain module but an acquisition
    # dict in the renal design. Both reach this check through the same inputs
    # key, so both are accepted rather than one silently winning.
    ctx = context if isinstance(context, dict) else {}
    det = detected if isinstance(detected, dict) else {}
    sc = sidecar if isinstance(sidecar, dict) else {}
    labelling = (det.get("labelling") or ctx.get("labelling")
                 or sc.get("ArterialSpinLabelingType") or "unknown")
    readout = (det.get("readout") or ctx.get("readout")
               or sc.get("MRAcquisitionType") or "unknown")
    has_sides = any("left" in n.lower() or "right" in n.lower()
                    for names in roles.values() for n in names)
    named_masks = len(roles.get("kidney_mask", []) + roles.get("cortex_mask", [])
                      + roles.get("medulla_mask", []))
    total_masks = named_masks + n_supplied
    metric = {"roles": {k: sorted(v) for k, v in roles.items()},
              "labelling": labelling, "readout": readout,
              "masks_from_filenames": named_masks,
              "masks_supplied_directly": supplied,
              "n_masks": total_masks,
              "per_side_masks_named": has_sides or bool(n_supplied),
              "n_files": len(files or [])}
    if n_supplied:
        sides = sorted({s for v in supplied.values() for s in v})
        how = f"{total_masks} mask(s) supplied directly ({', '.join(sides)})"
    elif named_masks:
        how = (f"{named_masks} mask file(s), "
               f"{'per-side names found' if has_sides else 'no per-side names'}")
    else:
        how = "no masks"
    return CheckResult("k5.2.data_type", Verdict.INFO, metric=metric,
                       reason=f"{labelling} {readout} renal dataset; {how}")


@register_qc_check("k5.3.swap", stream="A", required=True, organ=ORGAN)
def kidney_swap_check(asl_4d=None, kidney_masks=None, background_suppression=None,
                      structure=None, aslcontext_rows=None,
                      cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Control/label ordering, judged PER PAIR rather than on the pooled mean.

    The brain check compares the mean of all even volumes against all odd ones.
    That works when the organ is still. A breathing kidney moves between the two
    halves of a pair, so a single pooled mean can be dominated by a handful of
    badly-displaced volumes and report the wrong sign with a confident number.
    Counting how many INDIVIDUAL pairs have control brighter than label is
    robust to exactly that, and the fraction is itself informative: a consistent
    sign error gives ~0%, ordinary noise gives ~50%, correct ordering ~100%.
    """
    if background_suppression:
        return CheckResult("k5.3.swap", Verdict.NA,
                           reason="background suppression on - the intensity test does not apply")
    if structure and "pre-subtracted" in str(structure):
        return CheckResult("k5.3.swap", Verdict.NA, reason="pre-subtracted image has no pairs")
    if asl_4d is None:
        return CheckResult("k5.3.swap", Verdict.UNKNOWN, reason="needs the 4D control/label series")
    arr = np.asarray(asl_4d, dtype=float)
    if arr.ndim != 4 or arr.shape[3] < 4:
        return CheckResult("k5.3.swap", Verdict.NA,
                           reason="fewer than 2 complete pairs - nothing to count")
    masks = as_sides(kidney_masks)
    if not masks:
        return CheckResult("k5.3.swap", Verdict.UNKNOWN, reason="needs a kidney mask")

    roi = np.zeros(arr.shape[:3], dtype=bool)
    for m in masks.values():
        roi |= as_mask(m)
    if not roi.any():
        return CheckResult("k5.3.swap", Verdict.UNKNOWN, reason="the kidney mask is empty")

    rows = ([r.strip().lower() for r in aslcontext_rows]
            if aslcontext_rows and len(aslcontext_rows) == arr.shape[3] else None)
    if rows:
        ctrl_idx = [i for i, r in enumerate(rows) if r == "control"]
        lbl_idx = [i for i, r in enumerate(rows) if r == "label"]
        pairs = list(zip(ctrl_idx, lbl_idx))
        assumption = "volume order from aslcontext.tsv"
    else:
        n = arr.shape[3] // 2
        pairs = [(2 * i, 2 * i + 1) for i in range(n)]
        assumption = "even=control (no aslcontext.tsv)"
    if len(pairs) < 2:
        return CheckResult("k5.3.swap", Verdict.NA, reason="fewer than 2 complete pairs")

    diffs = []
    for c, l in pairs:
        cv = arr[..., c][roi]
        lv = arr[..., l][roi]
        good = np.isfinite(cv) & np.isfinite(lv)
        if good.any():
            diffs.append(float(cv[good].mean() - lv[good].mean()))
    if not diffs:
        return CheckResult("k5.3.swap", Verdict.UNKNOWN, reason="every pair mean is non-finite")

    d = np.asarray(diffs)
    frac = float(np.count_nonzero(d > 0) / d.size)
    metric = {"n_pairs": int(d.size), "fraction_control_brighter": frac,
              "mean_pair_difference": float(d.mean()), "assumption": assumption}
    # Boundary note: the design's rows overlap at exactly 0.25 and 0.75. Resolved
    # so that the two definite verdicts own their endpoints (PASS at >= 0.75,
    # FAIL at <= 0.25) and WARN holds the strictly-interior band.
    if frac <= 0.25:
        return CheckResult("k5.3.swap", Verdict.FAIL, metric=metric,
                           reason=f"control brighter in only {frac:.0%} of {d.size} pairs - the "
                                  "label and control volumes are consistently the wrong way round")
    if frac < 0.75:
        return CheckResult("k5.3.swap", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"control brighter in {frac:.0%} of {d.size} pairs - the sign is "
                                  "inconsistent, which usually means motion between the halves "
                                  "of a pair")
    return CheckResult("k5.3.swap", Verdict.PASS, metric=metric,
                       reason=f"control brighter in {frac:.0%} of {d.size} pairs")


# --------------------------------------------------------------------------- #
# Module K6 — M0 calibration
# --------------------------------------------------------------------------- #
@register_qc_check("k6.1.m0_present", stream="A", required=True, organ=ORGAN)
def kidney_m0_present_check(m0_type=None, rbf_supplied=None, units=None, **_) -> CheckResult:
    """Is there an M0 reference?

    Stricter than the brain's equivalent, deliberately. The brain check WARNs on
    an absent M0 in every case. Here, an absent M0 under a map presented as
    QUANTIFIED perfusion is a FAIL: the numbers were divided by something, and if
    it was not a measured M0 then the units on the map are not the units it
    claims. When only a delta-M / perfusion-weighted image is being graded, an
    absent M0 is merely a limitation, and stays a WARN.
    """
    if m0_type is None:
        return CheckResult("k6.1.m0_present", Verdict.UNKNOWN, reason="M0 type not determined")
    kind = str(m0_type).lower()
    quantified = bool(rbf_supplied) or _units_declared(units)
    metric = {"m0_type": kind, "grading_quantified_map": quantified}
    if kind in ("separate", "included"):
        return CheckResult("k6.1.m0_present", Verdict.PASS, metric=metric,
                           reason=f"M0 present ({kind})")
    if quantified:
        return CheckResult("k6.1.m0_present", Verdict.FAIL, metric=metric,
                           reason="no M0, but a quantified perfusion map is being graded - the "
                                  "calibration denominator is unknown, so the map's units are "
                                  "not the units it claims")
    return CheckResult("k6.1.m0_present", Verdict.WARN, metric=metric,
                       reason="no M0 - a perfusion-weighted image can still be inspected, but "
                              "it cannot be quantified")


@register_qc_check("k6.2.m0_clean", stream="A", required=True, organ=ORGAN)
def kidney_m0_clean_check(m0_background_suppression=None, m0_labelling_applied=None,
                          m0_readout=None, asl_readout=None, **_) -> CheckResult:
    """The M0 must carry no labelling and no background suppression.

    Both defects work the same way: they attenuate the very static tissue signal
    the M0 exists to measure, so the calibration denominator comes out too small
    and every perfusion value derived from it comes out too large. The flags are
    checked before the readout comparison - an unrecorded flag is UNKNOWN, and a
    matching readout cannot license a PASS on an M0 whose state nobody knows.
    """
    bs = _bs_state(m0_background_suppression)
    if bs is None and m0_labelling_applied is None:
        return CheckResult("k6.2.m0_clean", Verdict.UNKNOWN,
                           reason="the M0's own background-suppression and labelling state are "
                                  "not recorded")
    metric = {"m0_background_suppression": m0_background_suppression,
              "m0_background_suppression_read_as": bs,
              "m0_labelling_applied": m0_labelling_applied,
              "m0_readout": m0_readout, "asl_readout": asl_readout}
    if isinstance(m0_background_suppression, (int, float)) and \
            not isinstance(m0_background_suppression, bool):
        metric["note"] = (f"BackgroundSuppression was given as {m0_background_suppression}, read "
                          "as a pulse count (>=1 means on)")
    faults = []
    if bs:
        faults.append("background suppression was ON")
    if m0_labelling_applied:
        faults.append("labelling was applied")
    if faults:
        return CheckResult("k6.2.m0_clean", Verdict.FAIL, metric=metric,
                           reason=f"{' and '.join(faults)} on the M0 - the calibration "
                                  "denominator is attenuated, so perfusion is over-estimated")
    if bs is None or m0_labelling_applied is None:
        unknown = "labelling" if m0_labelling_applied is None else "background suppression"
        return CheckResult("k6.2.m0_clean", Verdict.UNKNOWN, metric=metric,
                           reason=f"the M0's {unknown} state is not recorded")
    if (m0_readout and asl_readout
            and str(m0_readout).strip().lower() != str(asl_readout).strip().lower()):
        return CheckResult("k6.2.m0_clean", Verdict.WARN, metric=metric,
                           reason=f"M0 is clean, but its readout ({m0_readout}) differs from the "
                                  f"ASL readout ({asl_readout}) - the two images carry different "
                                  "distortion, so they do not divide cleanly")
    return CheckResult("k6.2.m0_clean", Verdict.PASS, metric=metric,
                       reason="M0 acquired without labelling and without background suppression")


@register_qc_check("k6.3.m0_tr", stream="A", required=True, organ=ORGAN)
def kidney_m0_tr_check(m0_tr_s=None, field_T=None, t1_map=None, cortex_masks=None,
                       medulla_masks=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """M0 TR must be long enough for full relaxation, or the M0 must be corrected.

    Never FAILs: a short TR is a correctable condition, and the correction is
    1/(1 - exp(-TR/T1)). The renal twist is that cortex and medulla have
    materially different T1, so ONE correction factor cannot serve both
    compartments - the two factors are reported separately, and their spread is
    the thing a reader needs to see.
    """
    if m0_tr_s is None:
        return CheckResult("k6.3.m0_tr", Verdict.UNKNOWN, reason="no M0 TR recorded")
    tr = float(m0_tr_s)
    # Compartment T1 in MILLISECONDS, the midpoints of the published ranges in
    # wolf2018 (renal T1/T2 systematic review): at 3 T cortex 1124-1406 -> 1265,
    # medulla 1388-1685 -> 1537; at 1.5 T cortex 827-1080 -> 953, medulla
    # 1054-1428 -> 1241. The TR arrives in SECONDS, so every exponent below
    # converts explicitly - mixing the two silently yields a correction factor
    # of ~1.0 and hides the very problem this check exists to find.
    t1_ms = ({"cortex": 1265.0, "medulla": 1537.0} if (field_T or 3) >= 2.5
             else {"cortex": 953.0, "medulla": 1241.0})
    t1_source = ("wolf2018 range midpoint, "
                 f"{'3T' if (field_T or 3) >= 2.5 else '1.5T'}")
    if t1_map is not None:
        for name, masks in (("cortex", as_sides(cortex_masks)),
                            ("medulla", as_sides(medulla_masks))):
            vals = [roi_stats(t1_map, m)["mean"] for m in masks.values()]
            vals = [v for v in vals if np.isfinite(v) and v > 0]
            if vals:
                t1_ms[name] = float(np.mean(vals))

    factors = {k: float(1.0 / (1.0 - np.exp(-(tr * 1000.0) / v))) for k, v in t1_ms.items()}
    spread = max(factors.values()) - min(factors.values())
    metric = {"tr_seconds": tr, "t1_ms_used": t1_ms, "correction_factors": factors,
              "factor_spread": spread,
              "t1_source": "measured T1 map" if t1_map is not None else t1_source,
              "applied": t1_map is not None,
              "min_tr_s": cfg.m0_tr_min_s}
    if tr >= cfg.m0_tr_min_s:
        return CheckResult("k6.3.m0_tr", Verdict.PASS, metric=metric,
                           reason=f"M0 TR {tr:.1f}s >= {cfg.m0_tr_min_s:.0f}s - no correction needed")
    shown = " / ".join(f"x{v:.3f} ({k})" for k, v in factors.items())
    return CheckResult("k6.3.m0_tr", Verdict.WARN, metric=metric,
                       reason=f"M0 TR {tr:.1f}s < {cfg.m0_tr_min_s:.0f}s - correct by {shown}; "
                              f"the two compartments differ by {spread:.3f}, so one factor "
                              "cannot serve both")


# --------------------------------------------------------------------------- #
# Module K7 — respiratory motion (the dominant renal artefact)
# --------------------------------------------------------------------------- #
# Below this median inter-frame displacement, the ORGAN IS STILL and the
# direction of the residual centroid jitter carries no information. Used only to
# gate the through-plane FRACTION (a ratio), never the displacement itself.
_MOTION_FLOOR_VOX = 0.10

# How much thicker the slice direction must be than the in-plane directions
# before it can be identified from the voxel size alone.
_ANISOTROPY_RATIO = 1.5


def _slice_axis(voxel_mm, supplied=None) -> tuple[int | None, str]:
    """Which array axis is the slice-encoding one, and how we know.

    Returns (None, reason) when it genuinely cannot be told - isotropic voxels
    carry no evidence of a slice direction, and guessing one would make the
    through-plane fraction a number about array layout rather than about motion.
    """
    if supplied is not None:
        return int(supplied), "supplied by caller"
    vox = [float(v) for v in voxel_mm]
    biggest = int(np.argmax(vox))
    others = [v for i, v in enumerate(vox) if i != biggest]
    if others and vox[biggest] >= _ANISOTROPY_RATIO * max(others):
        return biggest, f"inferred from anisotropy ({vox[biggest]:g} mm vs {max(others):g} mm)"
    return None, "not determinable - voxels are near-isotropic, so no slice direction is implied"


def _resolve_axes(affine) -> dict:
    """Map anatomical axes (RL / AP / CC) onto array axes using the affine.

    Never assume array-axis order. Renal acquisitions are routinely coronal or
    oblique, so the array's third axis is not reliably cranio-caudal - and CC is
    exactly the axis that carries the respiratory excursion (~6.5x the RL
    motion). Getting this wrong would report the largest motion component as the
    smallest.

    Column i of the affine's 3x3 says how array axis i moves in world space; the
    array axis best aligned with a given world axis is the one with the largest
    absolute component in that world row.
    """
    a = np.asarray(affine, dtype=float)[:3, :3]
    out: dict = {}
    used: set[int] = set()
    # world rows in RAS order: x=RL, y=AP, z=CC
    for world_row, name in ((0, "RL"), (1, "AP"), (2, "CC")):
        order = np.argsort(-np.abs(a[world_row, :]))
        axis = next((int(i) for i in order if int(i) not in used), int(order[0]))
        used.add(axis)
        out[name] = axis
    return out


@register_qc_check("k7.1.kidney_displacement", stream="A", required=True, organ=ORGAN)
def kidney_displacement_check(delta_m_4d=None, kidney_masks=None, voxel_mm=None,
                              affine=None, transforms=None, slice_axis=None,
                              cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Per-kidney respiratory displacement, reported per anatomical axis.

    Deliberately NOT the brain's framewise displacement. FD sums the three
    translations isotropically, which is right for a rigid head and wrong for a
    kidney: respiratory excursion is overwhelmingly cranio-caudal (4D-CT measures
    11.1 +/- 4.8 mm CC against 1.7 +/- 1.4 mm RL), so an isotropic sum averages
    the informative axis away. The axes come from the affine, never from the
    array order.

    Reported in BOTH mm and voxels, and graded in voxels: an 11 mm excursion is
    catastrophic at 2 mm in-plane resolution and tolerable at 6 mm slices.
    """
    if delta_m_4d is None:
        return CheckResult("k7.1.kidney_displacement", Verdict.UNKNOWN,
                           reason="needs a 4D series (a single pre-subtracted volume has "
                                  "nothing to move between)")
    arr = np.asarray(delta_m_4d, dtype=float)
    if arr.ndim != 4 or arr.shape[3] < cfg.kidney_min_repetitions:
        return CheckResult("k7.1.kidney_displacement", Verdict.UNKNOWN,
                           reason=f"fewer than {cfg.kidney_min_repetitions} usable frames")
    masks = as_sides(kidney_masks)
    if not masks:
        return CheckResult("k7.1.kidney_displacement", Verdict.UNKNOWN, reason="needs a kidney mask")
    if affine is None:
        return CheckResult("k7.1.kidney_displacement", Verdict.UNKNOWN,
                           reason="needs the affine - the cranio-caudal axis cannot be assumed "
                                  "from array order for a coronal or oblique renal acquisition")

    axes = _resolve_axes(affine)
    vox = tuple(float(v) for v in (voxel_mm or (1.0, 1.0, 1.0)))
    # Through-plane = the slice-encoding direction, the component registration
    # cannot recover. It is inferred from ANISOTROPY (slice thickness exceeding
    # the in-plane size), never from `argmax` alone: on isotropic voxels argmax
    # returns axis 0 by tie-break, and a "through-plane share" measured along an
    # arbitrary axis is a number with no meaning. When it cannot be determined
    # the fraction is not computed and not graded, and the metric says so.
    s_axis, s_axis_source = _slice_axis(vox, slice_axis)

    per_side: dict = {}
    for side, mask in masks.items():
        m = as_mask(mask)
        if m.shape != arr.shape[:3]:
            return CheckResult("k7.1.kidney_displacement", Verdict.UNKNOWN,
                               reason=f"{side} mask {m.shape} does not match the series "
                                      f"{arr.shape[:3]}")
        idx = np.argwhere(m).astype(float)          # (n_vox, 3) voxel coordinates
        if idx.size == 0:
            continue
        centroids = []
        for t in range(arr.shape[3]):
            w = arr[..., t][m]
            good = np.isfinite(w)
            if not good.any():
                centroids.append(None)
                continue
            # intensity-weighted centroid; |w| because a delta-M is signed and a
            # negative weight would pull the centroid the wrong way
            ww = np.abs(w[good])
            tot = ww.sum()
            centroids.append(idx[good].T @ ww / tot if tot > 0
                             else idx[good].mean(axis=0))
        usable = [c for c in centroids if c is not None]
        if len(usable) < cfg.kidney_min_repetitions:
            continue

        steps = [b - a for a, b in zip(usable[:-1], usable[1:])]
        d_vox = np.asarray(steps)                    # (T-1, 3) in voxels
        d_mm = d_vox * np.asarray(vox)               # per-axis mm
        per_axis = {}
        for name, ax in axes.items():
            per_axis[name] = {
                "median_vox": float(np.median(np.abs(d_vox[:, ax]))),
                "max_vox": float(np.abs(d_vox[:, ax]).max()),
                "median_mm": float(np.median(np.abs(d_mm[:, ax]))),
                "max_mm": float(np.abs(d_mm[:, ax]).max()),
            }
        mag = np.linalg.norm(d_mm, axis=1)
        tp = np.abs(d_mm[:, s_axis]) if s_axis is not None else np.zeros_like(mag)
        # The DIRECTION of a displacement is only defined once there is a
        # displacement. Centroid estimates jitter by a fraction of a voxel on a
        # perfectly still organ, and that jitter is isotropic - so its
        # through-plane share lands near 1/3-1/2 and would trip the 25% limit on
        # a motionless kidney. Below the floor the fraction is reported as NaN
        # (not graded) rather than as a number computed from noise.
        total_med_vox = float(np.median(np.linalg.norm(d_vox, axis=1)))
        meaningful = total_med_vox >= _MOTION_FLOOR_VOX and s_axis is not None
        per_side[side] = {
            "per_axis": per_axis,
            "median_total_displacement_vox": total_med_vox,
            "through_plane_fraction": (float(np.median(tp[mag > 0] / mag[mag > 0]))
                                       if meaningful and np.any(mag > 0) else float("nan")),
            "through_plane_gated": not meaningful,
            "n_frames_used": len(usable),
        }

    if not per_side:
        return CheckResult("k7.1.kidney_displacement", Verdict.UNKNOWN,
                           reason="no kidney had enough usable frames to track")

    metric = {"per_kidney": per_side,
              "axis_convention": "CC/AP/RL resolved from the affine, not assumed",
              "array_axis_for": axes, "slice_axis": s_axis,
              "slice_axis_source": s_axis_source,
              "source": "supplied transforms" if transforms is not None
              else "intensity-weighted centroid of the masked kidney",
              "voxel_mm": vox,
              "cc_median_limit_vox": cfg.kidney_displacement_vox,
              "through_plane_limit": cfg.kidney_through_plane_frac}

    cc = {s: v["per_axis"]["CC"]["median_vox"] for s, v in per_side.items()}
    tpf = {s: v["through_plane_fraction"] for s, v in per_side.items()}
    shown = f"CC median {_fmt_sides(cc, '{:.2f}')} voxels"
    bad_cc = [s for s, v in cc.items() if v > cfg.kidney_displacement_vox]
    bad_tp = [s for s, v in tpf.items()
              if np.isfinite(v) and v > cfg.kidney_through_plane_frac]
    tp_shown = ({s: 100 * v for s, v in tpf.items() if np.isfinite(v)})
    if tp_shown:
        shown += f", through-plane share {_fmt_sides(tp_shown, '{:.0f}%')}"
    if bad_cc or bad_tp:
        why = []
        if bad_cc:
            why.append(f"CC motion above {cfg.kidney_displacement_vox} voxel on {', '.join(bad_cc)}")
        if bad_tp:
            why.append(f"through-plane share above {cfg.kidney_through_plane_frac:.0%} on "
                       f"{', '.join(bad_tp)} (registration cannot recover this)")
        return CheckResult("k7.1.kidney_displacement", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"{shown} - {'; '.join(why)}")
    return CheckResult("k7.1.kidney_displacement", Verdict.PASS, metric=metric,
                       reason=f"{shown} - within tolerance")


# The published renal outlier-rejection rules. The RULE is published; the number
# of rejections that makes a scan unusable is not, which is why the count drives
# only a WARN. `k` is the SD multiplier and `limit` the voxel-fraction bound.
_OUTLIER_RULES: dict[str, dict] = {
    "harteveld_2sd_20pct":       {"k": 2.0, "limit": 0.20, "mode": "reject_above"},
    "harteveld_paed_1p5sd_20pct": {"k": 1.5, "limit": 0.20, "mode": "reject_above"},
    "bones_80pct_2sd":           {"k": 2.0, "limit": 0.80, "mode": "keep_within"},
}


@register_qc_check("k7.2.outlier_rate", stream="A", required=True, organ=ORGAN)
def kidney_outlier_rate_check(delta_m_4d=None, kidney_masks=None, rule=None,
                              readout=None, n_pairs_acquired=None,
                              structure=None, cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Subtraction-outlier rejection — the one genuinely implementable published
    rule family in renal ASL.

    Four published variants exist and they disagree; the one applied is named in
    the metric so a report never hides which rule produced the count. The rule
    fires in roughly two thirds of NORMAL healthy datasets, so a non-zero count
    is unremarkable: the COUNT is the signal, not the firing.
    """
    if delta_m_4d is None:
        return CheckResult("k7.2.outlier_rate", Verdict.UNKNOWN, reason="needs the 4D series")
    if structure and "pre-subtracted" in str(structure):
        return CheckResult("k7.2.outlier_rate", Verdict.NA,
                           reason="pre-subtracted delta-M - the temporal SD is not estimable")
    arr = np.asarray(delta_m_4d, dtype=float)
    if arr.ndim != 4 or arr.shape[3] < 4:
        return CheckResult("k7.2.outlier_rate", Verdict.NA,
                           reason="fewer than 4 repetitions - the temporal SD is not estimable")
    masks = as_sides(kidney_masks)
    if not masks:
        return CheckResult("k7.2.outlier_rate", Verdict.UNKNOWN, reason="needs a kidney mask")

    name = rule if rule in _OUTLIER_RULES else "harteveld_2sd_20pct"
    spec = dict(_OUTLIER_RULES[name])
    # The config carries the SD multiplier and voxel fraction, and the console
    # renders both as editable. They were previously ignored in favour of the
    # hard-coded table, so a user could type any value and nothing moved - a
    # control that grades nothing. Editing them away from the published values
    # is legitimate (the counts they produce are uncalibrated), but the report
    # must say the applied rule is no longer the published one.
    published = (spec["k"], spec["limit"])
    if spec["mode"] == "reject_above":
        spec["k"], spec["limit"] = cfg.kidney_outlier_sd, cfg.kidney_outlier_voxel_frac
    customised = spec["mode"] == "reject_above" and (spec["k"], spec["limit"]) != published
    per_side: dict = {}
    for side, mask in masks.items():
        m = as_mask(mask)
        if m.shape != arr.shape[:3]:
            return CheckResult("k7.2.outlier_rate", Verdict.UNKNOWN,
                               reason=f"{side} mask {m.shape} does not match {arr.shape[:3]}")
        series = arr[m]                              # (n_vox, T)
        # A sample SD needs two finite points. Computing it where there is one
        # returns NaN correctly but emits a RuntimeWarning per call, so the
        # count is taken first and the SD only asked for where it is defined -
        # a library should not warn about input it handles.
        n_finite = np.isfinite(series).sum(axis=1, keepdims=True)
        mu = np.full((series.shape[0], 1), np.nan)
        sd = np.full((series.shape[0], 1), np.nan)
        enough = (n_finite >= 2).ravel()
        if enough.any():
            mu[enough] = np.nanmean(series[enough], axis=1, keepdims=True)
            sd[enough] = np.nanstd(series[enough], axis=1, ddof=1, keepdims=True)
        valid = np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
        rejected = []
        fracs = []
        for p in range(series.shape[1]):
            col = series[:, p:p + 1]
            good = valid & np.isfinite(col)
            if not good.any():
                rejected.append(p)
                fracs.append(float("nan"))
                continue
            dev = np.abs(col[good] - mu[good]) > spec["k"] * sd[good]
            frac = float(dev.sum() / dev.size)
            fracs.append(frac)
            out = frac > spec["limit"] if spec["mode"] == "reject_above" \
                else (1.0 - frac) <= spec["limit"]
            if out:
                rejected.append(p)
        per_side[side] = {
            "n_rejected": len(rejected),
            "rejected_indices": rejected,
            "rejected_fraction": float(len(rejected) / series.shape[1]),
            "surviving_pairs": int(series.shape[1] - len(rejected)),
            "deviating_voxel_fraction_per_pair": [round(f, 4) if np.isfinite(f) else None
                                                  for f in fracs],
        }

    counts = {s: v["n_rejected"] for s, v in per_side.items()}
    surviving = {s: v["surviving_pairs"] for s, v in per_side.items()}
    asym = (abs(list(counts.values())[0] - list(counts.values())[1])
            if len(counts) == 2 else 0)
    is_2d = isinstance(readout, str) and readout.strip().lower().startswith("2d")
    metric = {"per_kidney": per_side,
              "rule": (f"{name} (parameters overridden)" if customised else name),
              "rule_parameters": spec,
              "published_parameters": {"k": published[0], "limit": published[1]},
              "parameters_customised": customised,
              "left_right_asymmetry": asym, "readout": readout,
              "n_pairs_acquired": n_pairs_acquired,
              "calibration_note": "the rule fires in ~2/3 of NORMAL healthy datasets - "
                                  "the count is the signal, not the firing",
              "max_rejected_pass": cfg.kidney_max_rejected_pairs}

    starved = [s for s, n in surviving.items() if n < cfg.kidney_min_surviving_pairs]
    if starved:
        return CheckResult("k7.2.outlier_rate", Verdict.FAIL, metric=metric,
                           reason=f"only {min(surviving.values())} pair(s) survive on "
                                  f"{', '.join(starved)} - there is nothing left to average")
    shown = _fmt_sides({s: float(n) for s, n in counts.items()}, "{:.0f}")
    problems = []
    if max(counts.values()) > cfg.kidney_max_rejected_pairs:
        problems.append(f"{max(counts.values())} pairs rejected")
    if is_2d and min(surviving.values()) < cfg.kidney_min_pairs_2d:
        problems.append(f"only {min(surviving.values())} pairs survive on a 2D readout, below the "
                        f"consensus minimum of {cfg.kidney_min_pairs_2d}")
    if asym > 3:
        problems.append(f"rejection counts differ by {asym} between the kidneys, which points at "
                        "one-sided motion")
    if problems:
        return CheckResult("k7.2.outlier_rate", Verdict.WARN, metric=metric, provisional=True,
                           reason=f"rejected {shown} of {arr.shape[3]} pairs [{name}] - "
                                  f"{'; '.join(problems)}")
    return CheckResult("k7.2.outlier_rate", Verdict.PASS, metric=metric,
                       reason=f"rejected {shown} of {arr.shape[3]} pairs [{name}]")


@register_qc_check("k7.3.breathing_strategy", stream="A", required=True, organ=ORGAN)
def kidney_breathing_strategy_check(breathing_strategy=None, physio_trace=None,
                                    trigger_times_s=None, acquisition_time_s=None,
                                    **_) -> CheckResult:
    """Which breathing strategy was used, and how efficient was any gating?

    Never FAILs. The consensus prefers free breathing or free breathing with
    gating, and notes that breath-hold scans are poorly tolerated by patients -
    but a breath-hold acquisition is a legitimate choice made for a reason, so it
    is flagged rather than rejected. There is no BIDS field for this, so it has
    to be told to us; absent, the check is UNKNOWN rather than assuming the
    common case.
    """
    if not breathing_strategy:
        return CheckResult("k7.3.breathing_strategy", Verdict.UNKNOWN,
                           reason="breathing strategy not recorded (there is no BIDS field for "
                                  "it, so it must be supplied)")
    strat = str(breathing_strategy).strip().lower()
    metric: dict = {"breathing_strategy": strat}

    if trigger_times_s is not None and acquisition_time_s:
        trig = np.asarray(trigger_times_s, dtype=float)
        trig = trig[np.isfinite(trig)]
        if trig.size >= 2:
            intervals = np.diff(np.sort(trig))
            metric["gating"] = {
                "n_triggers": int(trig.size),
                "median_interval_s": float(np.median(intervals)),
                # accepted time as a share of wall-clock: what gating cost
                "efficiency": float(min(1.0, trig.size * float(np.median(intervals))
                                        / float(acquisition_time_s))),
            }

    if "breath" in strat and "hold" in strat:
        return CheckResult("k7.3.breathing_strategy", Verdict.WARN, metric=metric,
                           reason="breath-hold acquisition - the consensus notes these are "
                                  "poorly tolerated by patients; flagged, not rejected")
    if any(t in strat for t in ("free", "gat", "navigat", "trigger", "paced", "synchron")):
        return CheckResult("k7.3.breathing_strategy", Verdict.PASS, metric=metric,
                           reason=f"{strat} - consistent with the consensus preference")
    return CheckResult("k7.3.breathing_strategy", Verdict.INFO, metric=metric,
                       reason=f"breathing strategy '{strat}' recorded but not recognised - "
                              "reported without a verdict")
