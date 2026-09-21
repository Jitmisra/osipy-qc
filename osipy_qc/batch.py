"""
Batch QC — grade a whole cohort, not one scan at a time.

The proposal's dashboard is a cohort view: an overview of pass/warn/fail across N
subjects, a participant ledger, and a per-subject deep dive. This module is the
engine behind it — it grades a folder of subjects (or a synthetic demo cohort)
and computes the aggregate statistics the dashboard renders.

A "subject folder" is any immediate subdirectory of the batch folder that
contains a CBF map. GM/WM/CSF tissue maps are picked up if present (so the QEI
and level checks run); otherwise those checks report UNKNOWN, exactly as for a
single scan.

Pure NumPy + nibabel, like everything else.
"""

from __future__ import annotations

import glob
import os
import math

from dataclasses import dataclass, field, replace

from .core.config import POPULATIONS, QCConfig, for_population
from .core.result import Verdict
from .report import QCReport, run_qc

# Thresholds the frontend config panel exposes, grouped by module. This is
# EVERY threshold that grades the CBF-map (Stream B) checks the dashboard runs -
# so tuning any of them visibly moves a verdict. Labels are short because the
# group header carries the context ("min" under "GM CBF band").
#
# Three kinds of field are deliberately NOT here, because they are not grading
# knobs and exposing them would mislead:
#   * the QEI curve constants qei_a..qei_f - these are the FITTED QEI model
#     (byte-faithful to ASLPrep). They are not pass/fail cutoffs; changing them
#     silently de-calibrates the published 0.5 threshold.
#   * acquisition facts (labeling_efficiency, label/PLD, t1_blood, t1_tissue) -
#     these are per-scan metadata the QC layer is TOLD, not thresholds. They
#     belong on the upload form, not a cohort threshold panel.
#   * raw-data-stream thresholds (motion FD, M0 TR, coregistration Dice) - those
#     checks do not run on a folder of CBF maps, so a knob here would do nothing.
TUNABLE_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("QEI", [("qei_pass", "pass"), ("qei_warn", "cutoff"),
             ("tissue_thresh", "tissue prob"), ("smooth_fwhm_mm", "smooth mm")]),
    ("Spatial CoV", [("scov_vascular", "vascular"), ("scov_artifact", "artifact")]),
    ("GM CBF band", [("gm_cbf_lo", "min"), ("gm_cbf_hi", "max"),
                     ("gm_cbf_fail_lo", "fail-low"), ("gm_cbf_fail_hi", "fail-high")]),
    ("WM CBF band", [("wm_cbf_lo", "min"), ("wm_cbf_hi", "max"),
                     ("wm_cbf_fail_lo", "fail-low"), ("wm_cbf_fail_hi", "fail-high")]),
    ("GM/WM ratio", [("ratio_pass", "pass"), ("ratio_min", "min")]),
    ("Negative CBF", [("neg_gm_warn", "warn frac"), ("neg_gm_fail", "fail frac")]),
    ("Deep GM (neonatal)", [("deep_gm_ratio_lo", "min"), ("deep_gm_ratio_hi", "max")]),
    ("Coverage", [("coverage_warn", "warn"), ("coverage_fail", "fail")]),
]

# Flat {name: label} view, derived from the groups - the allow-list that
# cfg_from_params tunes and the drawer renders.
# The kidney and placenta thresholds, grouped for the same editor. They are kept
# separate from the brain's because they are not alternatives to it - a kidney
# has no GM/WM ratio and a placenta has no QEI, so showing the brain's groups
# under a renal heading would offer the reader controls that grade nothing.
#
# Almost every number here is UNCALIBRATED (the renal consensus states 59 rules
# and zero quality thresholds; the placenta has no consensus document at all),
# which is exactly why they are exposed for editing: they are engineering
# defaults awaiting calibration, not settled science.
KIDNEY_TUNABLE_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Cortical RBF sanity bound", [("kidney_rbf_sanity_lo", "min"),
                                   ("kidney_rbf_sanity_hi", "max")]),
    ("Implausible values", [("kidney_implausible_ceiling", "ceiling"),
                            ("kidney_frac_warn", "warn frac"),
                            ("kidney_frac_fail", "fail frac")]),
    ("Perfusion-weighted signal", [("kidney_pws_lo", "min %"), ("kidney_pws_hi", "max %")]),
    ("Cortico-medullary ratio", [("kidney_cmr_trip", "trip point")]),
    ("Left-right", [("kidney_asymmetry_tol", "tolerance")]),
    ("Masks", [("kidney_mask_component_frac", "one-object frac"),
               ("kidney_min_cortex_voxels", "min cortex vox"),
               ("kidney_min_roi_voxels", "min ROI vox")]),
    ("Slice coverage", [("kidney_slice_usable_pass", "pass"),
                        ("kidney_slice_usable_warn", "warn")]),
    ("Respiratory motion", [("kidney_displacement_vox", "CC median vox"),
                            ("kidney_through_plane_frac", "through-plane")]),
    ("Outlier rejection", [("kidney_outlier_sd", "SD"),
                           ("kidney_outlier_voxel_frac", "voxel frac"),
                           ("kidney_max_rejected_pairs", "max rejected"),
                           ("kidney_min_pairs_2d", "min pairs (2D)")]),
]

PLACENTA_TUNABLE_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Implausible values", [("placenta_neg_frac_warn", "negative warn"),
                            ("placenta_upper_frac_warn", "upper warn"),
                            ("placenta_nonfinite_fail", "non-finite fail"),
                            ("placenta_iqr_multiplier", "IQR multiplier")]),
    ("Mask & slab", [("placenta_mask_component_frac", "one-object frac"),
                     ("placenta_holes_frac_warn", "holes warn"),
                     ("placenta_edge_voxel_frac", "slab-edge frac"),
                     ("placenta_covered_frac", "coverage")]),
    ("Gestational age", [("placenta_ga_min_wk", "min wk"), ("placenta_ga_max_wk", "max wk")]),
    ("Outlier rejection", [("placenta_outlier_sd", "SD"),
                           ("placenta_outlier_voxel_frac", "voxel frac"),
                           ("placenta_rejected_frac_warn", "rejected warn"),
                           ("placenta_min_surviving_pairs", "min surviving")]),
    ("Temporal stability", [("placenta_tsd_warn_pct", "temporal SD %")]),
    ("Registration residual", [("placenta_ncc_pass", "NCC"),
                               ("placenta_ssim_pass", "local SSIM"),
                               ("placenta_bad_volume_frac", "bad-volume frac")]),
    ("Contractions", [("placenta_contraction_drop", "area drop")]),
]

TUNABLE_GROUPS_BY_ORGAN: dict[str, list] = {
    "brain": TUNABLE_GROUPS,
    "kidney": KIDNEY_TUNABLE_GROUPS,
    "placenta": PLACENTA_TUNABLE_GROUPS,
}


def tunable_groups(organ: str = "brain") -> list:
    """The threshold groups that actually grade this organ."""
    return TUNABLE_GROUPS_BY_ORGAN.get(organ, TUNABLE_GROUPS)


TUNABLE: dict[str, str] = {n: lab
                           for groups in TUNABLE_GROUPS_BY_ORGAN.values()
                           for _, fields in groups for n, lab in fields}

# Map a check id to a short human name for the ledger / artifact breakdown.
CHECK_LABELS: dict[str, str] = {
    "1.qei": "QEI", "2.1.spatial_cov": "sCoV", "2.2.snr": "SNR",
    "2.3.histogram": "Histogram", "3.1.cbf_level": "CBF level",
    "3.2.gm_wm_ratio": "GM/WM ratio", "3.3.negative_gm": "Negative CBF",
    "3.4.deep_gm_ratio": "Deep GM", "4.1.coregistration": "Coregistration",
    "4.2.coverage": "Coverage", "5.1.schema": "Schema",
    "5.2.volume_integrity": "Volume integrity", "5.3.swap": "Control/label",
    "6.1.m0_present": "M0 present", "6.2.m0_tr": "M0 TR",
    "6.3.m0_no_bs": "M0 background-suppression", "6.5.m0_geometry": "M0 geometry",
    "7.1.motion": "Motion", "8.2.data_type": "Data type",
}


def check_label(check: str) -> str:
    return CHECK_LABELS.get(check, check)


def stream_b_checks(organ: str = "brain") -> list[str]:
    """The CBF-map QC checks for one organ. A cohort of CBF maps is graded against
    these, so a clean map can genuinely PASS rather than being dragged to WARN by
    the raw-data checks that have no input to run on.

    Scoped to an organ since kidney and placenta arrived: unscoped, this returned
    the brain's Stream B plus the kidney's, and a brain upload was measured
    against a cortico-medullary ratio it can never have."""
    from .core.registry import all_checks
    return [name for name, e in all_checks(organ).items() if e.get("stream") == "B"]


def cbf_map_checks(organ: str = "brain") -> list[str]:
    """Stream B minus coregistration — the right set for a cohort of CBF maps with
    no T1. Coregistration needs both an ASL and a structural mask, which the CBF
    loader never has, so including it would only add an UNKNOWN and drag every
    subject to WARN. Callers who do have T1 masks can pass `checks=` explicitly."""
    return [c for c in stream_b_checks(organ) if c != "4.1.coregistration"]


@dataclass
class Subject:
    """One graded scan in a cohort."""

    sid: str
    report: QCReport
    inputs: dict = field(default_factory=dict)
    cfg: QCConfig = field(default_factory=QCConfig)

    @property
    def overall(self) -> str:
        return self.report.overall.value

    @property
    def qei(self):
        for r in self.report.results:
            if r.check == "1.qei" and "qei" in r.metric:
                return r.metric["qei"]
        return None

    @property
    def primary_artifact(self) -> str:
        """The check most responsible for this subject's verdict: the first FAIL,
        else the first WARN, else '-'. This is what the ledger shows.

        A scan that simply could not be fully graded reads as 'incomplete', not
        as a flag — 'couldn't grade' is a different message from 'looks
        borderline'.

        This used to be gated on `self.overall == "WARN"`, which was correct only
        while UNKNOWN escalated to WARN. Now that it does not, that condition can
        never hold here: reaching it means no result was FAIL or WARN, so the
        overall cannot be WARN either, and the ledger showed '-' for a subject
        graded on one check out of eleven — identical to a fully graded clean one.
        The gate is now the unknowns themselves, which is what the column is
        actually reporting."""
        for want in (Verdict.FAIL, Verdict.WARN):
            for r in self.report.results:
                if r.verdict is want:
                    return check_label(r.check)
        n_unknown = sum(1 for r in self.report.results if r.verdict is Verdict.UNKNOWN)
        if n_unknown:
            # "unknown", not "n/a": N/A means the check cannot apply to this data
            # and is not a gap, whereas these are gaps.
            return f"incomplete ({n_unknown} unknown)"
        return "-"


# --------------------------------------------------------------------------- #
# building a batch
# --------------------------------------------------------------------------- #
def subject_inputs(subject_dir: str, organ: str = "brain") -> dict | None:
    """Everything gradeable inside one subject folder, or None if nothing is.

    This used to carry its own glob patterns - a fourth private copy of the
    filename rules, after `classify_role`, the upload page, and the pipeline
    adapters. It had drifted the way every copy of those rules has drifted:
    `*_cbf.nii*` missed a real dataset's `..._label-meancbf.nii` (uncompressed,
    and no underscore before "cbf"), so the subject was skipped entirely and
    silently vanished from the cohort. `*gm*.nii*` went the other way and would
    have matched `segmentation.nii.gz` as a grey-matter map.

    It now calls `load_folder`, which is what the CLI and the upload console
    already use. Two things follow. Subjects are found by the one vocabulary the
    rest of the package applies, and a subject folder that also holds its raw
    acquisition gets Stream A graded too, instead of only the CBF map.
    """
    from .io import load_folder, load_organ_folder

    inputs = (load_folder(subject_dir) if organ == "brain"
              else load_organ_folder(subject_dir, organ))
    return inputs if _is_subject(inputs, organ) else None


def _is_subject(inputs: dict, organ: str = "brain") -> bool:
    """Is there a scan here, or only supporting files?

    A perfusion map or an acquisition. Deliberately NOT "any recognised file": a
    BIDS subject is laid out `sub-01/anat/` + `sub-01/perf/`, and counting a lone
    T1 as a subject would split one person into two rows of a cohort ledger -
    and, worse, make the upload console read a single BIDS subject as a
    two-subject cohort. For kidney and placenta the same rule applies to a lone
    mask, which is supporting data and not a scan.
    """
    if organ != "brain":
        from .io import classify_organ_file

        key = {"kidney": "rbf_map", "placenta": "perfusion_map"}[organ]
        if inputs.get(key) is not None:
            return True
        # header-only survey: load_organ_folder does not build the map without
        # arrays, so fall back to the same filename vocabulary it would apply
        return any(classify_organ_file(f["name"])[0] in ("perfusion", "asl")
                   for f in inputs.get("files") or [])
    # cbf_path as well as cbf: with load_arrays=False the loader records where
    # the map is without reading it, and `subject_dirs` relies on exactly that to
    # survey a cohort from headers alone. Checking only `cbf` reported an empty
    # cohort for a folder of four perfectly good CBF maps.
    return bool(inputs.get("cbf") is not None or inputs.get("cbf_path")
                or any(f.get("role") == "asl" for f in inputs.get("files") or []))


def _has_raw(inputs: dict) -> bool:
    """Did this folder contain an actual acquisition, as opposed to only maps?

    Decides whether the Stream A checks are worth running. Asking for them when
    the folder holds nothing but a CBF map adds ten UNKNOWNs and drags an
    otherwise clean subject down the ledger. Wider than `_is_subject` on purpose:
    an M0 sitting beside a CBF map justifies the M0 checks even though an M0 on
    its own is not a subject.
    """
    return any(f.get("role") in ("asl", "m0", "t1")
               for f in inputs.get("files") or [])


def subject_dirs(folder: str, organ: str = "brain") -> list[str]:
    """Names of the immediate subdirectories that hold a gradeable scan.

    Reads NIfTI headers only, never the voxels, so the upload console can ask
    "is this one subject or a cohort?" without paying to load every array twice.
    """
    from .io import load_folder, load_organ_folder

    if not os.path.isdir(folder):
        return []
    out = []
    for name in sorted(os.listdir(folder)):
        d = os.path.join(folder, name)
        if not os.path.isdir(d):
            continue
        try:
            got = (load_folder(d, load_arrays=False) if organ == "brain"
                   else load_organ_folder(d, organ, load_arrays=False))
            if _is_subject(got, organ):
                out.append(name)
        except Exception:
            # one unreadable folder must not hide the rest of the cohort
            continue
    return out


def checks_for(has_map: bool, has_raw: bool, organ: str = "brain") -> list[str]:
    """The check set the supplied inputs justify.

    Running the whole registry regardless is what made a flawless CBF map WARN:
    ten Stream-A checks had nothing to look at. The set follows the inputs in
    BOTH directions, which is the point - a raw-only upload must not be asked for
    a CBF map either, and Stream A grades the acquisition without one.

    Lives here rather than in web.py because the upload console, the cohort
    dashboard and the batch loader all have to answer this question the same way.
    """
    from .core.registry import all_checks

    if has_map and has_raw:
        # 4.1 needs an ASL mask AND a structural mask, and no loader in the
        # package produces either, so including it only ever reports a missing
        # check that no upload can supply.
        return [n for n in all_checks(organ) if n != "4.1.coregistration"]
    if has_raw:
        return [n for n, e in all_checks(organ).items() if e.get("stream") == "A"]
    return cbf_map_checks(organ)


def grade_folder(folder: str, cfg: QCConfig | None = None,
                 checks: list[str] | None = None) -> list[Subject]:
    """Grade every subject subfolder under `folder`. Each immediate subdirectory
    holding something gradeable becomes one subject.

    With `checks` left as None the set is chosen PER SUBJECT from what that
    subject's folder actually contains, via `checks_for`. A folder of bare CBF
    maps gets the CBF-map stream, because asking it for an M0 would only add
    UNKNOWNs and drag every subject to WARN; a folder that also holds the raw
    acquisition gets Stream A as well, because the inputs are there.

    Pass an explicit list to force one set across the whole cohort - which is
    what you want when the cohort must be compared column by column.
    """
    cfg = cfg or QCConfig()
    organ = getattr(cfg, "organ", "brain") or "brain"
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)
    subjects: list[Subject] = []
    for name in sorted(os.listdir(folder)):
        sub_dir = os.path.join(folder, name)
        if not os.path.isdir(sub_dir):
            continue
        inputs = subject_inputs(sub_dir, organ)
        if inputs is None:
            continue
        map_key = {"brain": "cbf", "kidney": "rbf_map", "placenta": "perfusion_map"}[organ]
        use = checks if checks is not None else checks_for(
            inputs.get(map_key) is not None, _has_raw(inputs), organ)
        subjects.append(Subject(sid=name, report=run_qc(inputs, cfg=cfg, checks=use),
                                inputs=inputs, cfg=cfg))
    return subjects


def demo_cohort(n: int = 14, cfg: QCConfig | None = None) -> list[Subject]:
    """A synthetic cohort with a realistic spread of quality, so the dashboard is
    populated for demos and tests. Quality is deterministic per index (no RNG
    seeding surprises)."""
    from .synth import synthetic_case

    cfg = cfg or QCConfig()
    # a repeating pattern that yields a believable pass/warn/fail mix
    pattern = ["clean", "clean", "borderline", "clean", "garbage",
               "clean", "borderline", "clean", "clean", "garbage",
               "borderline", "clean", "garbage", "clean"]
    checks = cbf_map_checks()          # a CBF-map cohort with no T1
    subjects: list[Subject] = []
    for i in range(n):
        quality = pattern[i % len(pattern)]
        c = synthetic_case(quality=quality, seed=i)
        inputs = {"cbf": c.cbf, "gm": c.gm, "wm": c.wm, "csf": c.csf,
                  "brain": c.brain, "voxel_mm": c.voxel_mm}
        sid = f"sub-{i + 1:02d}"
        subjects.append(Subject(sid=sid, report=run_qc(inputs, cfg=cfg, checks=checks),
                                inputs=inputs, cfg=cfg))
    return subjects


# --------------------------------------------------------------------------- #
# aggregate statistics
# --------------------------------------------------------------------------- #
@dataclass
class BatchSummary:
    total: int
    counts: dict[str, int]                 # verdict -> n subjects
    rates: dict[str, float]                # verdict -> fraction
    artifact_breakdown: list[tuple[str, int]]   # (check label, n subjects flagged), worst first

    @property
    def pass_rate(self) -> float:
        return self.rates.get("PASS", 0.0)

    @property
    def warn_rate(self) -> float:
        return self.rates.get("WARN", 0.0)

    @property
    def fail_rate(self) -> float:
        return self.rates.get("FAIL", 0.0)


def cfg_from_params(base: QCConfig, params: dict) -> QCConfig:
    """Build an effective config from the base config plus frontend overrides.

    `params` is a flat {name: str} map (query string). Population resets the CBF
    bands, then any tunable threshold override is applied on top. Unknown or
    unparseable values are ignored rather than raising, so a hand-edited URL can
    never take the server down."""
    pop = params.get("population")
    cfg = for_population(pop) if pop in POPULATIONS else replace(base)
    kw: dict = {}
    for name in TUNABLE:
        if params.get(name) not in (None, ""):
            try:
                val = float(params[name])
            except (TypeError, ValueError):
                continue
            # nan and inf parse fine and then make every comparison False, so a
            # clean scan falls through to FAIL against a cut-off of nan. This
            # path is server-wide, so one such value would poison the cohort.
            if math.isfinite(val):
                kw[name] = val
    strict = params.get("strict")
    if strict is not None:
        kw["strict"] = strict in ("on", "1", "true", "True")
    return replace(cfg, **kw)


def regrade(subjects: list[Subject], cfg: QCConfig,
            checks: list[str] | None = None) -> list[Subject]:
    """Re-grade already-loaded subjects with a new config, reusing their inputs
    (no disk I/O). This is what makes the frontend threshold panel live."""
    if checks is None:
        checks = cbf_map_checks()
    return [Subject(sid=s.sid, report=run_qc(s.inputs, cfg=cfg, checks=checks),
                    inputs=s.inputs, cfg=cfg) for s in subjects]


def summarise(subjects: list[Subject]) -> BatchSummary:
    """Cohort-level statistics for the overview page."""
    total = len(subjects)
    counts: dict[str, int] = {}
    flagged: dict[str, int] = {}
    for s in subjects:
        counts[s.overall] = counts.get(s.overall, 0) + 1
        # count each check that FAILed or WARNed, once per subject
        seen = set()
        for r in s.report.results:
            if r.verdict in (Verdict.FAIL, Verdict.WARN) and r.check not in seen:
                seen.add(r.check)
                flagged[r.check] = flagged.get(r.check, 0) + 1
    rates = {v: (counts.get(v, 0) / total if total else 0.0)
             for v in ("PASS", "WARN", "FAIL")}
    breakdown = sorted(((check_label(c), n) for c, n in flagged.items()),
                       key=lambda kv: (-kv[1], kv[0]))
    return BatchSummary(total=total, counts=counts, rates=rates,
                        artifact_breakdown=breakdown)
