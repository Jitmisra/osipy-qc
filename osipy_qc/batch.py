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
TUNABLE: dict[str, str] = {n: lab for _, fields in TUNABLE_GROUPS for n, lab in fields}

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


def stream_b_checks() -> list[str]:
    """The CBF-map QC checks. A cohort of CBF maps is graded against these, so a
    clean map can genuinely PASS rather than being dragged to WARN by the raw-data
    checks that have no input to run on."""
    from .core.registry import all_checks
    return [name for name, e in all_checks().items() if e.get("stream") == "B"]


def cbf_map_checks() -> list[str]:
    """Stream B minus coregistration — the right set for a cohort of CBF maps with
    no T1. Coregistration needs both an ASL and a structural mask, which the CBF
    loader never has, so including it would only add an UNKNOWN and drag every
    subject to WARN. Callers who do have T1 masks can pass `checks=` explicitly."""
    return [c for c in stream_b_checks() if c != "4.1.coregistration"]


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
def _find_cbf_inputs(subject_dir: str) -> dict | None:
    """Locate a CBF map (+ tissue maps) inside a subject folder, tolerant of the
    naming used by oxford_asl / ASLPrep / hand-named files. Returns the inputs
    dict for run_qc, or None if no CBF map is found."""
    from .io import load_cbf_inputs

    def first(*patterns):
        for pat in patterns:
            hits = sorted(glob.glob(os.path.join(subject_dir, "**", pat), recursive=True))
            hits = [h for h in hits if os.path.isfile(h)]
            if hits:
                return hits[0]
        return None

    cbf = first("perfusion_calib.nii*", "*_cbf.nii*", "*perfusion*.nii*", "cbf*.nii*")
    if not cbf:
        return None
    gm = first("pvgm_inasl.nii*", "*pvgm*.nii*", "*label-GM_probseg*.nii*", "*gm*.nii*")
    wm = first("pvwm_inasl.nii*", "*pvwm*.nii*", "*label-WM_probseg*.nii*", "*wm*.nii*")
    csf = first("*pvcsf*.nii*", "*label-CSF_probseg*.nii*", "*csf*.nii*")
    try:
        return load_cbf_inputs(cbf, gm=gm, wm=wm, csf=csf)
    except Exception:
        # a shape mismatch etc. — still grade the CBF map alone
        return load_cbf_inputs(cbf)


def grade_folder(folder: str, cfg: QCConfig | None = None,
                 checks: list[str] | None = None) -> list[Subject]:
    """Grade every subject subfolder under `folder`. Each immediate subdirectory
    that contains a CBF map becomes one subject.

    Defaults to the CBF-map stream (`checks=stream_b_checks()`), since a folder of
    CBF maps has no raw-data inputs — running the raw-data checks would only add
    UNKNOWNs and drag every subject to WARN. Pass `checks=None` explicitly to run
    the whole registry."""
    cfg = cfg or QCConfig()
    if checks is None:
        checks = cbf_map_checks()
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)
    subjects: list[Subject] = []
    for name in sorted(os.listdir(folder)):
        sub_dir = os.path.join(folder, name)
        if not os.path.isdir(sub_dir):
            continue
        inputs = _find_cbf_inputs(sub_dir)
        if inputs is None:
            continue
        subjects.append(Subject(sid=name, report=run_qc(inputs, cfg=cfg, checks=checks),
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
