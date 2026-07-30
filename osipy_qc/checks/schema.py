"""
Module 5 (+ 8.2) — Schema, data-type detection, and control/label integrity.

8.2 data-type detection  — infer vendor / 2D-3D / structure / M0 / BS from the
    NIfTI shapes + filenames when there is NO BIDS metadata (the real-data case).
5.1 BIDS schema          — validate a sidecar if present; otherwise degrade to inference.
5.2 volume/pair integrity— a control/label series must have an even number of volumes.
5.3 control/label swap   — control should be brighter than label (BS OFF).

The detector is a clean port of scripts/inspect_asl.py, operating on a list of
file descriptors so it is unit-testable without touching disk.
"""

from __future__ import annotations

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict


# --------------------------------------------------------------------------- #
# 8.2  data-type detection (pure inference from shape + filename)
# --------------------------------------------------------------------------- #
# The filename vocabulary, as one ordered table rather than a chain of ifs.
# The table exists because the upload page has to apply the SAME rules, and the
# hand-written second copy it used to carry had drifted: the page told the reader
# calib.nii.gz was an M0 while this function returned "other" and load_folder
# silently dropped the file. web.py now generates its rules from
# `role_vocabulary()`, so there is one vocabulary and it cannot drift again.
#
# "contains" = the token appears anywhere in the name; "starts" = the name begins
# with it. Two orderings are load-bearing:
#   * ASL before T1, so 'PCASL_T1corrected' is ASL — the modality token wins over
#     a stray 't1'.
#   * ASL before calib, and calib only as a PREFIX, because oxford_asl uses a
#     trailing _calib to mean "calibrated": perfusion_calib and aCBV_calib are
#     derived output maps, while a file NAMED calib/calibration is the calibration
#     (M0) scan itself.
_ROLE_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("m0",  "contains", ("m0",)),
    ("asl", "contains", ("pcasl", "pasl", "asl", "perf", "cbf", "deltam", "delta_m",
                         "pair", "control", "ctrl", "label", "tag", "subtract")),
    ("asl", "starts",   ("con", "tag", "diff", "dm")),
    ("m0",  "starts",   ("calib",)),
    ("t1",  "contains", ("mprage", "t1", "anat", "struct")),
)


def classify_role(filename: str) -> str:
    """Infer a file's role from its name: one of m0 / asl / t1 / other.

    Applies `_ROLE_RULES` in order; see the comment there for why the order and
    the contains/starts distinction matter."""
    name = filename.lower()
    for role, how, tokens in _ROLE_RULES:
        if any(name.startswith(t) if how == "starts" else t in name for t in tokens):
            return role
    return "other"


def role_vocabulary() -> list[dict]:
    """`_ROLE_RULES` as JSON-serialisable data.

    This is how the upload page gets the rules: it applies exactly these instead
    of a hand-written copy, and it lists them to the reader from the same source.
    """
    return [{"role": role, "how": how, "tokens": list(tokens)}
            for role, how, tokens in _ROLE_RULES]


def guess_vendor(text: str) -> str:
    t = text.lower()
    # 'siemens'/'philips' are long and safe as substrings (and often concatenated,
    # e.g. 'Siemens2DPCASL'). 'ge' is only 2 chars, so match it as a WHOLE TOKEN to
    # avoid false positives inside 'imaGE', 'stoRAGE', 'GEneral', etc.
    if "siemens" in t:
        return "Siemens"
    if "philips" in t:
        return "Philips"
    tokens = t.replace("_", " ").replace("/", " ").replace("-", " ").split()
    if "ge" in tokens:
        return "GE"
    return "unknown"


def guess_readout(text: str, slice_mm: float | None) -> str:
    t = text.lower()
    if "3d" in t:
        return "3D"
    if "2d" in t:
        return "2D"
    if slice_mm is not None:
        return "2D" if slice_mm >= 5.0 else "3D"
    return "unknown"


def guess_background_suppression(text: str):
    t = text.lower().replace("_", " ")
    if "bs" in t.split() or "bs3d" in text.lower() or "_bs" in text.lower():
        return True
    return None  # unknown, never a confident False without metadata


def detect_dataset(files: list[dict], context: str = "") -> dict:
    """files: list of {"name": str, "shape": tuple, "voxel_mm": tuple(optional)}.
    `context` is extra text to mine (e.g. the folder name, where the vendor often lives)."""
    roles = {"asl": [], "m0": [], "t1": [], "other": []}
    for f in files:
        roles[classify_role(f["name"])].append(f)

    context = context + " " + " ".join(f["name"] for f in files)
    asl = roles["asl"][0] if roles["asl"] else None
    slice_mm = asl.get("voxel_mm", (None, None, None))[2] if asl and "voxel_mm" in asl else None

    if asl is None:
        structure, n_vol = "unknown", 0
    elif len(asl["shape"]) == 3:
        structure, n_vol = "pre-subtracted deltaM", 1
    else:
        n_vol = asl["shape"][3]
        structure = f"control/label series ({n_vol} volumes)"

    return {
        "vendor": guess_vendor(context),
        "readout": guess_readout(context, slice_mm),
        "structure": structure,
        "n_volumes": n_vol,
        "m0": "separate" if roles["m0"] else "absent",
        "background_suppression": guess_background_suppression(context),
        "t1_structural": bool(roles["t1"]),
    }


@register_qc_check("8.2.data_type", stream="A", required=False)
def data_type_check(files=None, context: str = "", **_) -> CheckResult:
    """Routing/INFO check: classify the dataset so later checks can be gated."""
    if not files:
        return CheckResult("8.2.data_type", Verdict.UNKNOWN, reason="no files to inspect")
    det = detect_dataset(files, context)
    return CheckResult("8.2.data_type", Verdict.INFO, metric=det,
                       reason=f"{det['vendor']} {det['readout']} {det['structure']}")


# --------------------------------------------------------------------------- #
# 5.1  BIDS schema (graceful degradation)
# --------------------------------------------------------------------------- #
@register_qc_check("5.1.schema", stream="A", required=True)
def schema_check(sidecar: dict | None = None, detected: dict | None = None, **_) -> CheckResult:
    """If a BIDS sidecar exists, validate the required ASL fields; otherwise
    degrade to inference (WARN) rather than hard-failing."""
    required = ["ArterialSpinLabelingType", "MRAcquisitionType", "PostLabelingDelay"]
    if sidecar:
        missing = [k for k in required if k not in sidecar]
        if missing:
            return CheckResult("5.1.schema", Verdict.WARN,
                               metric={"missing_fields": missing},
                               reason=f"sidecar present but missing {missing}")
        return CheckResult("5.1.schema", Verdict.PASS, reason="BIDS sidecar valid")
    # no sidecar
    if detected:
        return CheckResult("5.1.schema", Verdict.WARN, metric={"inferred": detected},
                           reason="no BIDS sidecar - fields inferred from NIfTI + filenames")
    return CheckResult("5.1.schema", Verdict.UNKNOWN, reason="no sidecar and nothing to infer from")


# --------------------------------------------------------------------------- #
# 5.2  volume / pair integrity
# --------------------------------------------------------------------------- #
@register_qc_check("5.2.volume_integrity", stream="A", required=True)
def volume_integrity_check(asl_4d=None, n_volumes=None, structure=None, **_) -> CheckResult:
    """A control/label series must have an even number of volumes."""
    if structure and "pre-subtracted" in structure:
        return CheckResult("5.2.volume_integrity", Verdict.NA,
                           reason="pre-subtracted image has no control/label pairs")
    if n_volumes is None and asl_4d is not None:
        arr = np.asarray(asl_4d)
        n_volumes = arr.shape[3] if arr.ndim == 4 else 1
    if n_volumes is None:
        return CheckResult("5.2.volume_integrity", Verdict.UNKNOWN, reason="volume count unknown")
    if n_volumes < 2:
        return CheckResult("5.2.volume_integrity", Verdict.NA,
                           reason="single volume - no pairs")
    if n_volumes % 2 == 0:
        return CheckResult("5.2.volume_integrity", Verdict.PASS,
                           metric={"n_volumes": int(n_volumes), "n_pairs": int(n_volumes // 2)},
                           reason=f"{n_volumes} volumes -> {n_volumes//2} pairs")
    return CheckResult("5.2.volume_integrity", Verdict.FAIL,
                       metric={"n_volumes": int(n_volumes)},
                       reason=f"odd number of volumes ({n_volumes}) - incomplete pair")


# --------------------------------------------------------------------------- #
# 5.3  control / label swap
# --------------------------------------------------------------------------- #
@register_qc_check("5.3.swap", stream="A", required=True)
def swap_check(asl_4d=None, background_suppression=None, structure=None,
               cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Even volumes (assumed control) should be brighter than odd (label). N/A if
    background suppression is ON or the data is pre-subtracted."""
    if background_suppression:
        return CheckResult("5.3.swap", Verdict.NA,
                           reason="background suppression on - intensity logic does not apply")
    if structure and "pre-subtracted" in structure:
        return CheckResult("5.3.swap", Verdict.NA, reason="no control/label series")
    if asl_4d is None:
        return CheckResult("5.3.swap", Verdict.UNKNOWN, reason="needs the 4D control/label series")
    arr = np.asarray(asl_4d, dtype=float)
    if arr.ndim != 4 or arr.shape[3] < 2:
        return CheckResult("5.3.swap", Verdict.NA, reason="not a multi-volume series")

    even_slab, odd_slab = arr[..., 0::2], arr[..., 1::2]   # even=control, odd=label
    if not np.any(np.isfinite(even_slab)) or not np.any(np.isfinite(odd_slab)):
        return CheckResult("5.3.swap", Verdict.UNKNOWN,
                           reason="control/label volumes are entirely non-finite (all-NaN?)")
    even = float(np.nanmean(even_slab))        # NaN-robust
    odd = float(np.nanmean(odd_slab))
    denom = (abs(even) + abs(odd)) / 2 or 1.0
    rel = (even - odd) / denom
    metric = {"mean_control": round(even, 2), "mean_label": round(odd, 2),
              "rel_diff_pct": round(rel * 100, 2),
              "assumption": "even=control (no aslcontext.tsv); BS assumed off"}
    if even >= odd:
        return CheckResult("5.3.swap", Verdict.PASS, metric=metric,
                           reason=f"control brighter than label ({rel*100:+.2f}%)")
    return CheckResult("5.3.swap", Verdict.FAIL, metric=metric,
                       reason=f"label brighter than control ({rel*100:+.2f}%) - likely swap")
