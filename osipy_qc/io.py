"""
Lightweight NIfTI folder loader: turn a directory of raw .nii/.nii.gz into the
flat `inputs` dict that run_qc consumes. Recurses (handles nested ADNI trees),
needs no BIDS metadata, and loads the 4D ASL series when present so the swap and
motion checks can run.
"""

from __future__ import annotations

import glob
import json
import os

import nibabel as nib
import numpy as np

from .checks.schema import classify_role, detect_dataset
from .utils.masks import check_prob_range


# A NIfTI header declares its shape, and gzip hides how big that will be. A 6 MB
# file of zeros can declare 700^3 and expand to 2.7 GB as float64 — enough to
# take down a small host. The header is readable before any data is touched, so
# the size is checked there rather than discovered by running out of memory.
#
# The ceiling is generous against real data: a 208x300x320 T1 is 160 MB and an
# 80-volume 4D series about 141 MB, both far below it.
MAX_ARRAY_BYTES = 768 * 1024 * 1024


def _load(path: str) -> np.ndarray:
    img = nib.load(path)
    want = int(np.prod(img.shape)) * 8          # float64, what .astype(float) gives
    if want > MAX_ARRAY_BYTES:
        raise ValueError(
            f"{os.path.basename(path)} declares {tuple(int(x) for x in img.shape)}, "
            f"which is {want / 1e9:.1f} GB in memory; the limit is "
            f"{MAX_ARRAY_BYTES / 1e6:.0f} MB.")
    return np.asanyarray(img.dataobj).astype(float)


def _find_niftis(folder: str) -> list[str]:
    out = []
    for root, _dirs, files in os.walk(folder):
        for f in sorted(files):
            if not f.startswith(".") and f.endswith((".nii", ".nii.gz")):
                out.append(os.path.join(root, f))
    return out


def _shared_prefix(a: str, b: str) -> int:
    """Length of the shared leading run of two basenames, case-insensitive.

    BIDS entity naming makes this a pairing key for free: sub-01_run-1_asl.nii.gz
    shares 'sub-01_run-1_' with its own sidecars and only 'sub-01_run-' with the
    other run's."""
    a, b = a.lower(), b.lower()
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _find_sidecars(folder: str, asl_name: str = "", m0_name: str = "") -> tuple[dict, dict, list[str]]:
    """Read the BIDS metadata sitting beside the images.

    The three datasets this loader was first written against had none of this -
    no JSON, no aslcontext - which is why everything downstream was built to
    infer from shape and filename. The OSIPI ASL Challenge data does have it, and
    inference then produced answers that were merely plausible: it read a Philips
    2D acquisition as 3D from the slice thickness, while `MRAcquisitionType: "2D"`
    sat unread in the file next to it.

    So: read what is stated, and let the caller prefer it over the guess.

    Pairing: every candidate is collected, then the one whose name shares the
    longest prefix with the NIfTI actually loaded (`asl_name` / `m0_name`) wins.
    The first version used first-wins for JSON but last-wins for the TSV, and no
    pairing at all - so a recursed folder holding two individually valid subjects
    graded sub-01's image against sub-02's aslcontext and manufactured a 5.2 FAIL
    out of two clean runs. Ties (empty target, equal prefixes) fall back to walk
    order, which is deterministic. Among ASL sidecars, one that actually states
    ArterialSpinLabelingType outranks a name-only match, so a derivative cbf.json
    cannot shadow the real asl.json. M0 sidecars are routed by the same
    classify_role vocabulary as the images - the hand-rolled substring test it
    replaces missed M0.json and calib.json, dropping a stated TR on the floor.

    Returns (asl_sidecar, m0_sidecar, aslcontext_rows). Any of them may be empty -
    absent metadata is still the common case, not an error.
    """
    asl_c: list[tuple[str, dict]] = []
    m0_c: list[tuple[str, dict]] = []
    ctx_c: list[tuple[str, list[str]]] = []

    for root, _dirs, files in os.walk(folder):
        for f in sorted(files):
            if f.startswith("."):
                continue
            path = os.path.join(root, f)
            low = f.lower()
            try:
                if low.endswith(".json"):
                    with open(path) as fh:
                        data = json.load(fh)
                    if not isinstance(data, dict):
                        continue
                    role = classify_role(f)
                    if role == "m0":
                        m0_c.append((f, data))
                    elif role == "asl" or "ArterialSpinLabelingType" in data:
                        asl_c.append((f, data))
                elif "aslcontext" in low and low.endswith(".tsv"):
                    with open(path) as fh:
                        lines = [ln.strip() for ln in fh if ln.strip()]
                    # first line is the column header in a well-formed file
                    if lines and lines[0].lower().startswith("volume_type"):
                        lines = lines[1:]
                    ctx_c.append((f, lines))
            except (OSError, ValueError):
                # a malformed sidecar must degrade to "no metadata", never crash
                # the whole run - the images are still gradeable without it
                continue

    asl_json = max(asl_c, key=lambda c: ("ArterialSpinLabelingType" in c[1],
                                         _shared_prefix(c[0], asl_name)),
                   default=("", {}))[1] if asl_c else {}
    m0_json = max(m0_c, key=lambda c: _shared_prefix(c[0], m0_name),
                  default=("", {}))[1] if m0_c else {}
    rows = max(ctx_c, key=lambda c: _shared_prefix(c[0], asl_name),
               default=("", []))[1] if ctx_c else []
    return asl_json, m0_json, rows


def load_folder(folder: str, load_arrays: bool = True) -> dict:
    """Build an `inputs` dict for run_qc from a folder of NIfTIs."""
    paths = _find_niftis(folder)
    files = []
    by_role: dict[str, list[str]] = {"asl": [], "m0": [], "t1": [], "other": []}
    for p in paths:
        name = os.path.basename(p)
        img = nib.load(p)
        shape = tuple(int(s) for s in img.shape)
        voxel = tuple(round(float(z), 3) for z in img.header.get_zooms()[:3])
        files.append({"name": name, "shape": shape, "voxel_mm": voxel, "path": p})
        by_role[classify_role(name)].append(p)

    context = os.path.basename(folder.rstrip("/"))
    detected = detect_dataset(files, context)
    asl_files = [f for f in files if classify_role(f["name"]) == "asl"]
    m0_files = [f for f in files if classify_role(f["name"]) == "m0"]

    # ---- stated metadata beats inferred metadata --------------------------
    # detect_dataset guesses from shape and filename because that is all the
    # first three datasets offered. Where a sidecar states a field outright,
    # the guess is overwritten - a guess that lands on the right answer is
    # still a guess, and here one landed on the wrong one.
    #
    # Types are enforced field by field, because a wrong type is worse than an
    # absent field: BackgroundSuppression stated as the STRING "no" is truthy,
    # and it silently switched the required swap check off with a reason
    # claiming BS was on. A field of the wrong type degrades to "not stated".
    asl_json, m0_json, aslcontext = _find_sidecars(
        folder,
        asl_name=asl_files[0]["name"] if asl_files else "",
        m0_name=m0_files[0]["name"] if m0_files else "")
    if asl_json:
        for key, field in (("vendor", "Manufacturer"),
                           ("readout", "MRAcquisitionType"),
                           ("labelling", "ArterialSpinLabelingType")):
            val = asl_json.get(field)
            if isinstance(val, str) and val.strip():
                detected[key] = val
        if isinstance(asl_json.get("BackgroundSuppression"), bool):
            detected["background_suppression"] = asl_json["BackgroundSuppression"]
        detected["source"] = "BIDS sidecar"
        if isinstance(asl_json.get("M0Type"), str):
            # BIDS spells it "Separate"/"Included"/"Absent"/"Estimate"
            detected["m0"] = asl_json["M0Type"].lower()
    else:
        detected["source"] = "inferred from NIfTI shape + filenames"

    inputs: dict = {
        "files": files,
        "context": context,
        "detected": detected,
        "structure": detected["structure"],
        "background_suppression": detected["background_suppression"],
        "m0_type": detected["m0"],
    }

    if asl_json:
        inputs["sidecar"] = asl_json
    if m0_json:
        inputs["m0_sidecar"] = m0_json
        # The White Paper TR rule is about the M0's OWN repetition time, so it
        # is read from the M0 sidecar and never from the ASL one.
        tr = m0_json.get("RepetitionTimePreparation", m0_json.get("RepetitionTime"))
        # BIDS allows a per-volume array here; the White Paper rule cares about
        # full relaxation, so the SHORTEST TR is the one to grade.
        if isinstance(tr, list) and tr and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in tr):
            tr = min(tr)
        # bool is an int in Python: "RepetitionTimePreparation": true would
        # otherwise fabricate a confident 1.0 s TR out of a type error
        if isinstance(tr, (int, float)) and not isinstance(tr, bool):
            inputs["m0_tr_s"] = float(tr)
        if isinstance(m0_json.get("BackgroundSuppression"), bool):
            inputs["m0_background_suppression"] = m0_json["BackgroundSuppression"]
    if aslcontext:
        # the rows only - 5.2 derives its counts from them, and only when there
        # is an actual series to hold them against
        inputs["aslcontext_rows"] = aslcontext

    if asl_files:
        inputs["asl_shape"] = asl_files[0]["shape"]
        inputs["voxel_mm"] = asl_files[0]["voxel_mm"]
        if load_arrays and len(asl_files[0]["shape"]) == 4:
            inputs["asl_4d"] = np.asanyarray(nib.load(asl_files[0]["path"]).dataobj).astype(float)
    if m0_files:
        inputs["m0_shape"] = m0_files[0]["shape"]

    return inputs


# --------------------------------------------------------------------------- #
# Stream B: grade a CBF map produced by ANY pipeline (oxford_asl, ASLPrep, ...)
# --------------------------------------------------------------------------- #
def load_cbf_inputs(cbf: str, gm: str | None = None, wm: str | None = None,
                    csf: str | None = None) -> dict:
    """Build a Stream-B `inputs` dict from NIfTI FILE PATHS.

    Pipeline-agnostic: point it at the CBF map and (for the QEI / GM-WM checks)
    the GM/WM/CSF tissue maps **already in the CBF/ASL voxel space**.

        - `cbf` (required): the quantified CBF map (mL/100g/min).
        - `gm`, `wm`: tissue probability maps in ASL space. Needed for QEI,
          GM/WM level, ratio, negative-GM. Without them those checks return UNKNOWN.
        - `csf`: optional; if omitted but gm+wm are given, it's derived as
          clip(1 - gm - wm, 0, 1).

    All maps must share the CBF grid (same shape) — resample upstream if not.
    """
    img = nib.load(cbf)
    inputs: dict = {
        "cbf": _load(cbf),
        "voxel_mm": tuple(round(float(z), 3) for z in img.header.get_zooms()[:3]),
    }
    if gm is not None:
        inputs["gm"] = _load(gm)
    if wm is not None:
        inputs["wm"] = _load(wm)
    if csf is not None:
        inputs["csf"] = _load(csf)

    # validate the grid before deriving anything from it, so a mismatch reports
    # the actionable "resample first" error rather than a broadcast failure
    for key in ("gm", "wm", "csf"):
        if key in inputs and inputs[key].shape != inputs["cbf"].shape:
            raise ValueError(
                f"{key} shape {inputs[key].shape} != cbf shape {inputs['cbf'].shape} — "
                "resample the tissue maps into ASL space first")

    # Checked before the CSF derivation, which is also wrong for out-of-range
    # input: 1 - GM - WM on 0-255 maps clips to zero everywhere.
    for key in ("gm", "wm", "csf"):
        if key not in inputs:
            continue
        ok, pmax = check_prob_range(inputs[key], key)
        if not ok:
            scale = 255 if pmax > 100 else 100
            raise ValueError(
                f"{key} is not a probability map: its maximum is {pmax:g}, not <= 1. "
                f"It looks like a 0-{scale} segmentation — divide by {scale} before "
                "grading. Left as it is, every tissue mask would cover the whole "
                "brain and the score would be quietly wrong.")

    if csf is None and "gm" in inputs and "wm" in inputs:
        # 1 - GM - WM is only CSF *inside the head*. Outside it GM and WM are both
        # 0, so the complement is 1 everywhere in the air, and the QEI would pool
        # the entire background into its within-tissue variance - which deflates
        # the dispersion index and inflates the score. Restrict it to the voxels
        # the CBF map actually covers.
        derived = np.clip(1.0 - inputs["gm"] - inputs["wm"], 0.0, 1.0)
        inputs["csf"] = derived * (np.asarray(inputs["cbf"], dtype=float) != 0)
        inputs["csf_derived"] = True
    return inputs


def grade_cbf(cbf: str, gm: str | None = None, wm: str | None = None,
              csf: str | None = None, cfg=None):
    """Convenience: load CBF (+ tissue) paths and run the full QC in one call.
    Returns a QCReport."""
    from .report import run_qc
    return run_qc(load_cbf_inputs(cbf, gm, wm, csf), cfg=cfg)


def find_oxford_asl(out_dir: str) -> dict:
    """Locate the CBF map + GM/WM partial-volume maps inside an oxford_asl output
    directory (looks in `native_space/`). Filenames vary by FSL version, so the
    returned dict may have None entries — pass explicit paths to load_cbf_inputs
    if a file isn't found. Hand the result to load_cbf_inputs(**paths)."""
    ns = os.path.join(out_dir, "native_space")
    base = ns if os.path.isdir(ns) else out_dir

    def first(*patterns):
        for p in patterns:
            hits = sorted(glob.glob(os.path.join(base, p)))
            if hits:
                return hits[0]
        return None

    return {
        "cbf": first("perfusion_calib.nii.gz", "perfusion_calib.nii", "*perfusion_calib*"),
        "gm": first("pvgm_inasl.nii.gz", "gm_pv_inasl*", "*pvgm*", "*gm*pv*"),
        "wm": first("pvwm_inasl.nii.gz", "wm_pv_inasl*", "*pvwm*", "*wm*pv*"),
    }


def find_aslprep(perf_dir: str) -> dict:
    """Locate the CBF map + tissue probability maps inside an ASLPrep
    derivatives `.../perf/` directory. Tissue probsegs may live in the `anat/`
    derivatives and may need resampling into ASL space — verify the result and
    fall back to explicit paths if needed."""
    def first(*patterns):
        for p in patterns:
            hits = sorted(glob.glob(os.path.join(perf_dir, p)))
            if hits:
                return hits[0]
        return None

    return {
        "cbf": first("*_cbf.nii.gz", "*mean*cbf*.nii.gz", "*_cbf.nii"),
        "gm": first("*label-GM_probseg*", "*GM_probseg*", "*_GM*"),
        "wm": first("*label-WM_probseg*", "*WM_probseg*", "*_WM*"),
        "csf": first("*label-CSF_probseg*", "*CSF_probseg*"),
    }
