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


# --------------------------------------------------------------------------- #
# Non-brain organs: kidney and placenta
# --------------------------------------------------------------------------- #
# Kidney and placenta are graded inside masks the caller supplies, so the loader
# has to route those files. There is no BIDS convention for renal or placental
# ASL and no standard mask naming, so this is a filename vocabulary like the
# brain's - kept here as data so the upload page can list exactly what it
# recognises, and so it cannot drift from what the loader actually does.
_ORGAN_MASK_TOKENS: dict[str, tuple[str, ...]] = {
    "cortex": ("cortex", "cortical", "cor_"),
    "medulla": ("medulla", "medullary", "med_"),
    "kidney": ("kidney", "renal", "whole"),
    "placenta": ("placenta", "placental"),
}
_SIDE_TOKENS: dict[str, tuple[str, ...]] = {
    "left": ("left", "_l_", "_lt", "-l-", "lkid", "l_kidney"),
    "right": ("right", "_r_", "_rt", "-r-", "rkid", "r_kidney"),
}


def classify_organ_file(filename: str) -> tuple[str | None, str | None]:
    """Classify one file for a non-brain organ: (role, side).

    role is one of cortex_mask / medulla_mask / kidney_mask / placenta_mask /
    perfusion / m0 / other; side is "left"/"right"/None. Masks are tested BEFORE
    images because a renal dataset routinely ships `cortex_label.nii.gz` and
    `Label_Map_cortex_medulla.nii.gz`, whose names carry no modality token at
    all - reading those as images is how a mask ends up graded as a map.
    """
    low = filename.lower()
    side = None
    for name, tokens in _SIDE_TOKENS.items():
        if any(t in low for t in tokens):
            side = name
            break

    looks_like_mask = any(t in low for t in ("mask", "label", "seg", "roi", "_lbl"))
    # A file naming BOTH structures is a combined label map, not a cortex mask.
    # renaldro ships exactly this: Label_Map_cortex_medulla.nii.gz with 0=background,
    # 1=cortex, 2=medulla. Classifying it as "cortex" would silently grade the
    # medulla voxels as cortex.
    if (any(t in low for t in _ORGAN_MASK_TOKENS["cortex"])
            and any(t in low for t in _ORGAN_MASK_TOKENS["medulla"])):
        return "cortex_medulla_labelmap", side
    for role, tokens in _ORGAN_MASK_TOKENS.items():
        if any(t in low for t in tokens):
            # "cortex"/"medulla"/"placenta" name a structure, so they are masks
            # even without an explicit mask token; "kidney"/"renal" are also used
            # for images (ASL_RBF is not, but kidney_asl.nii.gz is), so those
            # require the mask token.
            if role in ("cortex", "medulla", "placenta") or looks_like_mask:
                return f"{role}_mask", side
    if any(t in low for t in ("rbf", "perfusion", "cbf", "flow", "_pwi")):
        return "perfusion", side
    if any(t in low for t in ("m0", "calib")):
        return "m0", side
    if any(t in low for t in ("asl", "pcasl", "pasl", "fair", "vsasl", "deltam", "control",
                              "label_", "tag")):
        return "asl", side
    return "other", side


def organ_mask_vocabulary() -> dict:
    """The filename rules above, as JSON-serialisable data for the upload page."""
    return {"structures": {k: list(v) for k, v in _ORGAN_MASK_TOKENS.items()},
            "sides": {k: list(v) for k, v in _SIDE_TOKENS.items()}}


def load_organ_folder(folder: str, organ: str, load_arrays: bool = True) -> dict:
    """Build an `inputs` dict for a kidney or placenta folder.

    Masks are REQUIRED for most checks and are never invented here: there is no
    renal or placental equivalent of "just run BET on it", and a mask this
    loader guessed would be a mask nobody could check. A folder with no mask
    loads fine and simply produces UNKNOWNs, which is the honest outcome.
    """
    if organ not in ("kidney", "placenta"):
        return load_folder(folder, load_arrays=load_arrays)

    paths = _find_niftis(folder)
    files, by_role = [], {}
    for p in paths:
        name = os.path.basename(p)
        img = nib.load(p)
        files.append({"name": name, "shape": tuple(int(s) for s in img.shape),
                      "voxel_mm": tuple(round(float(z), 3) for z in img.header.get_zooms()[:3]),
                      "path": p})
        role, side = classify_organ_file(name)
        by_role.setdefault((role, side), []).append(p)

    asl_json, m0_json, _rows = _find_sidecars(folder)
    inputs: dict = {"files": files, "context": os.path.basename(folder.rstrip("/")),
                    "organ": organ}
    if asl_json:
        inputs["sidecar"] = asl_json
    if m0_json:
        inputs["m0_sidecar"] = m0_json
        tr = m0_json.get("RepetitionTimePreparation", m0_json.get("RepetitionTime"))
        if isinstance(tr, list) and tr and all(
                isinstance(x, (int, float)) and not isinstance(x, bool) for x in tr):
            tr = min(tr)
        if isinstance(tr, (int, float)) and not isinstance(tr, bool):
            inputs["m0_tr_s"] = float(tr)

    def _first(role, side=None):
        hits = by_role.get((role, side)) or []
        return hits[0] if hits else None

    def _load_first(role, side=None):
        p = _first(role, side)
        return _load(p) if (p and load_arrays) else None

    if organ == "kidney":
        # A combined label map is unpacked first, so the separate-file route can
        # still override it if both are supplied.
        combined = _load_first("cortex_medulla_labelmap")
        if combined is not None:
            lab = np.squeeze(combined)
            inputs["cortex_masks"] = {"single": lab == 1}
            inputs["medulla_masks"] = {"single": lab == 2}
            inputs["kidney_masks"] = {"single": lab > 0}
            inputs["mask_note"] = ("cortex/medulla taken from a combined label map "
                                   "(1=cortex, 2=medulla); the two kidneys are not "
                                   "separated, so per-kidney checks report one ROI")
        for kind, key in (("kidney_mask", "kidney_masks"), ("cortex_mask", "cortex_masks"),
                          ("medulla_mask", "medulla_masks")):
            masks = {}
            for side in ("left", "right"):
                arr = _load_first(kind, side)
                if arr is not None:
                    masks[side] = arr > 0.5
            if masks:
                inputs[key] = masks
        # `a or b` on ndarrays raises: numpy refuses to reduce an array to a
        # single truth value. Chained fallbacks must test `is not None`.
        perf = _load_first("perfusion")
        if perf is None:
            perf = _load_first("perfusion", "left")
        if perf is not None:
            inputs["rbf_map"] = perf
            inputs["units"] = "mL/100g/min"      # declared by the caller in the UI
    else:
        arr = _load_first("placenta_mask")
        if arr is not None:
            inputs["placenta_mask"] = arr > 0.5
        perf = _load_first("perfusion")
        if perf is not None:
            inputs["perfusion_map"] = perf

    m0 = _load_first("m0")
    if m0 is not None:
        inputs["m0"] = m0
        inputs["m0_type"] = "separate"
    asl_p = _first("asl")
    if asl_p and load_arrays:
        arr = np.asanyarray(nib.load(asl_p).dataobj).astype(float)
        if arr.ndim == 4:
            inputs["asl_4d"] = arr
            inputs["delta_m_4d"] = arr
        inputs["asl_shape"] = tuple(int(s) for s in arr.shape)
    return inputs
