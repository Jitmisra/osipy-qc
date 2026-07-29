"""
Lightweight NIfTI folder loader: turn a directory of raw .nii/.nii.gz into the
flat `inputs` dict that run_qc consumes. Recurses (handles nested ADNI trees),
needs no BIDS metadata, and loads the 4D ASL series when present so the swap and
motion checks can run.
"""

from __future__ import annotations

import glob
import os

import nibabel as nib
import numpy as np

from .checks.schema import classify_role, detect_dataset


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

    inputs: dict = {
        "files": files,
        "context": context,
        "detected": detected,
        "structure": detected["structure"],
        "background_suppression": detected["background_suppression"],
        "m0_type": detected["m0"],
    }

    asl_files = [f for f in files if classify_role(f["name"]) == "asl"]
    m0_files = [f for f in files if classify_role(f["name"]) == "m0"]
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
