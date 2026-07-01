#!/usr/bin/env python3
"""
Grade a CBF map with osipy-qc.

    python examples/grade_cbf_map.py example_data/example_cbf.nii.gz \
        --gm example_data/example_gm.nii.gz --wm example_data/example_wm.nii.gz
    python examples/grade_cbf_map.py <your_cbf.nii.gz>            # rough (no tissue maps)

If GM/WM tissue maps are given, the FULL Stream B runs (incl. QEI). If not, a
ROUGH whole-brain mask is used so the level / noise checks still run (the QEI and
GM/WM ratio need real tissue separation, so they're skipped in that mode).
"""

import argparse

import nibabel as nib
import numpy as np

from osipy_qc import run_qc
from osipy_qc.utils.masks import brain_mask_fallback


def _load(p):
    return np.asanyarray(nib.load(p).dataobj).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cbf")
    ap.add_argument("--gm")
    ap.add_argument("--wm")
    ap.add_argument("--csf")
    args = ap.parse_args()

    cbf = _load(args.cbf)
    inputs = {"cbf": cbf}

    if args.gm and args.wm:
        gm, wm = _load(args.gm), _load(args.wm)
        csf = _load(args.csf) if args.csf else np.clip(1 - gm - wm, 0, 1)
        inputs.update(gm=gm, wm=wm, csf=csf)
        note = "full tissue maps -> full Stream B"
        checks = None  # run everything
    else:
        brain = brain_mask_fallback(cbf, 40).astype(float)
        inputs.update(gm=brain, wm=brain)
        note = "ROUGH whole-brain mask (no tissue maps) -> level + noise only"
        checks = ["3.1.cbf_level", "2.1.spatial_cov", "2.3.histogram", "3.3.negative_gm"]

    report = run_qc(inputs, checks=checks)
    print(f"\nCBF map : {args.cbf}")
    print(f"mode    : {note}")
    print(f"OVERALL : {report.overall.value}\n")
    for r in report.results:
        if r.verdict.value != "UNKNOWN":
            print(f"  {r.check:18s} {r.verdict.value:5s}  {r.reason}")
    print()


if __name__ == "__main__":
    main()
