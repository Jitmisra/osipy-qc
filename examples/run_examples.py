"""
End-to-end examples for osipy-qc — runs entirely on bundled/synthetic data.

    python examples/run_examples.py

1. Grade synthetic clean / borderline / garbage CBF maps (in memory).
2. Grade the bundled example_data/ CBF map with its tissue maps (full Stream B).
"""

from __future__ import annotations

import os

from osipy_qc import grade_cbf, run_qc
from osipy_qc.synth import synthetic_case


def grade_synthetic():
    print("\n========== SYNTHETIC CBF MAPS (quality is known) ==========")
    for quality in ("clean", "borderline", "garbage"):
        c = synthetic_case(quality=quality, seed=0)
        report = run_qc({"cbf": c.cbf, "gm": c.gm, "wm": c.wm, "csf": c.csf,
                         "brain": c.brain, "voxel_mm": c.voxel_mm})
        qei = next(r for r in report.results if r.check == "1.qei")
        print(f"  {quality:10s} overall={report.overall.value:5s}  QEI={qei.metric['qei']}")


def grade_example():
    here = os.path.dirname(__file__)
    d = os.path.normpath(os.path.join(here, "..", "example_data"))
    cbf = os.path.join(d, "example_cbf.nii.gz")
    if not os.path.exists(cbf):
        print(f"\n(example_data not found at {d} — skipping)")
        return
    print("\n========== BUNDLED EXAMPLE (from files, full Stream B) ==========")
    report = grade_cbf(cbf,
                       gm=os.path.join(d, "example_gm.nii.gz"),
                       wm=os.path.join(d, "example_wm.nii.gz"),
                       csf=os.path.join(d, "example_csf.nii.gz"))
    print(f"  overall = {report.overall.value}")
    for r in report.results:
        if r.verdict.value != "UNKNOWN":
            print(f"    {r.check:18s} {r.verdict.value:5s} {r.reason}")


if __name__ == "__main__":
    grade_synthetic()
    grade_example()
