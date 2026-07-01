"""
Command-line entry point.

    python -m osipy_qc <folder>            # QC a folder of raw NIfTIs (Stream A)
    python -m osipy_qc <folder> --json     # machine-readable report
    python -m osipy_qc --demo              # run the full Stream B on a synthetic CBF map
"""

from __future__ import annotations

import argparse

from .report import run_qc


def _print_report(report) -> None:
    bar = {"PASS": "✅", "WARN": "🟠", "FAIL": "❌", "UNKNOWN": "⚪", "N/A": "⊘", "INFO": "ℹ️"}
    print(f"\n=== OVERALL: {bar.get(report.overall.value, '')} {report.overall.value} "
          f"=== ({report.to_dict()['summary']})\n")
    for r in report.results:
        print(f"  {bar.get(r.verdict.value, ' ')} {r.check:22s} {r.verdict.value:8s} {r.reason}")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="osipy_qc", description="ASL QC ToolBox")
    ap.add_argument("folder", nargs="?", help="folder of raw NIfTIs to QC")
    ap.add_argument("--json", action="store_true", help="print the JSON report")
    ap.add_argument("--demo", action="store_true", help="run the full Stream B on synthetic data")
    args = ap.parse_args(argv)

    if args.demo:
        from .synth import synthetic_case
        c = synthetic_case(quality="clean", seed=0)
        inputs = {"cbf": c.cbf, "gm": c.gm, "wm": c.wm, "csf": c.csf,
                  "brain": c.brain, "voxel_mm": c.voxel_mm}
        report = run_qc(inputs)
    elif args.folder:
        from .io import load_folder
        report = run_qc(load_folder(args.folder))
    else:
        ap.error("provide a folder, or use --demo")
        return 2

    if args.json:
        print(report.to_json())
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
