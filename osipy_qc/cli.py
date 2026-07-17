"""
Command-line entry point.

    python -m osipy_qc <folder>            # QC a folder of raw NIfTIs (Stream A)
    python -m osipy_qc <folder> --json     # machine-readable report
    python -m osipy_qc --demo              # run the full Stream B on a synthetic CBF map
"""

from __future__ import annotations

import argparse

from .core.config import Provenance, THRESHOLD_PROVENANCE
from .report import run_qc


def _print_provenance() -> None:
    """Where every threshold came from. Answers 'how did you get this number?'
    for each row - including where the honest answer is 'I didn't'."""
    from .core.config import QCConfig

    cfg = QCConfig()
    order = [Provenance.PUBLISHED, Provenance.IMPLEMENTATION, Provenance.UNCALIBRATED]
    header = {
        Provenance.PUBLISHED: "PUBLISHED - a paper states this number for this purpose",
        Provenance.IMPLEMENTATION: "IMPLEMENTATION - reference code uses it; no paper states it",
        Provenance.UNCALIBRATED: ("UNCALIBRATED - our engineering default, NOT calibrated. "
                                  "These never FAIL alone."),
    }
    for level in order:
        rows = [(f, c, n) for f, (lv, c, n) in THRESHOLD_PROVENANCE.items() if lv is level]
        print(f"\n=== {header[level]} ({len(rows)}) ===\n")
        for field, citation, note in rows:
            value = getattr(cfg, field, "?")
            print(f"  {field:22s} = {value}")
            print(f"    source: {citation}")
            print(f"    says  : {note}\n")
    print("Full write-up: THRESHOLD_PROVENANCE.md\n")


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
    ap.add_argument("--provenance", action="store_true",
                    help="print where every threshold came from, then exit")
    ap.add_argument("--html", metavar="PATH",
                    help="write a self-contained visual HTML report to PATH")
    ap.add_argument("--population", default="adult",
                    help="age group for the CBF bands: neonate_term, neonate_preterm, "
                         "infant, child, adolescent, adult, elderly (default: adult)")
    ap.add_argument("--serve", action="store_true",
                    help="start the local web UI: upload a CBF map in the browser")
    ap.add_argument("--port", type=int, default=8000, help="port for --serve (default 8000)")
    ap.add_argument("--no-browser", action="store_true",
                    help="with --serve, do not open a browser window")
    args = ap.parse_args(argv)

    if args.provenance:
        _print_provenance()
        return 0

    if args.serve:
        from .web import serve
        serve(port=args.port, open_browser=not args.no_browser)
        return 0

    from .core.config import for_population
    try:
        cfg = for_population(args.population)
    except ValueError as exc:
        ap.error(str(exc))
        return 2

    if args.demo:
        from .synth import synthetic_case
        c = synthetic_case(quality="clean", seed=0)
        inputs = {"cbf": c.cbf, "gm": c.gm, "wm": c.wm, "csf": c.csf,
                  "brain": c.brain, "voxel_mm": c.voxel_mm}
        report = run_qc(inputs, cfg=cfg)
    elif args.folder:
        from .io import load_folder
        inputs = load_folder(args.folder)
        report = run_qc(inputs, cfg=cfg)
    else:
        ap.error("provide a folder, or use --demo")
        return 2

    if args.json:
        print(report.to_json())
    else:
        _print_report(report)

    if args.html:
        from .report_html import write_html
        write_html(report, args.html, inputs=inputs, cfg=cfg)
        print(f"visual report written to {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
