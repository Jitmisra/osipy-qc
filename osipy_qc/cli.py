"""
Command-line entry point.

    python -m osipy_qc <folder>            # QC a folder of raw NIfTIs (Stream A)
    python -m osipy_qc <folder> --json     # machine-readable report
    python -m osipy_qc --demo              # run the full Stream B on a synthetic CBF map
"""

from __future__ import annotations

import argparse
import os

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
                    help="population for the CBF bands: adult or neonate (default: adult)")
    ap.add_argument("--serve", action="store_true",
                    help="local web UI: upload a single CBF map in the browser")
    ap.add_argument("--dashboard", metavar="FOLDER",
                    help="cohort dashboard: grade every subject subfolder of FOLDER and serve it")
    ap.add_argument("--dashboard-demo", action="store_true",
                    help="cohort dashboard populated with a synthetic demo cohort")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)),
                    help="port for the web UI (default 8000, or $PORT)")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="interface to bind (default 127.0.0.1; use 0.0.0.0 to accept "
                         "traffic from outside this machine)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = ap.parse_args(argv)

    if args.provenance:
        _print_provenance()
        return 0

    if args.dashboard_demo or args.dashboard:
        from .web import serve_dashboard
        open_browser = not args.no_browser
        if args.dashboard_demo:
            from .batch import demo_cohort
            print("grading a synthetic demo cohort...")
            serve_dashboard(demo_cohort(14), dataset="demo_cohort (synthetic)",
                            host=args.host, port=args.port, open_browser=open_browser)
        else:
            from .batch import grade_folder
            print(f"grading subjects under {args.dashboard} ...")
            subs = grade_folder(args.dashboard)
            if not subs:
                ap.error(f"no subject folders with a CBF map found under {args.dashboard!r}")
                return 2
            serve_dashboard(subs, dataset=args.dashboard, host=args.host,
                            port=args.port, open_browser=open_browser)
        return 0

    if args.serve:
        from .web import serve
        serve(host=args.host, port=args.port, open_browser=not args.no_browser)
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
