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
from . import __version__
from .core.result import coverage

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
                                  "A FAIL decided by one of these is marked provisional; "
                                  "--no-strict demotes it to a WARN."),
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
    # The verdict alone says "of what could be measured, this is the worst of it"
    # and nothing about how much WAS measured - a PASS over 4 checks and a PASS
    # over 18 are different claims. The JSON and the HTML report have always
    # carried coverage beside the verdict; the terminal did not, which made it
    # the one output that could show a bare PASS on an almost-empty report.
    cov = coverage(report.results)
    if cov["complete"]:
        print(f"\n  coverage: all {cov['graded']} applicable checks were decided.")
    else:
        missing = ", ".join(cov["missing"][:6])
        more = f" (+{len(cov['missing']) - 6} more)" if len(cov["missing"]) > 6 else ""
        print(f"\n  coverage: {cov['graded']} of {cov['total']} applicable checks decided; "
              f"{cov['unknown']} had no input to look at.")
        print(f"            not decided: {missing}{more}")
    print()


def _organ_demo_inputs(organ: str):
    """Inputs + config for `--organ-demo`, from a phantom of known quality.

    Exists so a user can see what an organ's report looks like without owning
    any data of that organ - which matters most for kidney and placenta, where
    openly shareable data barely exists.
    """
    import numpy as np

    from .core.config import QCConfig
    cfg = QCConfig(organ=organ)
    if organ == "kidney":
        from .synth import synthetic_kidney_case
        c = synthetic_kidney_case(quality="clean", seed=0)
        rng = np.random.default_rng(0)
        dm4 = np.stack([c.delta_m + rng.normal(0, 3, c.delta_m.shape) for _ in range(12)], -1)
        return dict(
            rbf_map=c.rbf, m0=c.m0, delta_m=c.delta_m, delta_m_4d=dm4,
            kidney_masks=c.kidney_masks, cortex_masks=c.cortex_masks,
            medulla_masks=c.medulla_masks, units="mL/100g/min", pld_or_ti_s=1.4,
            voxel_mm=(2.0, 2.0, 8.0), affine=np.diag([2.0, 2.0, 8.0, 1.0]),
            m0_type="separate", m0_tr_s=6.0, m0_background_suppression=False,
            m0_labelling_applied=False, breathing_strategy="free breathing",
            readout="3D", files=[{"name": "ASL_RBF.nii.gz"}],
            sidecar={"ArterialSpinLabelingType": "pCASL", "MagneticFieldStrength": 3,
                     "PostLabelingDelay": 1.4},
            context={"labelling": "pCASL", "field_strength_t": 3, "readout": "3D"},
        ), cfg
    if organ == "placenta":
        from .synth import synthetic_placenta_case
        c = synthetic_placenta_case(quality="clean", seed=0)
        rng = np.random.default_rng(0)
        dm4 = np.stack([c.perfusion / 50 + rng.normal(0, 0.3, c.perfusion.shape)
                        for _ in range(16)], -1)
        return dict(
            perfusion_map=c.perfusion, placenta_mask=c.placenta_mask, m0=c.m0,
            delta_m_4d=dm4, asl_source_4d=dm4, declared_units="mL/100g/min",
            quantified=True,
            constants={"lambda": 0.9, "alpha": 0.767, "t1_blood_ms": 1650},
            field_strength_T=3, labelling_scheme="VSASL",
            scheme_params={"cutoff_velocity_cm_s": 1.6, "post_labeling_delay_s": 1.6},
            gestational_age_wk=30, maternal_position="lateral",
            mask_source="manual, single rater", roi_definition="whole placenta",
            m0_labelled=False, m0_background_suppressed=False,
            asl_background_suppressed=True, normalisation_mode="scalar",
            registration_model="non-rigid (DSVR)", tr_s=6.0, m0_tr_s=8.0,
            context={"cohort_comparable": True},
        ), cfg
    from .synth import synthetic_case
    c = synthetic_case(quality="clean", seed=0)
    return dict(cbf=c.cbf, gm=c.gm, wm=c.wm, csf=c.csf, brain=c.brain,
                voxel_mm=c.voxel_mm), cfg


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
    ap.add_argument("--no-strict", action="store_true",
                    help="demote every PROVISIONAL failure to a warning. A provisional FAIL is "
                         "one decided by an uncalibrated cut-off - an engineering default with "
                         "no published derivation. Strict grading is the default; this is the "
                         "escape hatch for a clinical cohort where a guess must not reject a scan.")
    ap.add_argument("--version", action="version",
                    version=f"osipy-qc {__version__}",
                    help="print the version and exit (a report's provenance is only "
                         "meaningful next to the version that produced it)")
    ap.add_argument("--organ", default="brain", choices=["brain", "kidney", "placenta"],
                    help="which organ's check set to run (default: brain). Each organ has "
                         "its own checks: brain 20, kidney 19, placenta 15.")
    ap.add_argument("--organ-demo", metavar="ORGAN",
                    choices=["brain", "kidney", "placenta"],
                    help="run that organ's checks on a synthetic phantom of known quality")
    ap.add_argument("--serve", action="store_true",
                    help="local web UI: upload a single CBF map in the browser")
    ap.add_argument("--dashboard", metavar="FOLDER",
                    help="cohort dashboard: grade every subject subfolder of FOLDER and serve it")
    ap.add_argument("--dashboard-demo", action="store_true",
                    help="cohort dashboard populated with a synthetic demo cohort")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)),
                    help="port for the web UI (default 8000, or $PORT)")
    # Deliberately NOT defaulted from a HOST environment variable. A stray HOST
    # in the shell would have moved the server off loopback, and _host_ok stops
    # enforcing its DNS-rebinding check the moment the bind address is not
    # loopback - so an ambient variable silently downgraded the security posture.
    # Binding publicly must be an explicit act: render.yaml already passes
    # --host 0.0.0.0 on the command line.
    ap.add_argument("--host", default="127.0.0.1",
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

    from dataclasses import replace as _replace

    from .core.config import for_population
    try:
        cfg = _replace(for_population(args.population), organ=args.organ,
                       strict=not args.no_strict)
    except ValueError as exc:
        ap.error(str(exc))
        return 2

    if args.organ_demo:
        demo_inputs, demo_cfg = _organ_demo_inputs(args.organ_demo)
        # the demo builds its own config for the organ, but the user's grading
        # choice still applies - otherwise --no-strict is silently ignored here
        demo_cfg = _replace(demo_cfg, strict=not args.no_strict)
        report = run_qc(demo_inputs, cfg=demo_cfg)
        _print_report(report)
        if args.json:
            print(report.to_json())
        if args.html:
            from .report_html import render_html
            with open(args.html, "w") as fh:
                fh.write(render_html(report, inputs=demo_inputs, cfg=demo_cfg))
            print(f"\nwrote {args.html}")
        return 0

    if args.demo:
        from .synth import synthetic_case  # noqa: F401
        c = synthetic_case(quality="clean", seed=0)
        inputs = {"cbf": c.cbf, "gm": c.gm, "wm": c.wm, "csf": c.csf,
                  "brain": c.brain, "voxel_mm": c.voxel_mm}
        report = run_qc(inputs, cfg=cfg)
    elif args.folder:
        from .io import load_folder, load_organ_folder
        # the organ-aware loader routes masks; the brain loader has no concept
        # of them, so `--organ kidney <folder>` used to silently drop every mask
        # and report UNKNOWN for the checks that needed them
        inputs = (load_organ_folder(args.folder, args.organ) if args.organ != "brain"
                  else load_folder(args.folder))
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
