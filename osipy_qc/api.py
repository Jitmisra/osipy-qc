"""
JSON API payloads for the web frontend.

Why this module exists
----------------------
`QCReport.to_dict()` is the source of truth for a graded scan, and the design has
always been that the UI is a *dumb consumer* of that JSON. This module is the
adapter: it shapes the core objects (Subject, BatchSummary, QCConfig) into plain
JSON-serialisable dicts that a browser client can render, and it renders the
figures on demand as bytes.

It deliberately adds no science. Every number here already came out of a check;
this module only labels, groups and serialises. That keeps the QC logic in
`checks/` and the presentation concerns out of it.

Pure stdlib + NumPy, like the rest of the package - no web framework.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields

import numpy as np

from .batch import TUNABLE_GROUPS, BatchSummary, Subject, check_label
from .core.config import (POPULATIONS, THRESHOLD_PROVENANCE, QCConfig,
                          provenance_of)
from .core.registry import all_checks
from .utils.imaging import encode_png, histogram_svg, slice_mosaic
from .utils.masks import covered_tissue_mask

#: Stream id -> the human name used in the UI.
STREAM_TITLES = {"B": "CBF map", "A": "Raw acquisition", "?": "Other"}

#: Verdict severity, worst first - the UI sorts with this.
SEVERITY = {"FAIL": 0, "WARN": 1, "INFO": 2, "PASS": 3, "UNKNOWN": 4, "N/A": 5}


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    m = n // 2
    return xs[m] if n % 2 else (xs[m - 1] + xs[m]) / 2


def _num(v):
    """JSON-safe number: NumPy scalars -> Python, non-finite -> None."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return v if np.isfinite(v) else None
    return v


def _jsonable(v):
    """Recursively make a metric value JSON-safe."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in list(v)]
    if isinstance(v, (bool, str)) or v is None:
        return v
    return _num(v)


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def _streams() -> dict[str, str]:
    return {name: (entry.get("stream") or "?") for name, entry in all_checks().items()}


def check_payload(res: dict, streams: dict[str, str] | None = None) -> dict:
    """One check result, labelled and grouped for display."""
    streams = streams if streams is not None else _streams()
    cid = res["check"]
    verdict = res["verdict"]
    return {
        "id": cid,
        "label": check_label(cid),
        "verdict": verdict,
        "severity": SEVERITY.get(verdict, 9),
        "stream": streams.get(cid, "?"),
        "streamTitle": STREAM_TITLES.get(streams.get(cid, "?"), "Other"),
        # a verdict decided by an uncalibrated cutoff: shown as provisional, never
        # as a hard evidence-backed failure
        "provisional": bool(res.get("provisional")) and verdict in ("FAIL", "WARN"),
        "reason": res.get("reason", ""),
        "metric": _jsonable(res.get("metric") or {}),
    }


# --------------------------------------------------------------------------- #
# subject
# --------------------------------------------------------------------------- #
def _kpis(by: dict[str, dict], cfg: QCConfig) -> list[dict]:
    """The headline numbers.

    Each carries the verdict colour of its own check plus the range it is graded
    against, so the UI can show *where* a value sits rather than just what it is:
      band - the expected/good range for this population
      axis - the scale to draw that range on, so a wildly out-of-band value is
             visibly pinned past the end instead of silently rescaling the meter
    """
    out: list[dict] = []

    def add(check, key, label, unit, fmt, foot, scale=1.0, band=None, axis=None):
        m = by.get(check, {}).get("metric") or {}
        v = m.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return
        out.append({
            "key": key, "label": label, "value": _num(float(v) * scale),
            "unit": unit, "format": fmt, "foot": foot,
            "verdict": by.get(check, {}).get("verdict", "UNKNOWN"),
            "check": check,
            "band": [_num(band[0]), _num(band[1])] if band else None,
            "axis": [_num(axis[0]), _num(axis[1])] if axis else None,
        })

    add("1.qei", "qei", "QEI", "", "0.000", f"cut-off {cfg.qei_warn:g}",
        band=(cfg.qei_warn, 1.0), axis=(0.0, 1.0))
    add("3.1.cbf_level", "mean_gm_cbf", "GM CBF", "mL/100g/min", "0.0",
        f"{cfg.population} band {cfg.gm_cbf_lo:g}–{cfg.gm_cbf_hi:g}",
        band=(cfg.gm_cbf_lo, cfg.gm_cbf_hi), axis=(0.0, cfg.gm_cbf_fail_hi))
    add("3.2.gm_wm_ratio", "gm_wm_ratio", "GM / WM ratio", "", "0.00",
        f"expected above {cfg.ratio_pass:g}",
        band=(cfg.ratio_pass, 4.0), axis=(0.0, 5.0))
    add("2.1.spatial_cov", "sCoV_pct", "Spatial CoV", "%", "0.0",
        f"vascular above {cfg.scov_vascular * 100:g}%",
        band=(0.0, cfg.scov_vascular * 100), axis=(0.0, cfg.scov_artifact * 100))
    add("4.2.coverage", "gm_coverage", "GM coverage", "%", "0",
        f"of the ROI imaged (want ≥ {cfg.coverage_warn * 100:g}%)", scale=100.0,
        band=(cfg.coverage_warn * 100, 100.0), axis=(0.0, 100.0))
    return out


def _figure_list(inputs: dict) -> list[dict]:
    """Which figures can be rendered for this subject, given its inputs."""
    figs = []
    if inputs.get("cbf") is not None:
        figs.append({"name": "cbf", "kind": "png", "title": "CBF map",
                     "caption": "Evenly spaced axial slices. Blue marks negative CBF, "
                                "which is non-physical and indicates noise or a subtraction artefact."})
    if inputs.get("cbf") is not None and inputs.get("gm") is not None:
        figs.append({"name": "gm_hist", "kind": "svg", "title": "Grey-matter CBF distribution",
                     "caption": "Shaded band is the expected range for this population."})
        figs.append({"name": "gm_masked", "kind": "png", "title": "CBF inside the GM mask",
                     "caption": "Gaps mean the mask covers voxels the ASL never imaged."})
    if inputs.get("cbf") is not None and inputs.get("wm") is not None:
        figs.append({"name": "wm_hist", "kind": "svg", "title": "White-matter CBF distribution",
                     "caption": "Shaded band is the expected white-matter range."})
    return figs


def subject_payload(subject: Subject) -> dict:
    """Everything the deep-dive view needs for one scan."""
    d = subject.report.to_dict()
    streams = _streams()
    by = {r["check"]: r for r in d["checks"]}
    checks = [check_payload(r, streams) for r in d["checks"]]
    checks.sort(key=lambda c: (c["stream"] != "B", c["severity"], c["id"]))
    drivers = [{"id": c["id"], "label": c["label"], "verdict": c["verdict"]}
               for c in checks if c["verdict"] in ("FAIL", "WARN")]
    return {
        "sid": subject.sid,
        "verdict": d["overall_verdict"],
        "summary": d["summary"],
        "nChecks": d["n_checks"],
        "population": subject.cfg.population,
        "qeiCutoff": _num(subject.cfg.qei_warn),
        "organ": subject.cfg.organ,
        "strict": bool(subject.cfg.strict),
        "drivers": drivers,
        "kpis": _kpis(by, subject.cfg),
        "checks": checks,
        "figures": _figure_list(subject.inputs),
    }


def ledger_row(subject: Subject) -> dict:
    """The compact per-subject row used in the cohort table."""
    q = subject.qei
    flag = subject.primary_artifact
    return {
        "sid": subject.sid,
        "verdict": subject.overall,
        "severity": SEVERITY.get(subject.overall, 9),
        "qei": _num(q) if isinstance(q, (int, float)) else None,
        "flag": flag,
        "incomplete": flag.startswith("incomplete"),
    }


# --------------------------------------------------------------------------- #
# cohort
# --------------------------------------------------------------------------- #
def cohort_payload(subjects: list[Subject], summary: BatchSummary, cfg: QCConfig,
                   dataset: str = "cohort", overrides: dict | None = None) -> dict:
    """The batch overview: headline stats, the ledger, and the per-check matrix."""
    overrides = overrides or {}
    rows = sorted((ledger_row(s) for s in subjects),
                  key=lambda r: (r["severity"], r["qei"] if r["qei"] is not None else 2.0))
    qeis = [r["qei"] for r in rows if r["qei"] is not None]

    # checks x subjects matrix, so the UI can show a systemic problem at a glance
    lut: dict[tuple, str] = {}
    flagged: dict[str, int] = {}
    for s in subjects:
        for r in s.report.results:
            lut[(s.sid, r.check)] = r.verdict.value
            flagged.setdefault(r.check, 0)
            if r.verdict.value in ("FAIL", "WARN"):
                flagged[r.check] += 1
    matrix_checks = sorted(flagged, key=lambda c: (-flagged[c], c))
    matrix = [{
        "id": c, "label": check_label(c), "flagged": flagged[c],
        "cells": [{"sid": r["sid"], "verdict": lut.get((r["sid"], c))} for r in rows],
    } for c in matrix_checks]

    return {
        "dataset": dataset,
        "population": cfg.population,
        "organ": cfg.organ,
        "strict": bool(cfg.strict),
        "customThresholds": bool(overrides),
        "total": summary.total,
        "counts": summary.counts,
        "rates": {k: _num(v) for k, v in summary.rates.items()},
        "medianQei": _num(_median(qeis)) if qeis else None,
        "belowCutoff": sum(1 for q in qeis if q < cfg.qei_warn),
        "nWithQei": len(qeis),
        "qeiCutoff": _num(cfg.qei_warn),
        "flagBreakdown": [{"label": lab, "n": n} for lab, n in summary.artifact_breakdown],
        "subjects": rows,
        "matrix": matrix,
    }


# --------------------------------------------------------------------------- #
# config / provenance
# --------------------------------------------------------------------------- #
def config_payload(cfg: QCConfig, overrides: dict | None = None) -> dict:
    """The tunable thresholds, grouped, each with its provenance so the user can
    see at the point of editing whether a number is published or a default."""
    overrides = overrides or {}
    groups = []
    for title, fieldset in TUNABLE_GROUPS:
        entries = []
        for name, label in fieldset:
            level, citation, note = provenance_of(name)
            entries.append({
                "name": name, "label": label, "value": _num(getattr(cfg, name)),
                "provenance": level.value, "citation": citation, "note": note,
                "overridden": name in overrides,
            })
        groups.append({"title": title, "fields": entries})

    populations = {}
    for pop in POPULATIONS:
        from .core.config import for_population
        p = for_population(pop)
        populations[pop] = {name: _num(getattr(p, name))
                            for _t, fs in TUNABLE_GROUPS for name, _l in fs}

    return {
        "population": cfg.population,
        "populations": sorted(POPULATIONS),
        "populationValues": populations,
        "strict": bool(cfg.strict),
        "groups": groups,
        "hasOverrides": bool(overrides),
    }


def provenance_payload() -> dict:
    """The full threshold-provenance table, for the methodology view."""
    rows = []
    for field, (level, citation, note) in sorted(THRESHOLD_PROVENANCE.items()):
        rows.append({"field": field, "level": level.value,
                     "citation": citation, "note": note})
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["level"]] = counts.get(r["level"], 0) + 1
    return {"rows": rows, "counts": counts, "total": len(rows)}


def checks_catalogue() -> list[dict]:
    """Every registered check: what the toolbox can grade, for the docs view."""
    out = []
    for name, entry in sorted(all_checks().items()):
        stream = entry.get("stream") or "?"
        fn = entry.get("fn")
        doc = (fn.__doc__ or "").strip().split("\n\n")[0] if fn else ""
        out.append({
            "id": name, "label": check_label(name), "stream": stream,
            "streamTitle": STREAM_TITLES.get(stream, "Other"),
            "required": bool(entry.get("required")),
            "description": " ".join(doc.split()),
        })
    return out


# --------------------------------------------------------------------------- #
# figures, rendered on demand
# --------------------------------------------------------------------------- #
def figure_bytes(subject: Subject, name: str) -> tuple[bytes, str]:
    """Render one figure for a subject. Returns (payload, content-type).

    Raises KeyError for an unknown figure so the caller can 404 it.
    """
    inputs, cfg = subject.inputs, subject.cfg
    cbf = inputs.get("cbf")
    if cbf is None:
        raise KeyError(name)

    if name == "cbf":
        return encode_png(slice_mosaic(cbf)), "image/png"

    if name == "gm_masked":
        gm = inputs.get("gm")
        if gm is None:
            raise KeyError(name)
        gm_a, cbf_a = np.asarray(gm, dtype=float), np.asarray(cbf, dtype=float)
        if gm_a.shape != cbf_a.shape:
            raise KeyError(name)
        overlay = np.where(gm_a > cfg.tissue_thresh, cbf_a, 0.0)
        return encode_png(slice_mosaic(overlay)), "image/png"

    if name in ("gm_hist", "wm_hist"):
        tissue = inputs.get("gm" if name == "gm_hist" else "wm")
        if tissue is None:
            raise KeyError(name)
        vals = np.asarray(cbf, dtype=float)[covered_tissue_mask(cbf, tissue, cfg.tissue_thresh)]
        if name == "gm_hist":
            label = f"GM CBF (mL/100g/min) - {cfg.population} band shaded"
            band = [(cfg.gm_cbf_lo, cfg.gm_cbf_hi, "#CFEBDD")]
        else:
            label = "WM CBF (mL/100g/min)"
            band = [(cfg.wm_cbf_lo, cfg.wm_cbf_hi, "#CFEBDD")]
        return histogram_svg(vals, label=label, bands=band).encode("utf-8"), "image/svg+xml"

    raise KeyError(name)


def config_fields() -> list[str]:
    """Every QCConfig field name (used to validate incoming overrides)."""
    return [f.name for f in dataclass_fields(QCConfig)]
