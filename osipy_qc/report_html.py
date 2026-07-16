"""
The visual report.

Why this exists
---------------
Raised in the 2026-07 review: the CLI prints numbers, and "for somebody who
doesn't know what they're looking at, it would be difficult" — a reviewer needs
to *see* the CBF distribution and the map, not just read a table. The report also
lets the mentors check whether the histograms are being computed correctly at all.

Design, copied from the tools we were told to look at
----------------------------------------------------
* **JSON is the product; the page renders it.** ExploreASL writes
  `QC_collection_<Subject>.json` and its PDF does nothing but iterate that struct.
  We do the same: `QCReport.to_dict()` is the source of truth and this module is a
  dumb consumer, so a new `@register_qc_check` shows up here for free.
* **Tissue overlays, not just numbers.** ExploreASL's report devotes a whole
  column to tissue-over-CBF projections precisely because registration can only
  be judged by eye.
* **Provenance next to every number.** Neither ExploreASL nor ASLPrep does this;
  it is the direct answer to "how did you get this number?".

Self-contained: images are base64 PNG/inline SVG (see utils.imaging), so the
output is one file you can email. No matplotlib, no scipy, no nilearn.
"""

from __future__ import annotations

import html
import numpy as np

from .core.config import Provenance, QCConfig, provenance_of
from .core.result import Verdict
from .utils.imaging import histogram_svg, png_data_uri, slice_mosaic
from .utils.masks import covered_tissue_mask

# Which config threshold(s) each check leans on, so the report can show the
# provenance of the numbers behind a verdict.
_CHECK_THRESHOLDS: dict[str, tuple[str, ...]] = {
    "1.qei": ("qei_pass", "qei_warn", "tissue_thresh", "smooth_fwhm_mm"),
    "2.1.spatial_cov": ("scov_vascular", "scov_artifact"),
    "2.2.snr": ("scov_vascular",),
    "2.3.histogram": (),
    "3.1.cbf_level": ("gm_cbf_lo", "gm_cbf_hi", "gm_cbf_fail_lo", "gm_cbf_fail_hi",
                      "wm_cbf_lo", "wm_cbf_hi"),
    "3.2.gm_wm_ratio": ("ratio_pass", "ratio_min"),
    "3.3.negative_gm": ("neg_gm_warn", "neg_gm_fail"),
    "3.4.deep_gm_ratio": ("deep_gm_ratio_lo", "deep_gm_ratio_hi"),
    "4.1.coregistration": ("dice_pass", "dice_warn"),
    "4.2.coverage": ("coverage_warn", "coverage_fail"),
    "6.2.m0_tr": ("m0_tr_min_s",),
    "7.1.motion": ("fd_mean_fail_mm", "fd_frame_censor_mm"),
}

_COLOUR = {
    "PASS": "#2F9E5B", "WARN": "#B98900", "FAIL": "#C0392B",
    "UNKNOWN": "#8a7d6d", "N/A": "#8a7d6d", "INFO": "#3B6FA8",
}
_PROV_BADGE = {
    Provenance.PUBLISHED: ("published", "#2F9E5B"),
    Provenance.IMPLEMENTATION: ("implementation", "#3B6FA8"),
    Provenance.UNCALIBRATED: ("uncalibrated", "#B98900"),
}

_CSS = """
:root{--paper:#fff;--warm:#FDF9F5;--ink:#241C15;--soft:#6B5D4F;--line:#EDE2D5;--accent:#C2571A;}
*{box-sizing:border-box}
body{margin:0;background:var(--warm);color:var(--ink);
     font-family:ui-serif,"Iowan Old Style",Georgia,serif;line-height:1.5}
.wrap{max-width:1000px;margin:0 auto;padding:2rem 1.25rem 4rem}
h1{font-size:1.7rem;margin:0 0 .25rem}
h2{font-size:1.05rem;margin:2rem 0 .6rem;padding-bottom:.3rem;border-bottom:1px solid var(--line)}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.sub{color:var(--soft);font-size:.9rem;margin:0}
.verdict{display:inline-block;padding:.45rem 1rem;border-radius:4px;color:#fff;
         font-weight:600;letter-spacing:.04em;font-family:ui-monospace,Menlo,monospace}
.card{background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:1rem 1.1rem;margin:.6rem 0}
.head{display:flex;align-items:center;gap:.7rem;flex-wrap:wrap}
.pill{font-family:ui-monospace,Menlo,monospace;font-size:.7rem;letter-spacing:.03em;
      padding:.12rem .5rem;border-radius:3px;color:#fff}
.name{font-family:ui-monospace,Menlo,monospace;font-weight:600}
.reason{color:var(--soft);margin:.4rem 0 0;font-size:.95rem}
table{border-collapse:collapse;width:100%;font-size:.85rem;margin-top:.6rem}
td,th{padding:.3rem .5rem;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--soft);font-weight:400;font-family:ui-monospace,Menlo,monospace;font-size:.7rem;
   text-transform:uppercase;letter-spacing:.06em}
.prov{margin-top:.6rem;font-size:.78rem;color:var(--soft)}
.prov code{color:var(--ink)}
.badge{font-family:ui-monospace,Menlo,monospace;font-size:.62rem;padding:.05rem .35rem;
       border-radius:3px;color:#fff;margin-right:.35rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}
img{max-width:100%;height:auto;border-radius:4px;background:#000;display:block}
figure{margin:0}
figcaption{font-size:.75rem;color:var(--soft);margin-top:.35rem;font-family:ui-monospace,Menlo,monospace}
.note{background:#FBE4D0;border-left:3px solid var(--accent);padding:.7rem .9rem;
      border-radius:0 4px 4px 0;font-size:.88rem;margin:1rem 0}
.legend{font-size:.72rem;color:var(--soft);font-family:ui-monospace,Menlo,monospace}
.scroll{overflow-x:auto}
"""


def _e(s) -> str:
    return html.escape(str(s))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


def _provenance_block(check: str) -> str:
    fields = _CHECK_THRESHOLDS.get(check, ())
    if not fields:
        return ""
    cfg = QCConfig()
    rows = []
    for f in fields:
        level, citation, _note = provenance_of(f)
        label, colour = _PROV_BADGE[level]
        value = getattr(cfg, f, "?")
        src = "no published source" if citation == "NONE" else citation
        rows.append(
            f'<div><span class="badge" style="background:{colour}">{label}</span>'
            f'<code class="mono">{_e(f)} = {_e(value)}</code> &mdash; {_e(src)}</div>'
        )
    return '<div class="prov">' + "".join(rows) + "</div>"


def _check_card(res: dict) -> str:
    verdict = res["verdict"]
    colour = _COLOUR.get(verdict, "#8a7d6d")
    metric = res.get("metric") or {}
    rows = "".join(
        f"<tr><td class='mono'>{_e(k)}</td><td>{_e(_fmt(v))}</td></tr>"
        for k, v in metric.items()
    )
    table = f'<div class="scroll"><table>{rows}</table></div>' if rows else ""
    return (
        '<div class="card">'
        f'<div class="head"><span class="pill" style="background:{colour}">{_e(verdict)}</span>'
        f'<span class="name">{_e(res["check"])}</span></div>'
        f'<p class="reason">{_e(res.get("reason", ""))}</p>'
        f"{table}{_provenance_block(res['check'])}"
        "</div>"
    )


def _figures(inputs: dict, cfg: QCConfig) -> str:
    """CBF mosaic + GM-CBF histogram + tissue overlay, when the inputs allow."""
    cbf = inputs.get("cbf")
    if cbf is None:
        return ""
    gm, wm = inputs.get("gm"), inputs.get("wm")
    figs = []

    # 1. the CBF map itself
    try:
        figs.append(
            '<figure><img alt="CBF slices" src="' + png_data_uri(slice_mosaic(cbf)) + '">'
            '<figcaption>CBF map &mdash; evenly spaced axial slices. '
            'Blue = negative (physically impossible) voxels.</figcaption></figure>'
        )
    except Exception as exc:                       # a figure must never break the report
        figs.append(f'<figcaption>CBF mosaic unavailable: {_e(exc)}</figcaption>')

    # 2. the histogram Maria asked for, with the population's expected band shaded
    if gm is not None:
        try:
            vals = np.asarray(cbf, dtype=float)[covered_tissue_mask(cbf, gm, cfg.tissue_thresh)]
            bands = [(cfg.gm_cbf_lo, cfg.gm_cbf_hi, "#CDEAE5")]
            svg = histogram_svg(vals, label=f"GM CBF (mL/100g/min) - {cfg.population} band shaded",
                                bands=bands)
            figs.append(
                f"<figure>{svg}<figcaption>Grey-matter CBF distribution. Shaded = expected "
                f"{cfg.gm_cbf_lo:g}-{cfg.gm_cbf_hi:g} band for <b>{_e(cfg.population)}</b>. "
                "Blue bars are negative CBF.</figcaption></figure>"
            )
        except Exception as exc:
            figs.append(f'<figcaption>GM histogram unavailable: {_e(exc)}</figcaption>')

    # 3. tissue-over-CBF, so registration/coverage can be judged by eye
    if gm is not None:
        try:
            gm_arr = np.asarray(gm, dtype=float)
            cbf_arr = np.asarray(cbf, dtype=float)
            if gm_arr.shape == cbf_arr.shape:
                overlay = np.where(gm_arr > cfg.tissue_thresh, cbf_arr, 0.0)
                figs.append(
                    '<figure><img alt="GM-masked CBF" src="' + png_data_uri(slice_mosaic(overlay))
                    + '"><figcaption>CBF inside the GM mask only. Gaps here mean the mask '
                      'covers voxels the ASL never imaged (see 4.2.coverage).</figcaption></figure>'
                )
        except Exception as exc:
            figs.append(f'<figcaption>overlay unavailable: {_e(exc)}</figcaption>')

    if wm is not None and gm is not None:
        try:
            vals = np.asarray(cbf, dtype=float)[covered_tissue_mask(cbf, wm, cfg.tissue_thresh)]
            figs.append(
                "<figure>" + histogram_svg(vals, label="WM CBF (mL/100g/min)",
                                           bands=[(cfg.wm_cbf_lo, cfg.wm_cbf_hi, "#CDEAE5")])
                + "<figcaption>White-matter CBF distribution. Shaded = expected "
                  f"{cfg.wm_cbf_lo:g}-{cfg.wm_cbf_hi:g} band.</figcaption></figure>"
            )
        except Exception as exc:
            figs.append(f'<figcaption>WM histogram unavailable: {_e(exc)}</figcaption>')

    return '<h2>Images</h2><div class="grid">' + "".join(figs) + "</div>"


def render_html(report, inputs: dict | None = None, cfg: QCConfig | None = None,
                title: str = "ASL QC report") -> str:
    """Render a QCReport (+ the inputs it graded) as one self-contained HTML page."""
    cfg = cfg or QCConfig()
    inputs = inputs or {}
    d = report.to_dict()
    overall = d["overall_verdict"]
    summary = ", ".join(f"{k} {v}" for k, v in d["summary"].items())

    graded = [r for r in d["checks"] if r["verdict"] not in ("N/A", "INFO")]
    other = [r for r in d["checks"] if r["verdict"] in ("N/A", "INFO")]

    body = [
        '<div class="wrap">',
        f"<h1>{_e(title)}</h1>",
        f'<p class="sub mono">population: {_e(cfg.population)} &middot; organ: {_e(cfg.organ)} '
        f'&middot; strict: {_e(cfg.strict)}</p>',
        f'<p style="margin:1rem 0"><span class="verdict" style="background:'
        f'{_COLOUR.get(overall, "#8a7d6d")}">OVERALL: {_e(overall)}</span></p>',
        f'<p class="sub mono">{_e(summary)}</p>',
        '<div class="note"><b>Reading the verdicts.</b> Any FAIL &rarr; FAIL; otherwise any '
        'WARN/UNKNOWN &rarr; WARN; otherwise PASS. <b>N/A</b> (structurally inapplicable) and '
        '<b>INFO</b> (reported, not graded) are excluded. Each number below carries a badge '
        'saying whether its threshold is <b>published</b>, taken from a reference '
        '<b>implementation</b>, or an <b>uncalibrated</b> engineering default &mdash; '
        'uncalibrated thresholds never drive a FAIL on their own. '
        'See THRESHOLD_PROVENANCE.md.</div>',
        _figures(inputs, cfg),
        "<h2>Graded checks</h2>",
        *[_check_card(r) for r in graded],
    ]
    if other:
        body += ["<h2>Reported, not graded (N/A &amp; INFO)</h2>",
                 *[_check_card(r) for r in other]]
    body.append('<p class="legend">Generated by osipy-qc &mdash; pure NumPy + nibabel; '
                "images encoded from the arrays with the standard library only.</p>")
    body.append("</div>")

    return ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{_e(title)}</title><style>{_CSS}</style></head><body>"
            + "".join(body) + "</body></html>")


def write_html(report, path: str, inputs: dict | None = None,
               cfg: QCConfig | None = None, title: str = "ASL QC report") -> str:
    """Render and write the report. Returns the path written."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(report, inputs=inputs, cfg=cfg, title=title))
    return path
