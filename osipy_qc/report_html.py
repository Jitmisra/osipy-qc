"""
The visual report.

Why this exists
---------------
Raised in the 2026-07 review: the CLI prints numbers, and "for somebody who
doesn't know what they're looking at, it would be difficult" — a reviewer needs
to *see* the CBF distribution and the map, not just read a table. The report also
lets the mentors check whether the histograms are being computed correctly.

Design, copied from the tools we were told to look at
----------------------------------------------------
* **JSON is the product; the page renders it.** ExploreASL writes a QC JSON and
  its PDF just iterates that struct. Here `QCReport.to_dict()` is the source of
  truth and this module is a dumb consumer, so a new `@register_qc_check` shows
  up in the report for free.
* **Tissue overlays, not just numbers.** ExploreASL devotes a whole column to
  tissue-over-CBF projections precisely because registration can only be judged
  by eye.
* **Provenance next to every number.** Neither ExploreASL nor ASLPrep does this;
  it is the direct answer to "how did you get this number?".

Self-contained: images are base64 PNG / inline SVG (see utils.imaging), styling
comes from the shared design system (_webassets), so the output is one file you
can email. No matplotlib, no scipy, no nilearn.
"""

from __future__ import annotations

import math

import numpy as np

from ._webassets import BASE_CSS, VERDICT_COLOURS, brand, esc
from .core.config import QCConfig
from .utils.imaging import histogram_svg, png_data_uri, slice_mosaic
from .utils.masks import covered_tissue_mask

# One-line plain-English gloss per overall verdict, shown in the hero.
_OVERALL_GLOSS = {
    "PASS": "This scan looks usable. Every applicable check passed.",
    "WARN": "Usable with caveats — one or more checks flagged something to review.",
    "FAIL": "At least one check found a disqualifying problem. Inspect before using this scan.",
    "UNKNOWN": "Not enough was provided to reach a verdict.",
}

_STREAM_TITLES = {"B": "The CBF map", "A": "The raw acquisition", "?": "Other"}

_REPORT_CSS = """
.wrap{max-width:1080px;margin:0 auto;padding:0 1.5rem}

/* hero */
.hero{border-radius:var(--radius-pane);padding:1.6rem 1.7rem;margin:.4rem 0 1.4rem;position:relative;
  background:var(--surface);border:1px solid var(--line);box-shadow:var(--shadow);overflow:hidden;
  overflow:hidden;border:1px solid var(--line)}
.hero .eyebrow{opacity:.9}
/* A real target: 40px tall, readable at 15px, and above the hero's spine and
   any stacking the surrounding cards create. It was 13px in a 28px pill, which
   is below the size anything should be asked to hit. */
.newscan{position:absolute;top:1rem;right:1.1rem;z-index:2;
  display:inline-flex;align-items:center;gap:.4rem;min-height:40px;
  font-size:.95rem;font-weight:600;line-height:1;
  color:#fff;background:var(--accent-fill);border:1px solid transparent;
  border-radius:100px;padding:0 1.1rem;text-decoration:none;
  box-shadow:0 1px 2px rgba(16,24,40,.16)}
.newscan:hover{background:var(--accent-600);text-decoration:none}
.newscan:active{transform:translateY(1px)}
@media (max-width:640px){.newscan{position:static;margin-bottom:.9rem}}
@media print{.newscan{display:none}}
.hero::before{content:"";position:absolute;inset:0 auto 0 0;width:4px;background:var(--hero-fg,var(--accent))}
.hero h1{font-size:clamp(1.8rem,4vw,2.5rem);margin:.35rem 0 .3rem;font-weight:700}
.hero p{margin:0;font-size:1rem;max-width:60ch}
.hero .chips{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1rem}
.hero .chips .chip{background:rgba(255,255,255,.7);border-color:rgba(0,0,0,.06)}

/* kpi tiles */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.9rem;margin:0 0 .4rem}
.kpi{padding:1rem 1.1rem;display:flex;flex-direction:column;gap:.15rem}
.kpi .label{font-family:var(--mono);font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;color:var(--faint)}
.kpi .value{font-size:1.7rem;font-weight:680;letter-spacing:-.02em}
.kpi .unit{font-size:.8rem;color:var(--muted);font-weight:500}
.kpi .foot{font-size:.74rem;color:var(--muted)}
.kpi.gauge-kpi{align-items:center;text-align:center}
.gauge{width:150px;max-width:100%}

/* gallery */
.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}
figure{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  overflow:hidden;box-shadow:var(--shadow)}
figure img{display:block;width:100%;height:auto;background:#0c0b0a}
figure svg{display:block;width:100%;height:auto;background:var(--surface);padding:.4rem}
figcaption{font-size:.78rem;color:var(--muted);padding:.6rem .8rem;border-top:1px solid var(--line)}
.gallery figure img{cursor:zoom-in}
/* the lightbox is a :target, so it needs no JavaScript and survives being
   saved as a single file and opened offline */
.lb{position:fixed;inset:0;background:rgba(15,14,13,.86);display:none;
  align-items:center;justify-content:center;padding:2rem;z-index:50;cursor:zoom-out}
.lb:target{display:flex}
.lb img{max-width:96vw;max-height:92vh;border-radius:var(--radius-sm);background:#0b0a09}
@media print{.lb{display:none!important}}

/* driving-check chips in the hero */
.drivers{margin-top:1rem;font-size:.82rem;color:var(--muted);display:flex;flex-wrap:wrap;
  gap:.4rem;align-items:center}
.drivers .dchip{display:inline-flex;align-items:center;gap:.35rem;background:var(--surface);
  border:1px solid var(--line);border-radius:100px;padding:.2rem .6rem;font-size:.78rem;
  font-family:var(--mono);color:var(--ink)}
.drivers .dchip:hover{text-decoration:none;border-color:var(--faint)}
.drivers .dchip .dot{width:7px;height:7px;border-radius:50%;flex:none}

/* check cards */
.card.check .head{display:flex;align-items:baseline;gap:.45rem;flex-wrap:wrap}
.bignum{margin-left:auto;font-size:1.55rem;font-weight:650;letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1;white-space:nowrap}
.bignum small{font-size:.62rem;font-weight:500;color:var(--muted);margin-left:.18rem}
.checks{display:grid;gap:.75rem;grid-template-columns:repeat(auto-fill,minmax(340px,1fr))}
.check{display:grid;grid-template-columns:5px 1fr;overflow:hidden;scroll-margin-top:80px}
.check .rail{width:5px}
.check .body{padding:.95rem 1.1rem}
.check .head{display:flex;align-items:center;gap:.55rem;flex-wrap:wrap}
.check .vpill{font-family:var(--mono);font-size:.68rem;font-weight:700;letter-spacing:.04em;
  padding:.16rem .5rem;border-radius:6px;color:#fff}
.check .cname{font-weight:650;font-size:.95rem;letter-spacing:-.01em}
.check .cid{font-family:var(--mono);font-size:.68rem;color:var(--faint)}
.check .prov-tag{font-family:var(--mono);font-size:.62rem;letter-spacing:.03em;text-transform:uppercase;
  color:var(--warn);border:1px dashed var(--warn);border-radius:5px;padding:.08rem .38rem}
.check .reason{color:var(--muted);margin:.4rem 0 0;font-size:.92rem}
.check details{margin-top:.6rem}
.check summary{cursor:pointer;font-size:.76rem;color:var(--accent-600);font-family:var(--mono);
  list-style:none}
.check summary::-webkit-details-marker{display:none}
.check summary::before{content:"▸ ";color:var(--faint)}
.check details[open] summary::before{content:"▾ "}
.mtable{width:100%;border-collapse:collapse;font-size:.8rem;margin-top:.5rem}
.mtable td{padding:.28rem .5rem;border-bottom:1px solid var(--line);vertical-align:top}
.mtable td:first-child{color:var(--muted);font-family:var(--mono);white-space:nowrap;width:38%}
.prov{margin-top:.7rem;display:flex;flex-direction:column;gap:.35rem}
.prov .row{font-size:.76rem;color:var(--muted);line-height:1.4}
.prov code{color:var(--ink);background:var(--well);padding:.05rem .3rem;border-radius:4px}
.scroll{overflow-x:auto}

/* acquisition chips */
.acqrow{display:flex;gap:.5rem;flex-wrap:wrap}
.chip.acq{font-size:.76rem}
.chip.acq b{color:var(--ink);font-weight:650}
.chip.acq.na b{color:var(--faint);font-weight:500}
"""


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


def _by_name(report_dict) -> dict[str, dict]:
    return {r["check"]: r for r in report_dict["checks"]}


def _slug(check: str) -> str:
    """Stable anchor id for a check, so the hero can jump to its card."""
    import re
    return "chk-" + re.sub(r"[^a-z0-9]+", "-", check.lower()).strip("-")


# verdict severity for sorting cards worst-first (triage order)
_SEV = {"FAIL": 0, "WARN": 1, "INFO": 2, "PASS": 3, "UNKNOWN": 4, "N/A": 5}


# --------------------------------------------------------------------------- #
# hero
# --------------------------------------------------------------------------- #
def _hero(d: dict, cfg: QCConfig, served: bool = False) -> str:
    overall = d["overall_verdict"]
    fg, bg = VERDICT_COLOURS.get(overall, ("#8A8079", "#F1ECE4"))
    chips = []
    for v, n in d["summary"].items():
        c, cbg = VERDICT_COLOURS.get(v, ("#8A8079", "#F1ECE4"))
        chips.append(f'<span class="chip"><span class="dot" style="background:{c}"></span>'
                     f'{n} {esc(v)}</span>')
    gloss = _OVERALL_GLOSS.get(overall, "")
    # only meaningful when the report is being served; a saved file has nowhere
    # to go back to, and the link simply will not resolve
    new_scan = ('<a class="newscan" href="/">&#43; Grade a new scan</a>' if served else "")
    drivers = _driver_chips(d)
    # The surface stays neutral and a spine down the edge carries the status.
    # A whole panel flooded with verdict colour competes with the accent the rest
    # of the product is built from, and shouts a fact stated three lines later.
    return (
        f'<div class="hero" style="--hero-fg:{fg}">'
        f'{new_scan}'
        f'<div class="eyebrow" style="color:{fg}">OVERALL VERDICT</div>'
        f'<h1 style="color:{fg}">{esc(overall)}</h1>'
        f'<p>{esc(gloss)}</p>'
        f'<div class="chips">{"".join(chips)}</div>'
        f'{drivers}'
        f'</div>'
    )


def _driver_chips(d: dict) -> str:
    """The checks actually responsible for a non-PASS verdict, as jump-links to
    their cards — so a reviewer sees *what* failed without scrolling."""
    from .batch import check_label

    bad = [r for r in d["checks"] if r["verdict"] in ("FAIL", "WARN")]
    bad.sort(key=lambda r: _SEV.get(r["verdict"], 9))
    if not bad:
        return ""
    chips = []
    for r in bad:
        c = VERDICT_COLOURS.get(r["verdict"], ("#8A8079", ""))[0]
        chips.append(f'<a class="dchip" href="#{_slug(r["check"])}">'
                     f'<span class="dot" style="background:{c}"></span>{esc(check_label(r["check"]))}</a>')
    return f'<div class="drivers"><span>Needs attention:</span>{"".join(chips)}</div>'


# --------------------------------------------------------------------------- #
# QEI gauge (semicircle, hand-drawn SVG)
# --------------------------------------------------------------------------- #
def _gauge_svg(value: float, cutoff: float, colour: str) -> str:
    """A 0..1 semicircle gauge with a tick at the pass cutoff."""
    v = max(0.0, min(1.0, float(value)))
    cx, cy, r = 60.0, 58.0, 46.0

    def pt(t: float) -> tuple[float, float]:
        theta = math.pi * (1.0 - max(0.0, min(1.0, t)))
        return cx + r * math.cos(theta), cy - r * math.sin(theta)

    lx, ly = pt(0.0)
    vx, vy = pt(v)
    tx0, ty0 = pt(cutoff)
    # tick pointing inward at the cutoff
    theta_c = math.pi * (1.0 - cutoff)
    tx1 = cx + (r - 9) * math.cos(theta_c)
    ty1 = cy - (r - 9) * math.sin(theta_c)
    ex, ey = pt(1.0)
    return (
        f'<svg class="gauge" viewBox="0 0 120 72" xmlns="http://www.w3.org/2000/svg" role="img">'
        f'<path d="M{lx:.1f} {ly:.1f} A{r} {r} 0 0 1 {ex:.1f} {ey:.1f}" '
        f'fill="none" stroke="var(--well)" stroke-width="10" stroke-linecap="round"/>'
        f'<path d="M{lx:.1f} {ly:.1f} A{r} {r} 0 0 1 {vx:.1f} {vy:.1f}" '
        f'fill="none" stroke="{colour}" stroke-width="10" stroke-linecap="round"/>'
        f'<line x1="{tx0:.1f}" y1="{ty0:.1f}" x2="{tx1:.1f}" y2="{ty1:.1f}" '
        f'stroke="var(--ink)" stroke-width="1.6"/>'
        f'</svg>'
    )


# --------------------------------------------------------------------------- #
# KPI tiles
# --------------------------------------------------------------------------- #
def _kpi_tiles(by: dict[str, dict], cfg: QCConfig) -> str:
    tiles: list[str] = []

    def colour_of(check: str) -> str:
        v = by.get(check, {}).get("verdict", "UNKNOWN")
        return VERDICT_COLOURS.get(v, ("#8A8079", ""))[0]

    # QEI gauge tile (the anchor metric)
    qei = by.get("1.qei", {}).get("metric", {}).get("qei")
    if isinstance(qei, (int, float)):
        col = colour_of("1.qei")
        tiles.append(
            f'<div class="card kpi gauge-kpi">'
            f'{_gauge_svg(qei, cfg.qei_warn, col)}'
            f'<div class="value num" style="color:{col};margin-top:-.4rem">{qei:.3f}</div>'
            f'<div class="label">QEI</div>'
            f'<div class="foot">cutoff {cfg.qei_warn:g}</div></div>'
        )

    def simple(check: str, key: str, label: str, unit: str = "", fmt="{:.1f}", foot=""):
        m = by.get(check, {}).get("metric", {})
        if key not in m or not isinstance(m[key], (int, float)):
            return
        col = colour_of(check)
        tiles.append(
            f'<div class="card kpi"><div class="label">{esc(label)}</div>'
            f'<div class="value num" style="color:{col}">{fmt.format(m[key])}'
            f'<span class="unit"> {esc(unit)}</span></div>'
            f'<div class="foot">{esc(foot)}</div></div>'
        )

    simple("3.1.cbf_level", "mean_gm_cbf", "GM CBF", "mL/100g/min",
           foot=f"{cfg.population} band {cfg.gm_cbf_lo:g}–{cfg.gm_cbf_hi:g}")
    simple("3.2.gm_wm_ratio", "gm_wm_ratio", "GM / WM ratio", "", "{:.2f}",
           foot="normal ~2–4")
    simple("2.1.spatial_cov", "sCoV_pct", "spatial CoV", "%", "{:.1f}",
           foot="transit-time proxy")
    # coverage as a percent
    cov = by.get("4.2.coverage", {}).get("metric", {}).get("gm_coverage")
    if isinstance(cov, (int, float)):
        col = colour_of("4.2.coverage")
        tiles.append(
            f'<div class="card kpi"><div class="label">GM coverage</div>'
            f'<div class="value num" style="color:{col}">{cov*100:.0f}<span class="unit"> %</span></div>'
            f'<div class="foot">of the ROI imaged by the ASL</div></div>'
        )

    return f'<div class="kpis">{"".join(tiles)}</div>' if tiles else ""


# --------------------------------------------------------------------------- #
# acquisition parameters
# --------------------------------------------------------------------------- #
def _acq_panel(cfg: QCConfig) -> str:
    """The White-Paper QA-checklist acquisition facts. These live in the config but
    default to None (they are absent from NIfTI headers); showing 'not provided'
    honestly explains why quantification-dependent checks degrade."""
    fields = [
        ("PLD", cfg.post_labeling_delay_s, " s"),
        ("Label duration", cfg.label_duration_s, " s"),
        ("Labeling eff. α", cfg.labeling_efficiency, ""),
        ("T1 blood", cfg.t1_blood_s, " s"),
    ]
    chips = []
    for lab, val, unit in fields:
        if isinstance(val, (int, float)):
            chips.append(f'<span class="chip acq">{esc(lab)}: <b>{val:g}{esc(unit)}</b></span>')
        else:
            chips.append(f'<span class="chip acq na">{esc(lab)}: <b>not provided</b></span>')
    return ('<div class="section-title">Acquisition</div>'
            f'<div class="acqrow">{"".join(chips)}</div>')


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _figures(inputs: dict, cfg: QCConfig) -> str:
    cbf = inputs.get("cbf")
    if cbf is None:
        return ""
    gm, wm = inputs.get("gm"), inputs.get("wm")
    figs: list[str] = []

    try:
        figs.append(
            '<figure><img alt="Axial CBF slices, inferno colour map" '
            'src="' + png_data_uri(slice_mosaic(cbf)) + '">'
            '<figcaption>CBF map &mdash; evenly spaced axial slices. '
            'Blue = negative CBF (non-physical &mdash; a noise / subtraction artefact).</figcaption></figure>'
        )
    except Exception as exc:
        figs.append(f'<figure><figcaption>CBF mosaic unavailable: {esc(exc)}</figcaption></figure>')

    if gm is not None:
        try:
            vals = np.asarray(cbf, dtype=float)[covered_tissue_mask(cbf, gm, cfg.tissue_thresh)]
            svg = histogram_svg(vals, label=f"GM CBF (mL/100g/min) - {cfg.population} band shaded",
                                bands=[(cfg.gm_cbf_lo, cfg.gm_cbf_hi, "#CFEBDD")])
            figs.append(
                f"<figure>{svg}<figcaption>Grey-matter CBF distribution. Shaded = expected "
                f"{cfg.gm_cbf_lo:g}–{cfg.gm_cbf_hi:g} for <b>{esc(cfg.population)}</b>. "
                "Blue bars are negative CBF.</figcaption></figure>"
            )
        except Exception as exc:
            figs.append(f'<figure><figcaption>GM histogram unavailable: {esc(exc)}</figcaption></figure>')

        try:
            gm_arr, cbf_arr = np.asarray(gm, dtype=float), np.asarray(cbf, dtype=float)
            if gm_arr.shape == cbf_arr.shape:
                overlay = np.where(gm_arr > cfg.tissue_thresh, cbf_arr, 0.0)
                figs.append(
                    '<figure><img alt="GM-masked CBF" src="' + png_data_uri(slice_mosaic(overlay))
                    + '"><figcaption>CBF inside the GM mask only. Gaps here mean the mask covers '
                      'voxels the ASL never imaged (see the Coverage check).</figcaption></figure>'
                )
        except Exception as exc:
            figs.append(f'<figure><figcaption>overlay unavailable: {esc(exc)}</figcaption></figure>')

    if wm is not None and gm is not None:
        try:
            vals = np.asarray(cbf, dtype=float)[covered_tissue_mask(cbf, wm, cfg.tissue_thresh)]
            figs.append(
                "<figure>" + histogram_svg(vals, label="WM CBF (mL/100g/min)",
                                           bands=[(cfg.wm_cbf_lo, cfg.wm_cbf_hi, "#CFEBDD")])
                + "<figcaption>White-matter CBF distribution. Shaded = expected "
                  f"{cfg.wm_cbf_lo:g}–{cfg.wm_cbf_hi:g}.</figcaption></figure>"
            )
        except Exception as exc:
            figs.append(f'<figure><figcaption>WM histogram unavailable: {esc(exc)}</figcaption></figure>')

    # Give each raster figure a click-to-zoom target. SVG histograms are already
    # vector and scale with the page, so only the PNG mosaics get one.
    import re as _re
    panes: list[str] = []
    for i, fig in enumerate(figs):
        m = _re.search(r'<img([^>]*?)src="([^"]+)"', fig)
        if not m:
            continue
        fid = f"fig{i}"
        figs[i] = fig.replace(m.group(0), f'<a href="#{fid}"><img{m.group(1)}src="{m.group(2)}"></a>', 1)
        figs[i] = figs[i].replace("</a>", "</a>", 1)
        panes.append(f'<a class="lb" id="{fid}" href="#_"><img alt="" src="{m.group(2)}"></a>')

    return ('<div class="section-title">Images</div><div class="gallery">'
            + "".join(figs) + "</div>" + "".join(panes))


# --------------------------------------------------------------------------- #
# check cards
# --------------------------------------------------------------------------- #
# The one number a check produced that is worth reading first, with its unit.
# Same order and same formatting as the React tiles, so a printed report and the
# screen agree on what the headline is.
_HEADLINE: tuple[tuple[str, str], ...] = (
    ("qei", ""), ("sCoV_pct", "%"), ("spatial_snr", ""), ("gm_wm_ratio", ""),
    ("mean_gm_cbf", "mL/100g/min"), ("gm_coverage", "%"), ("negGM_fraction", ""),
    ("dice", ""), ("mean_fwd_mm", "mm"), ("skewness", ""),
)


def _headline(metric: dict) -> tuple[str, str] | None:
    for key, unit in _HEADLINE:
        v = metric.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        shown = v * 100 if key == "gm_coverage" else float(v)
        if abs(shown) >= 100:
            text = f"{round(shown):,}"
        elif abs(shown) >= 1:
            text = f"{shown:.2f}"
        elif abs(shown) >= 0.001:
            text = f"{shown:.3f}"
        elif shown == 0:
            text = "0"
        else:
            text = f"{shown:.1e}"
        return text, unit
    return None


def _check_card(res: dict) -> str:
    from .batch import check_label

    verdict = res["verdict"]
    fg, _bg = VERDICT_COLOURS.get(verdict, ("#8A8079", ""))
    provisional = bool(res.get("provisional")) and verdict in ("FAIL", "WARN")
    metric = res.get("metric") or {}
    rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{esc(_fmt(v))}</td></tr>"
        for k, v in metric.items() if k != "provenance"
    )
    details = ""
    if rows:
        details = (f'<details><summary>details</summary>'
                   f'<div class="scroll"><table class="mtable">{rows}</table></div></details>')
    # a provisional verdict gets a striped rail (vs a solid one) + a dashed tag,
    # so an uncalibrated-driven FAIL never reads as a hard, evidence-backed one.
    rail = (f'background:repeating-linear-gradient(45deg,{fg} 0 4px,transparent 4px 8px)'
            if provisional else f'background:{fg}')
    prov_tag = ('<span class="prov-tag" title="decided by an uncalibrated cutoff">'
                'provisional</span>' if provisional else "")
    big = _headline(metric)
    big_html = (f'<div class="bignum" style="color:{fg}">{esc(big[0])}'
                f'{f"<small>{esc(big[1])}</small>" if big[1] else ""}</div>') if big else ""
    return (
        f'<div class="card check" id="{_slug(res["check"])}">'
        f'<div class="rail" style="{rail}"></div>'
        f'<div class="body"><div class="head">'
        f'<span class="vpill" style="background:{fg}">{esc(verdict)}</span>'
        f'<span class="cname">{esc(check_label(res["check"]))}</span>'
        f'<span class="cid">{esc(res["check"])}</span>{prov_tag}'
        f'{big_html}</div>'
        f'<p class="reason">{esc(res.get("reason", ""))}</p>'
        f'{details}</div></div>'
    )


def _checks_grouped(report, d: dict) -> str:
    from .core.registry import all_checks

    streams = {name: entry.get("stream") or "?" for name, entry in all_checks().items()}
    order = {"B": 0, "A": 1, "?": 2}
    buckets: dict[str, list[dict]] = {}
    for res in d["checks"]:
        buckets.setdefault(streams.get(res["check"], "?"), []).append(res)

    out = []
    for stream in sorted(buckets, key=lambda s: order.get(s, 9)):
        title = _STREAM_TITLES.get(stream, "Other")
        # worst-first within each stream, so failing checks lead (triage order)
        ordered = sorted(buckets[stream], key=lambda r: _SEV.get(r["verdict"], 9))
        cards = "".join(_check_card(r) for r in ordered)
        out.append(f'<div class="section-title">Checks &mdash; {esc(title)}</div>'
                   f'<div class="checks">{cards}</div>')
    return "".join(out)


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
#: appended to any page that includes a report body, so its CSS is available.
REPORT_CSS = _REPORT_CSS


def report_body(report, inputs: dict | None = None, cfg: QCConfig | None = None,
                with_note: bool = True, served: bool = False) -> str:
    """The inner report content (hero + KPIs + figures + grouped checks), without
    the page chrome. Reused by both the standalone report and the dashboard's
    subject deep-dive so there is exactly one report renderer."""
    cfg = cfg or QCConfig()
    inputs = inputs or {}
    d = report.to_dict()
    note = (
        '<div class="note" style="margin-top:1.6rem">A verdict marked '
        '<b>provisional</b> was decided by an <b>uncalibrated</b> cutoff &mdash; an engineering '
        'default with no published derivation. Under strict grading (the default) it can raise a '
        'FAIL; turn strict off to demote it to a WARN. Checks marked <b>N/A</b> or <b>INFO</b> are '
        'excluded from the overall verdict. Full sourcing in THRESHOLD_PROVENANCE.md '
        '(<code>osipy-qc --provenance</code>).</div>'
    ) if with_note else ""
    # Findings before images: on a FAIL scan the checks are the diagnosis and the
    # mosaics are the supporting evidence.
    return (
        f'{_hero(d, cfg, served)}'
        f'{_kpi_tiles(_by_name(d), cfg)}'
        f'{_acq_panel(cfg)}'
        f'{_checks_grouped(report, d)}'
        f'{_figures(inputs, cfg)}'
        f'{note}'
    )


def render_html(report, inputs: dict | None = None, cfg: QCConfig | None = None,
                served: bool = False,
                title: str = "ASL QC report") -> str:
    """Render a QCReport (+ the inputs it graded) as one self-contained HTML page."""
    cfg = cfg or QCConfig()
    meta = (f'population: {esc(cfg.population)} &middot; organ: {esc(cfg.organ)} '
            f'&middot; strict: {esc(cfg.strict)}')
    body = (
        '<div class="topbar">'
        f'{brand("ASL quality control")}'
        f'<div class="spacer"></div><div class="meta mono">{meta}</div></div>'
        f'<div class="wrap">{report_body(report, inputs, cfg, served=served)}</div>'
        '<div class="footer">Generated by osipy-qc &mdash; pure NumPy + nibabel. '
        'Every pixel is drawn from the arrays the checks graded; images encoded with the '
        'standard library only.</div>'
    )
    return ("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{BASE_CSS}{_REPORT_CSS}</style></head>"
            f"<body>{body}</body></html>")


def write_html(report, path: str, inputs: dict | None = None,
               cfg: QCConfig | None = None, title: str = "ASL QC report") -> str:
    """Render and write the report. Returns the path written."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_html(report, inputs=inputs, cfg=cfg, title=title))
    return path
