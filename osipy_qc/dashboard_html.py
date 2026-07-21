"""
The cohort dashboard — the proposal's UI, built local and dependency-free.

Views (one design system, matched to the OSIPI QC-ToolBox proposal):
  * a sidebar with the participant list (verdict-coloured dots)
  * a Batch Overview: stat cards, a participant ledger, an artifact breakdown
  * a Subject Deep Dive: the full per-scan report (reused from report_html)

Interactive, still no dependencies:
  * a Thresholds panel that re-grades the whole cohort live (GET /apply)
  * click any image/plot to enlarge it (a pure-JS lightbox)

Server-rendered pages; styling from the shared design system (_webassets).
"""

from __future__ import annotations

import json

from ._webassets import (BASE_CSS, PROVENANCE_COLOURS, VERDICT_COLOURS, brand,
                         esc)
from .batch import TUNABLE, TUNABLE_GROUPS, BatchSummary, Subject
from .core.config import POPULATIONS, QCConfig, for_population, provenance_of
from .report_html import REPORT_CSS, report_body

# verdict severity for worst-first display ordering (triage)
_SEV = {"FAIL": 0, "WARN": 1, "PASS": 2, "UNKNOWN": 3, "N/A": 4, "INFO": 5}


def _worst_first(subjects: list[Subject]) -> list[Subject]:
    """A display copy sorted worst-first (FAIL→WARN→PASS), then lowest QEI first.
    The batch engine's own order is left untouched."""
    def key(s: Subject):
        q = s.qei
        return (_SEV.get(s.overall, 9), q if isinstance(q, (int, float)) else 2.0)
    return sorted(subjects, key=key)

_POP_ORDER = ["neonate_preterm", "neonate_term", "infant", "child",
              "adolescent", "adult", "elderly"]

# Effective value of every tunable field, per population. Embedded as JSON so the
# drawer can repopulate the number inputs the instant a population is picked -
# without this, the fields keep the old bands and the stale values get submitted
# as overrides, clobbering the population reset on the server. for_population(p)
# gives QCConfig defaults with that population's CBF bands laid on top.
_POP_FIELD_VALUES = {p: {name: getattr(for_population(p), name) for name in TUNABLE}
                     for p in _POP_ORDER}
_POP_VALUES_JSON = json.dumps(_POP_FIELD_VALUES)

_DASH_CSS = """
body{display:flex;min-height:100vh}
.side{width:230px;flex:none;background:var(--surface);border-right:1px solid var(--line);
  display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto}
.side .side-brand{padding:1.15rem 1.2rem;border-bottom:1px solid var(--line)}
.side .nav{padding:.9rem .7rem .3rem}
.side .nav a{display:flex;align-items:center;gap:.55rem;padding:.5rem .65rem;border-radius:9px;
  color:var(--ink);font-size:.9rem;font-weight:500}
.side .nav a:hover{background:var(--well);text-decoration:none}
.side .nav a.on{background:var(--accent-050);color:var(--accent-600);font-weight:600}
.side .plist{padding:.3rem .7rem 1.2rem}
.side .plist .h{font-family:var(--mono);font-size:.64rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--faint);padding:.75rem .65rem .4rem}
.side .plist a{display:flex;align-items:center;gap:.55rem;padding:.42rem .65rem;border-radius:8px;
  color:var(--muted);font-size:.86rem;font-family:var(--mono)}
.side .plist a:hover{background:var(--well);text-decoration:none;color:var(--ink)}
.side .plist a.on{background:var(--accent-050);color:var(--accent-600);font-weight:600}
.side .pdot{width:15px;height:15px;border-radius:50%;flex:none;display:inline-flex;
  align-items:center;justify-content:center;color:#fff;font-size:.6rem;font-weight:800;line-height:1}

.main{flex:1;min-width:0}
.mtop{display:flex;align-items:center;gap:.85rem;padding:.85rem 1.6rem;border-bottom:1px solid var(--line);
  position:sticky;top:0;background:rgba(250,248,245,.92);backdrop-filter:blur(8px);z-index:6}
.mtop .app{font-weight:740;font-size:1rem;color:var(--accent);letter-spacing:-.01em}
.mtop .organ{font-family:var(--mono);font-size:.74rem}
.mtop .spacer{flex:1}
.mtop nav{display:flex;gap:.15rem}
.mtop nav a{font-size:.85rem;color:var(--muted);padding:.35rem .7rem;border-radius:8px;font-weight:500}
.mtop nav a:hover{background:var(--well);text-decoration:none}
.mtop nav a.on{color:var(--accent-600);font-weight:650}
.mtop nav a.off{color:var(--faint);cursor:default}.mtop nav a.off:hover{background:none}
.btn-sm{font-size:.82rem;padding:.42rem .8rem;border-radius:9px;font-weight:600}
.crumb{font-size:.85rem;color:var(--muted);margin:0 0 1rem;font-family:var(--mono)}
.crumb a{color:var(--muted)} .crumb b{color:var(--accent-600)}
.content{padding:1.5rem 2rem 3.5rem;max-width:1600px}
.page-head{display:flex;align-items:flex-end;justify-content:space-between;gap:1rem;flex-wrap:wrap;margin-bottom:1.4rem}
.page-h{margin:.1rem 0 .25rem;font-size:1.9rem;font-weight:740;letter-spacing:-.02em}
.page-sub{color:var(--muted);margin:0;font-size:.92rem}
.page-actions{display:flex;gap:.5rem}

/* stat cards */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.9rem;margin-bottom:1.3rem}
.stat{padding:1.1rem 1.2rem;border-radius:16px}
.stat .top{display:flex;justify-content:space-between;align-items:center}
.stat .k{color:var(--muted);font-size:.88rem}
.stat .tag{font-family:var(--mono);font-size:.6rem;letter-spacing:.06em;font-weight:700}
.stat .big{font-size:2.5rem;font-weight:760;letter-spacing:-.03em;line-height:1.05;margin:.3rem 0 .1rem}
.stat .big .pct{font-size:1.05rem;color:var(--muted);font-weight:600}
.stat .cap{color:var(--faint);font-size:.77rem}
.bar{height:6px;border-radius:100px;background:var(--well);margin-top:.65rem;overflow:hidden}
.bar>i{display:block;height:100%;border-radius:100px}

.cols{display:grid;grid-template-columns:1fr 330px;gap:1rem;align-items:start}
@media (max-width:920px){.cols{grid-template-columns:1fr}}
.cols2{display:grid;grid-template-columns:1fr 1fr;gap:1rem;align-items:start;margin-top:1rem}
@media (max-width:920px){.cols2{grid-template-columns:1fr}}

/* cohort viz: QEI strip + check-matrix carpet */
.stripsvg{width:100%;height:auto;margin-top:.3rem}
.carpet-panel .ph{align-items:flex-start}
.carpet-legend{display:flex;gap:.7rem;font-family:var(--mono);font-size:.6rem;color:var(--muted);flex-wrap:wrap}
.carpet-legend .lg{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:.25rem;vertical-align:middle}
.carpet-scroll{overflow-x:auto}
.carpet{border-collapse:separate;border-spacing:3px;font-size:.7rem}
.carpet th.rk{text-align:right;font-weight:600;color:var(--muted);white-space:nowrap;padding-right:.4rem;font-family:var(--sans)}
.carpet th.ck{font-family:var(--mono);font-size:.62rem;color:var(--faint);font-weight:600;
  writing-mode:vertical-rl;transform:rotate(180deg);height:64px;vertical-align:bottom}
.carpet th.ck a{color:var(--faint)}
.carpet .cell{display:flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:5px;
  color:#fff;font-size:.62rem;font-weight:800;text-decoration:none}
.carpet .cell:hover{outline:2px solid var(--ink);outline-offset:1px}
.panel{padding:1.15rem 1.25rem;border-radius:16px}
.panel h3{font-size:1.02rem;margin-bottom:.15rem}
.panel .ph{display:flex;justify-content:space-between;align-items:center;margin-bottom:.7rem}
.tag-muted{font-family:var(--mono);font-size:.64rem;color:var(--muted);background:var(--well);padding:.2rem .5rem;border-radius:100px}
.mtop .chip.mode{font-size:.7rem;color:var(--accent-600);border-color:#EFDBCC;background:var(--accent-050)}

/* ledger filter bar */
.filterbar{display:flex;align-items:center;justify-content:space-between;gap:.6rem;flex-wrap:wrap;margin:.2rem 0 .8rem}
.fchips{display:flex;gap:.4rem;flex-wrap:wrap}
.fchip{appearance:none;border:1px solid var(--line);background:var(--surface);color:var(--muted);
  font-family:var(--mono);font-size:.72rem;font-weight:600;padding:.3rem .6rem;border-radius:100px;
  cursor:pointer;display:inline-flex;align-items:center;gap:.35rem;transition:.12s}
.fchip:hover{border-color:var(--faint);color:var(--ink)}
.fchip.on{background:var(--accent-050);border-color:var(--accent);color:var(--accent-600)}
.fchip .fn{font-size:.66rem;opacity:.7}
.factive{font-size:.76rem;color:var(--faint);font-family:var(--mono)}
.ledger tr.hidden{display:none}

/* provenance dots in the drawer */
.pv-dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:.4rem;flex:none;vertical-align:middle}
.pv-legend{display:flex;gap:.9rem;flex-wrap:wrap;margin:.2rem 0 .1rem;font-family:var(--mono);
  font-size:.64rem;color:var(--muted)}
.pv-legend span{display:inline-flex;align-items:center}

/* ledger */
.ledger{width:100%;border-collapse:collapse}
.ledger th{text-align:left;font-family:var(--mono);font-size:.64rem;letter-spacing:.06em;text-transform:uppercase;
  color:var(--faint);font-weight:600;padding:.4rem .6rem;border-bottom:1px solid var(--line)}
.ledger td{padding:.62rem .6rem;border-bottom:1px solid var(--line);vertical-align:middle;font-size:.89rem}
.ledger tr:last-child td{border-bottom:none}
.ledger tbody tr{transition:.1s}.ledger tbody tr:hover td{background:var(--paper)}
.ledger .sid{font-family:var(--mono);font-weight:600}
.vpill{display:inline-block;font-family:var(--mono);font-size:.65rem;font-weight:700;padding:.18rem .55rem;border-radius:100px}
.qei-cell{display:flex;align-items:center;gap:.55rem}
.qei-cell .qbar{width:50px;height:5px;border-radius:100px;background:var(--well);overflow:hidden;flex:none}
.qei-cell .qbar>i{display:block;height:100%}
.viewbtn{display:inline-flex;align-items:center;gap:.3rem;font-family:var(--sans);font-size:.78rem;font-weight:600;
  color:#fff;background:var(--accent);padding:.4rem .7rem;border-radius:8px;white-space:nowrap}
.viewbtn:hover{background:var(--accent-600);text-decoration:none}

/* artifact breakdown */
.abd{display:flex;flex-direction:column;gap:.65rem}
.abd .row .lab{display:flex;justify-content:space-between;font-size:.83rem;margin-bottom:.24rem}
.abd .row .lab .n{font-family:var(--mono);color:var(--muted)}
.abd .row .track{height:7px;border-radius:100px;background:var(--well);overflow:hidden}
.abd .row .track>i{display:block;height:100%;border-radius:100px;background:var(--accent)}
.insight{margin-top:1rem;background:var(--accent-050);border:1px solid #EFDBCC;border-radius:12px;
  padding:.85rem 1rem;font-size:.85rem;color:#5c4028}.insight b{color:var(--accent-600)}
.content .hero{margin-top:0}

/* threshold drawer */
.drawer{position:fixed;top:0;right:0;height:100vh;width:390px;max-width:94vw;background:var(--surface);
  border-left:1px solid var(--line);box-shadow:var(--shadow-lg);z-index:20;transform:translateX(100%);
  transition:transform .22s ease;overflow-y:auto;padding:1.3rem 1.3rem 2.5rem}
.drawer.open{transform:translateX(0)}
.drawer h3{font-size:1.05rem;margin-bottom:.2rem}
.drawer .d-sub{color:var(--muted);font-size:.82rem;margin:0 0 1.1rem}
.drawer label{display:block;font-size:.74rem;font-family:var(--mono);letter-spacing:.03em;text-transform:uppercase;
  color:var(--muted);margin:.9rem 0 .3rem}
.drawer input[type=number],.drawer select{width:100%;padding:.5rem .55rem;border:1px solid var(--line);
  border-radius:9px;background:var(--paper);font-family:var(--mono);font-size:.88rem}
.drawer .grp{margin-top:1.15rem}
.drawer .grp-h{display:flex;align-items:center;gap:.5rem;font-family:var(--mono);font-size:.66rem;letter-spacing:.08em;
  text-transform:uppercase;color:var(--accent-600);font-weight:700;margin-bottom:.5rem}
.drawer .grp-h::after{content:"";flex:1;height:1px;background:var(--line)}
.drawer .two{display:grid;grid-template-columns:1fr 1fr;gap:.15rem .6rem}
.drawer .f label{margin:.35rem 0 .25rem}
.drawer input.changed{background:var(--accent-050);border-color:var(--accent);
  transition:background .35s ease,border-color .35s ease}
.drawer .chk{display:flex;align-items:center;gap:.5rem;margin-top:1.15rem;font-size:.88rem;color:var(--ink);font-family:var(--sans);text-transform:none;letter-spacing:0}
.drawer .actions{display:flex;gap:.6rem;margin-top:1.5rem}
.drawer .actions .btn-primary{flex:1}
.d-close{position:absolute;top:1rem;right:1.1rem;border:0;background:none;font-size:1.3rem;color:var(--faint);cursor:pointer;line-height:1}
.scrim{position:fixed;inset:0;background:rgba(28,27,26,.28);z-index:15;display:none}
.scrim.open{display:block}
.applied{font-family:var(--mono);font-size:.68rem;color:var(--accent-600);background:var(--accent-050);
  border:1px solid #EFDBCC;padding:.28rem .6rem;border-radius:100px}

/* lightbox */
#lb{position:fixed;inset:0;background:rgba(20,18,16,.93);z-index:60;display:none;place-items:center;padding:2vw;cursor:zoom-out}
#lb.open{display:grid}
/* enlarge to fill the viewport (SVG plots are vector -> crisp at any size),
   capped by both dimensions so the aspect ratio is preserved. */
#lb>*{width:min(1300px,96vw);height:auto;max-width:96vw;max-height:94vh;
  border-radius:12px;box-shadow:0 24px 70px rgba(0,0,0,.55);background:#0c0b0a;padding:14px}
.content figure img,.content figure svg{cursor:zoom-in}

/* new-analysis modal */
.modal{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%) scale(.98);opacity:0;
  pointer-events:none;width:560px;max-width:94vw;max-height:90vh;overflow-y:auto;background:var(--surface);
  border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow-lg);z-index:25;
  padding:1.5rem 1.5rem 1.7rem;transition:opacity .18s ease,transform .18s ease}
.modal.open{opacity:1;pointer-events:auto;transform:translate(-50%,-50%) scale(1)}
.modal h3{font-size:1.15rem;margin-bottom:.2rem}
.modal .snip-h{font-family:var(--mono);font-size:.66rem;letter-spacing:.08em;text-transform:uppercase;
  color:var(--accent-600);font-weight:700;margin:1.1rem 0 .4rem}
.modal .snip{background:#161311;color:#EDE6DE;border-radius:10px;padding:.85rem 1rem;margin:0;
  font-family:var(--mono);font-size:.82rem;line-height:1.55;overflow-x:auto;white-space:pre}
.modal .actions{display:flex;gap:.6rem;margin-top:1.4rem}

/* organ disclosure menu */
.organ-menu{position:relative}
.organ-menu summary{cursor:pointer;list-style:none}
.organ-menu summary::-webkit-details-marker{display:none}
.organ-menu .caret{color:var(--faint);font-size:.7em}
.omenu{position:absolute;top:calc(100% + .4rem);left:0;width:270px;background:var(--surface);
  border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow-lg);z-index:12;padding:.4rem}
.orow{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:.5rem;
  padding:.45rem .55rem;border-radius:8px}
.orow.sel{background:var(--accent-050)}
.orow .og{font-size:1rem}.orow .ol{font-weight:600;font-size:.86rem}
.orow .on-note{grid-column:2/4;font-size:.72rem;color:var(--muted)}
.ostate{font-family:var(--mono);font-size:.6rem;letter-spacing:.05em;text-transform:uppercase;
  color:var(--faint);border:1px solid var(--line);border-radius:100px;padding:.1rem .4rem}
.ostate.on{color:var(--pass);border-color:#BFE3D8;background:#EAF6F1}
.omenu-foot{font-size:.7rem;color:var(--faint);padding:.5rem .55rem 0;border-top:1px solid var(--line);margin-top:.3rem}
.omenu-foot code{background:var(--well);border-radius:4px;padding:.02rem .3rem}

/* print: strip the app chrome, keep the report */
@media print{
  .side,.mtop,.drawer,.scrim,.modal,#lb,.page-actions,.viewbtn,.filterbar,.crumb{display:none!important}
  body{display:block}.main{min-width:0}.content{max-width:none;padding:0}
  .card,.check,figure{break-inside:avoid}
  *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
}
"""

_GEAR = ('<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
         'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
         '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1'
         '-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 '
         '1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 '
         '.33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82'
         'l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 '
         '4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 '
         '1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>')


# a redundant (non-colour) glyph per verdict, so the triage list is legible
# without relying on colour alone.
_VGLYPH = {"PASS": "✓", "WARN": "!", "FAIL": "✕",
           "UNKNOWN": "?", "N/A": "–", "INFO": "i"}


def _pdot(v: str) -> str:
    c = VERDICT_COLOURS.get(v, ("#8A8079", ""))[0]
    g = _VGLYPH.get(v, "")
    return (f'<span class="pdot" style="background:{c}" role="img" '
            f'aria-label="{esc(v)}">{g}</span>')


def _sidebar(subjects: list[Subject], active: str | None, view: str, q: str) -> str:
    plist = ['<div class="plist"><div class="h">Participants</div>']
    for s in _worst_first(subjects):
        on = "on" if s.sid == active else ""
        plist.append(f'<a class="{on}" href="/subject/{esc(s.sid)}" title="{esc(s.sid)} &mdash; '
                     f'{esc(s.overall)}">{_pdot(s.overall)}{esc(s.sid)}</a>')
    plist.append("</div>")
    return (
        '<aside class="side">'
        f'<div class="side-brand">{brand("QC-ToolBox V1.0")}</div>'
        '<nav class="nav">'
        f'<a class="{"on" if view=="overview" else ""}" href="/">&#9638;&nbsp; Overview</a>'
        f'<a class="{"on" if view=="upload" else ""}" href="/upload">&#43;&nbsp; Grade a new scan</a>'
        '</nav>' + "".join(plist) + '</aside>'
    )


def _mtop(cfg: QCConfig, view: str, overrides: dict) -> str:
    applied = ('<span class="applied">custom thresholds</span>' if overrides else "")
    # the grading regime redefines what FAIL means, so keep it always visible
    strict = ('<span class="chip mode" title="Uncalibrated cutoffs can raise a FAIL">'
              'strict grading</span>' if cfg.strict else
              '<span class="chip mode" title="Uncalibrated cutoffs are demoted to WARN">'
              'lenient grading</span>')
    return (
        '<div class="mtop">'
        '<span class="app">QC-ToolBox V1.0</span>'
        f'{_organ_menu(cfg)}'
        f'{strict}'
        '<div class="spacer"></div>'
        f'{applied}'
        f'<nav><a class="{"on" if view=="overview" else ""}" href="/">Dashboard</a>'
        '<a class="off" title="planned">Projects</a><a class="off" title="planned">Archive</a></nav>'
        '<button type="button" class="btn btn-sm" onclick="openRun()">&#43;&nbsp;New analysis</button>'
        f'<button type="button" class="btn btn-sm" onclick="openCfg()">{_GEAR}&nbsp;Thresholds</button>'
        '</div>'
    )


# organ glyph + the QC that changes per organ (honest 'planned' states from ORGANS)
_ORGAN_META = {
    "brain": ("\U0001F9E0", "Brain", "active", "All 19 checks apply."),
    "kidney": ("\U0001FAC0", "Kidney", "planned", "Skips QEI, GM/WM ratio, deep-GM."),
    "placenta": ("\U0001FAC3", "Placenta", "planned", "Body-ASL profile, not yet built."),
    "preclinical": ("\U0001F401", "Preclinical", "planned", "Rodent ASL, not yet built."),
}


def _organ_menu(cfg: QCConfig) -> str:
    """A disclosure menu of organs: the active one selected, the rest shown as
    honest, inert 'planned' rows naming what each would change. Never a live link
    to an organ the engine cannot grade."""
    from .core.config import ORGANS

    glyph, label, _st, _note = _ORGAN_META.get(cfg.organ, ("\U0001F9E0", cfg.organ.title(), "active", ""))
    rows = []
    for name, (g, lab, st, note) in _ORGAN_META.items():
        active = name == cfg.organ
        is_real = name in ORGANS          # brain + kidney stub exist in config
        badge = ('<span class="ostate on">active</span>' if active else
                 f'<span class="ostate">{esc(st)}</span>')
        rows.append(f'<div class="orow{" sel" if active else ""}">'
                    f'<span class="og">{g}</span><span class="ol">{esc(lab)}</span>{badge}'
                    f'<span class="on-note">{esc(note)}</span></div>')
    return (
        '<details class="organ-menu"><summary class="chip organ">'
        f'{glyph} {esc(label)} <span class="caret">&#9662;</span></summary>'
        f'<div class="omenu">{"".join(rows)}'
        '<div class="omenu-foot">Organ sets are chosen at grading time '
        '(<code>--organ</code>); the dashboard shows the active one.</div>'
        '</div></details>'
    )


def _config_drawer(cfg: QCConfig, back: str) -> str:
    pops = [p for p in _POP_ORDER if p in POPULATIONS]
    pop_opts = "".join(
        f'<option value="{p}"{" selected" if p == cfg.population else ""}>{p.replace("_"," ")}</option>'
        for p in pops)

    def cell(name: str, label: str) -> str:
        val = getattr(cfg, name)
        lvl, cite, _why = provenance_of(name)
        col = PROVENANCE_COLOURS.get(lvl.value, ("", "#9A938C"))[1]
        dot = (f'<span class="pv-dot" style="background:{col}" '
               f'title="{esc(lvl.value)} &mdash; {esc(cite)}"></span>')
        return (f'<div class="f"><label>{dot}{esc(label)}</label>'
                f'<input type="number" step="any" name="{name}" value="{val}"></div>')

    legend = '<div class="pv-legend">' + "".join(
        f'<span><span class="pv-dot" style="background:{c}"></span>{esc(lab)}</span>'
        for lab, c in PROVENANCE_COLOURS.values()) + '</div>'
    groups = legend + "".join(
        f'<div class="grp"><div class="grp-h">{esc(title)}</div>'
        f'<div class="two">{"".join(cell(n, lab) for n, lab in fields)}</div></div>'
        for title, fields in TUNABLE_GROUPS)
    return (
        f'<div class="scrim" id="scrim" onclick="closeCfg()"></div>'
        f'<form class="drawer" id="drawer" method="get" action="/apply">'
        f'<button type="button" class="d-close" onclick="closeCfg()" aria-label="close">&times;</button>'
        '<h3>Thresholds</h3>'
        '<p class="d-sub">Every cutoff that grades the CBF-map checks. Pick a population to '
        'reset the CBF bands, then fine-tune. Re-grades the whole cohort live.</p>'
        f'<input type="hidden" name="back" value="{esc(back)}">'
        f'<label>Population</label>'
        f'<select name="population" onchange="applyPop(this.value)">{pop_opts}</select>'
        f'{groups}'
        f'<label class="chk"><input type="checkbox" name="strict" value="on"'
        f'{" checked" if cfg.strict else ""}> strict &mdash; uncalibrated checks may FAIL</label>'
        '<div class="actions"><button type="submit" class="btn btn-primary">Apply &amp; re-grade</button>'
        '<a class="btn" href="/apply?back=' + esc(back) + '">Reset</a></div>'
        '</form>'
    )


_CLI_SNIPPET = (
    "# grade a whole cohort and open the dashboard\n"
    "osipy-qc --dashboard ./cohort\n\n"
    "# a single scan -> one self-contained HTML report\n"
    "osipy-qc ./scan --html report.html\n\n"
    "# try it on synthetic data (no data needed)\n"
    "osipy-qc --dashboard-demo"
)
_PY_SNIPPET = (
    "from osipy_qc import grade_cbf\n\n"
    'report = grade_cbf("cbf.nii.gz", gm="gm.nii.gz", wm="wm.nii.gz")\n'
    "print(report.overall.value)   # 'PASS' | 'WARN' | 'FAIL'\n"
    "report.to_dict()              # full per-check JSON"
)
_REPO_URL = "https://github.com/Jitmisra/osipy-qc"


def _new_analysis_modal() -> str:
    """The proposal's 'New Analysis' modal: verbatim-runnable CLI + Python, and a
    link to the upload console. Every command here is real (see cli.py / io.py)."""
    return (
        '<div class="scrim" id="runscrim" onclick="closeRun()"></div>'
        '<div class="modal" id="runmodal" role="dialog" aria-label="Run a new analysis">'
        '<button type="button" class="d-close" onclick="closeRun()" aria-label="close">&times;</button>'
        '<h3>Run a new analysis</h3>'
        '<p class="d-sub">Point the toolbox at your data from the command line or Python &mdash; '
        'pure NumPy + nibabel, no framework, no build step.</p>'
        '<div class="snip-h">Command line</div>'
        f'<pre class="snip">{esc(_CLI_SNIPPET)}</pre>'
        '<div class="snip-h">Python</div>'
        f'<pre class="snip">{esc(_PY_SNIPPET)}</pre>'
        '<div class="actions">'
        '<a class="btn btn-primary" href="/upload">Open the upload console</a>'
        f'<a class="btn" href="{_REPO_URL}" target="_blank" rel="noopener">README</a></div>'
        '</div>'
    )


_LIGHTBOX_JS = """
<div id="lb" onclick="this.classList.remove('open');this.innerHTML=''"></div>
<script>
var POP_VALUES=__POP_VALUES__;
function applyPop(p){var v=POP_VALUES[p]; if(!v)return;
  // repopulate the number inputs so what's shown (and submitted) matches the
  // chosen population - otherwise the stale values clobber the reset on Apply.
  for(var k in v){var el=document.querySelector('#drawer [name="'+k+'"]');
    if(el){el.value=v[k]; el.classList.add('changed');
      setTimeout(function(e){return function(){e.classList.remove('changed')}}(el),700);}}}
function openRun(){document.getElementById('runmodal').classList.add('open');
  document.getElementById('runscrim').classList.add('open');}
function closeRun(){document.getElementById('runmodal').classList.remove('open');
  document.getElementById('runscrim').classList.remove('open');}
function filterLedger(btn){
  var f=btn.dataset.f, shown=0;
  document.querySelectorAll('.fchip').forEach(function(b){b.classList.toggle('on',b===btn);});
  document.querySelectorAll('#ledgerbody tr').forEach(function(tr){
    var hit=(f==='all'||tr.dataset.v===f); tr.classList.toggle('hidden',!hit); if(hit)shown++;});
  var c=document.getElementById('fcount'); if(c)c.textContent=shown;}
function openCfg(){document.getElementById('drawer').classList.add('open');
  document.getElementById('scrim').classList.add('open');}
function closeCfg(){document.getElementById('drawer').classList.remove('open');
  document.getElementById('scrim').classList.remove('open');}
document.addEventListener('click',function(e){
  var el=e.target.closest('.content figure img, .content figure svg');
  if(!el)return;
  var lb=document.getElementById('lb');
  lb.innerHTML=''; lb.appendChild(el.cloneNode(true)); lb.classList.add('open');
});
document.addEventListener('keydown',function(e){if(e.key==='Escape'){
  document.getElementById('lb').classList.remove('open');closeCfg();closeRun();}});
</script>
""".replace("__POP_VALUES__", _POP_VALUES_JSON)


def _shell(subjects, active, view, crumb, content, cfg, overrides, back) -> str:
    crumb_html = f'<p class="crumb">{crumb}</p>' if crumb else ""
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>OSIPI QC-ToolBox</title><style>{BASE_CSS}{REPORT_CSS}{_DASH_CSS}</style></head><body>"
        + _sidebar(subjects, active, view, back)
        + '<main class="main">' + _mtop(cfg, view, overrides)
        + f'<div class="content">{crumb_html}{content}</div></main>'
        + _config_drawer(cfg, back) + _new_analysis_modal() + _LIGHTBOX_JS
        + "</body></html>"
    )


# --------------------------------------------------------------------------- #
# Batch Overview
# --------------------------------------------------------------------------- #
def _stat_card(k, big, cap, tag, tag_col, frac, bar_col) -> str:
    bar = (f'<div class="bar"><i style="width:{max(0.0,min(1.0,frac))*100:.0f}%;'
           f'background:{bar_col}"></i></div>') if frac is not None else ""
    return (f'<div class="card stat"><div class="top"><span class="k">{esc(k)}</span>'
            f'<span class="tag" style="color:{tag_col}">{esc(tag)}</span></div>'
            f'<div class="big">{big}</div><span class="cap">{esc(cap)}</span>{bar}</div>')


def _ledger(subjects: list[Subject]) -> str:
    rows = []
    for s in _worst_first(subjects):
        fg, bg = VERDICT_COLOURS.get(s.overall, ("#8A8079", "#F1ECE4"))
        q = s.qei
        if isinstance(q, (int, float)):
            qbar = (f'<span class="qbar"><i style="width:{max(0,min(1,q))*100:.0f}%;background:{fg}"></i></span>')
            qtxt = f'<span class="num">{q:.2f}</span>'
        else:
            qbar, qtxt = "", '<span class="num" style="color:var(--faint)">&mdash;</span>'
        flag = s.primary_artifact
        # 'incomplete (...)' is a not-graded state, not a flag — mute it.
        flag_cell = (f'<span style="color:var(--faint)">{esc(flag)}</span>'
                     if flag.startswith("incomplete") else esc(flag))
        rows.append(
            f'<tr data-v="{esc(s.overall)}"><td class="sid">{esc(s.sid)}</td>'
            f'<td><span class="vpill" style="color:{fg};background:{bg}">'
            f'{_VGLYPH.get(s.overall,"")} {esc(s.overall)}</span></td>'
            f'<td><div class="qei-cell">{qbar}{qtxt}</div></td>'
            f'<td>{flag_cell}</td>'
            f'<td><a class="viewbtn" href="/subject/{esc(s.sid)}">View report &rarr;</a></td></tr>')
    return ('<table class="ledger"><thead><tr><th>Participant</th><th>Verdict</th>'
            '<th>QEI</th><th>Primary flag</th><th>Action</th></tr></thead>'
            f'<tbody id="ledgerbody">{"".join(rows)}</tbody></table>')


def _artifact_breakdown(summary: BatchSummary) -> str:
    top = summary.artifact_breakdown[:6]
    if not top:
        return '<p class="cap" style="color:var(--muted)">No checks flagged &mdash; a clean cohort.</p>'
    denom = summary.total or 1        # fraction OF THE COHORT, not of the worst bar
    return '<div class="abd">' + "".join(
        f'<div class="row"><div class="lab"><span>{esc(lab)}</span><span class="n">{n}/{summary.total}</span></div>'
        f'<div class="track"><i style="width:{n/denom*100:.0f}%"></i></div></div>' for lab, n in top) + "</div>"


def _insight(summary: BatchSummary, cfg: QCConfig) -> str:
    worst = summary.artifact_breakdown[0][0] if summary.artifact_breakdown else None
    driver = f" The most common flag is <b>{esc(worst)}</b>." if worst else ""
    return (f'<div class="insight">Analysed <b>{summary.total}</b> subjects against the '
            f'<b>{esc(cfg.population)}</b> profile: <b>{summary.pass_rate*100:.0f}%</b> pass, '
            f'{summary.warn_rate*100:.0f}% warn, {summary.fail_rate*100:.0f}% fail.{driver}</div>')


def _median(xs: list[float]):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    m = n // 2
    return xs[m] if n % 2 else (xs[m - 1] + xs[m]) / 2


def _check_carpet(subjects: list[Subject]) -> str:
    """A checks x subjects verdict grid (the MRIQC group-report pattern). A whole
    row lit red is a systemic acquisition problem the summary bars would hide.
    Hand-built table, verdict colour + a redundant glyph, every cell a jump-link."""
    from .batch import check_label

    cols = _worst_first(subjects)
    # rows = checks that ran, worst (most-flagged) first
    flags: dict[str, int] = {}
    lut: dict[tuple, str] = {}
    for s in subjects:
        for r in s.report.results:
            flags.setdefault(r.check, 0)
            if r.verdict.value in ("FAIL", "WARN"):
                flags[r.check] += 1
            lut[(s.sid, r.check)] = r.verdict.value
    checks = sorted(flags, key=lambda c: (-flags[c], c))
    if not checks or not cols:
        return ""

    head = '<th class="rk"></th>' + "".join(
        f'<th class="ck"><a href="/subject/{esc(s.sid)}">{esc(s.sid)}</a></th>' for s in cols)
    body = []
    for check in checks:
        cells = []
        for s in cols:
            v = lut.get((s.sid, check))
            fg = VERDICT_COLOURS.get(v, ("#C9C4BD", ""))[0] if v else "#E5E0DA"
            g = _VGLYPH.get(v, "")
            cells.append(f'<td><a class="cell" href="/subject/{esc(s.sid)}" '
                         f'style="background:{fg}" title="{esc(s.sid)}: {esc(v or "n/a")}">{g}</a></td>')
        body.append(f'<tr><th class="rk">{esc(check_label(check))}</th>{"".join(cells)}</tr>')
    legend = " ".join(
        f'<span><span class="lg" style="background:{VERDICT_COLOURS[v][0]}"></span>{v}</span>'
        for v in ("PASS", "WARN", "FAIL", "N/A"))
    return ('<div class="card panel carpet-panel"><div class="ph"><h3>Check matrix</h3>'
            f'<span class="carpet-legend">{legend}</span></div>'
            '<p class="cap" style="color:var(--muted);margin:.1rem 0 .7rem">Every check &times; every '
            'subject. A whole row lit up points to a systemic problem.</p>'
            '<div class="carpet-scroll"><table class="carpet"><thead><tr>'
            f'{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div></div>')


def _qei_strip(subjects: list[Subject], cfg: QCConfig) -> str:
    """A 1-D strip of every subject's QEI against the published 0.5 cutoff — the
    most defensible cohort chart for the QEI author. A distribution, NOT a time
    series (subjects carry no dates). Missing-QEI subjects get their own lane."""
    W, H, padx = 560, 96, 30
    x0, x1 = padx, W - padx
    y = 46

    def px(q):
        return x0 + (x1 - x0) * max(0.0, min(1.0, q))

    have = [s for s in subjects if isinstance(s.qei, (int, float))]
    na = [s for s in subjects if not isinstance(s.qei, (int, float))]
    cut = px(cfg.qei_warn)
    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" class="stripsvg" role="img">']
    # axis
    parts.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="var(--line)" stroke-width="2"/>')
    for t in (0.0, 0.5, 1.0):
        parts.append(f'<text x="{px(t):.0f}" y="{y+22}" text-anchor="middle" font-size="9" '
                     f'fill="#9A938C" font-family="monospace">{t:g}</text>')
    # cutoff line
    parts.append(f'<line x1="{cut:.1f}" y1="18" x2="{cut:.1f}" y2="{y+6}" stroke="#A43122" '
                 f'stroke-width="1.4" stroke-dasharray="3 2"/>'
                 f'<text x="{cut:.1f}" y="14" text-anchor="middle" font-size="8.5" fill="#A43122" '
                 f'font-family="monospace">cutoff {cfg.qei_warn:g}</text>')
    # dots, jittered vertically by index parity to reduce overlap
    for i, s in enumerate(have):
        c = VERDICT_COLOURS.get(s.overall, ("#8A8079", ""))[0]
        dy = (-1 if i % 2 else 1) * (3 + 2 * (i % 3))
        parts.append(f'<circle cx="{px(s.qei):.1f}" cy="{y+dy}" r="5.5" fill="{c}" '
                     f'fill-opacity="0.85" stroke="#fff" stroke-width="1"><title>{esc(s.sid)}: '
                     f'QEI {s.qei:.3f} ({esc(s.overall)})</title></circle>')
    parts.append("</svg>")
    na_note = (f'<p class="cap" style="color:var(--faint);margin:.4rem 0 0">'
               f'{len(na)} subject(s) had no QEI (no tissue maps).</p>' if na else "")
    return ('<div class="card panel"><h3>QEI across the cohort</h3>'
            '<p class="cap" style="color:var(--muted);margin:.1rem 0 .3rem">Each dot is one subject '
            "against the published 0.5 cutoff. Left of the dashed line is a likely reject.</p>"
            + "".join(parts) + na_note + '</div>')


def _filter_bar(summary: BatchSummary) -> str:
    """Client-side verdict filter for the ledger — the proposal's Filter control.
    Rows carry data-v; the chips show/hide them with no server round-trip."""
    chips = [('all', 'All', summary.total)]
    for v in ("PASS", "WARN", "FAIL"):
        chips.append((v, v.title(), summary.counts.get(v, 0)))
    btns = "".join(
        f'<button type="button" class="fchip{" on" if key == "all" else ""}" '
        f'data-f="{key}" onclick="filterLedger(this)">{esc(lab)}'
        f'<span class="fn">{n}</span></button>' for key, lab, n in chips)
    return ('<div class="filterbar"><div class="fchips">' + btns + '</div>'
            '<span class="factive">Showing <b id="fcount">' + str(summary.total)
            + '</b> of ' + str(summary.total) + '</span></div>')


def render_overview(subjects, summary, cfg, dataset="cohort", overrides=None) -> str:
    overrides = overrides or {}
    p, w, f = VERDICT_COLOURS["PASS"][0], VERDICT_COLOURS["WARN"][0], VERDICT_COLOURS["FAIL"][0]

    def tag(rate, good_high):
        if good_high:
            return ("OPTIMAL", p) if rate >= 0.8 else ("MODERATE", w) if rate >= 0.5 else ("LOW", f)
        return ("CRITICAL", f) if rate >= 0.4 else ("ELEVATED", w) if rate >= 0.15 else ("LOW", p)

    # Median QEI — the one PUBLISHED, calibrated cohort number (robust on the
    # small cohorts this tool runs). '—' honestly when no subject had tissue maps.
    qeis = [s.qei for s in subjects if isinstance(s.qei, (int, float))]
    med = _median(qeis)
    if med is None:
        qei_card = _stat_card("Median QEI", "&mdash;", "no tissue maps supplied",
                              "PUBLISHED", PROVENANCE_COLOURS["published"][1], None, "")
    else:
        below = sum(1 for q in qeis if q < cfg.qei_warn)
        qcol = p if med >= cfg.qei_warn else f
        qei_card = _stat_card("Median QEI", f"{med:.2f}",
                              f"{below}/{len(qeis)} below the {cfg.qei_warn:g} cutoff",
                              "PUBLISHED", PROVENANCE_COLOURS["published"][1], frac=med, bar_col=qcol)

    stats = (
        _stat_card("Total scans", f"{summary.total}", "Analysed cohort", "AGGREGATE", "var(--muted)", None, "")
        + qei_card
        + _stat_card("Pass rate", f"{summary.pass_rate*100:.0f}<span class='pct'>%</span>",
                     "usable without caveats", *tag(summary.pass_rate, True), frac=summary.pass_rate, bar_col=p)
        + _stat_card("Warning", f"{summary.warn_rate*100:.0f}<span class='pct'>%</span>",
                     "review recommended", *tag(summary.warn_rate, False), frac=summary.warn_rate, bar_col=w)
        + _stat_card("Fail rate", f"{summary.fail_rate*100:.0f}<span class='pct'>%</span>",
                     "disqualifying problem", *tag(summary.fail_rate, False), frac=summary.fail_rate, bar_col=f))
    content = (
        '<div class="page-head"><div>'
        '<h1 class="page-h">Batch overview</h1>'
        f'<p class="page-sub">Dataset: {esc(dataset)} &middot; {summary.total} participants '
        f'&middot; {esc(cfg.population)} profile</p></div>'
        '<div class="page-actions">'
        f'<button class="btn btn-sm" onclick="openCfg()">{_GEAR}&nbsp;Thresholds</button>'
        '<button class="btn btn-sm btn-primary" onclick="window.print()">Print / Save PDF</button>'
        '</div></div>'
        f'<div class="stats">{stats}</div>'
        '<div class="cols">'
        '<div class="card panel"><div class="ph"><h3>Participant ledger</h3>'
        '<span class="tag-muted">worst first</span></div>'
        + _filter_bar(summary) + _ledger(subjects) + '</div>'
        '<div class="card panel"><h3>Flagged checks</h3>'
        '<p class="cap" style="color:var(--muted);margin:.1rem 0 .85rem">How often each check flagged a subject.</p>'
        + _artifact_breakdown(summary) + _insight(summary, cfg) + '</div></div>'
        + '<div class="cols2">' + _qei_strip(subjects, cfg) + _check_carpet(subjects) + '</div>')
    return _shell(subjects, None, "overview", "", content, cfg, overrides, "/")


def render_subject(subjects, subject, cfg, overrides=None) -> str:
    overrides = overrides or {}
    scfg = subject.cfg
    # prev/next through the cohort in the SAME (worst-first) order as the ledger
    order = _worst_first(subjects)
    ids = [s.sid for s in order]
    i = ids.index(subject.sid) if subject.sid in ids else 0
    prev_sid = ids[i - 1] if i > 0 else None
    next_sid = ids[i + 1] if i < len(ids) - 1 else None

    def step(sid, label, arrow_first):
        if not sid:
            return f'<span class="btn btn-sm" aria-disabled="true" style="opacity:.4">{label}</span>'
        txt = f'&larr; {label}' if arrow_first else f'{label} &rarr;'
        return f'<a class="btn btn-sm" href="/subject/{esc(sid)}">{txt}</a>'

    fg = VERDICT_COLOURS.get(subject.overall, ("#8A8079", ""))[0]
    head = (
        '<div class="page-head"><div>'
        '<h1 class="page-h">Participant quality report</h1>'
        f'<p class="page-sub"><b class="num">{esc(subject.sid)}</b> &middot; '
        f'{esc(scfg.population)} profile &middot; '
        f'<b style="color:{fg}">{esc(subject.overall)}</b></p></div>'
        '<div class="page-actions">'
        + step(prev_sid, "Prev", True) + step(next_sid, "Next", False)
        + '<button class="btn btn-sm" onclick="openCfg()">' + _GEAR + '&nbsp;Thresholds</button>'
        '<button class="btn btn-sm btn-primary" onclick="window.print()">Print / Save PDF</button>'
        '</div></div>'
    )
    body = head + report_body(subject.report, subject.inputs, scfg, with_note=True)
    crumb = f'<a href="/">Overview</a> / <b>{esc(subject.sid)}</b>'
    back = f"/subject/{subject.sid}"
    return _shell(subjects, subject.sid, "subject", crumb, body, scfg, overrides, back)
