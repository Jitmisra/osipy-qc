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

from ._webassets import BASE_CSS, VERDICT_COLOURS, brand, esc
from .batch import TUNABLE, BatchSummary, Subject
from .core.config import POPULATIONS, QCConfig
from .report_html import REPORT_CSS, report_body

_POP_ORDER = ["neonate_preterm", "neonate_term", "infant", "child",
              "adolescent", "adult", "elderly"]

_DASH_CSS = """
body{display:flex;min-height:100vh}
.side{width:230px;flex:none;background:var(--surface);border-right:1px solid var(--line);
  display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto}
.side .brand{padding:1.15rem 1.2rem;border-bottom:1px solid var(--line)}
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
.side .pdot{width:8px;height:8px;border-radius:50%;flex:none}

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
.content{padding:1.5rem 1.8rem 3.5rem;max-width:1220px}
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
.panel{padding:1.15rem 1.25rem;border-radius:16px}
.panel h3{font-size:1.02rem;margin-bottom:.15rem}
.panel .ph{display:flex;justify-content:space-between;align-items:center;margin-bottom:.7rem}
.tag-muted{font-family:var(--mono);font-size:.64rem;color:var(--muted);background:var(--well);padding:.2rem .5rem;border-radius:100px}

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
.drawer{position:fixed;top:0;right:0;height:100vh;width:340px;max-width:92vw;background:var(--surface);
  border-left:1px solid var(--line);box-shadow:var(--shadow-lg);z-index:20;transform:translateX(100%);
  transition:transform .22s ease;overflow-y:auto;padding:1.3rem 1.3rem 2.5rem}
.drawer.open{transform:translateX(0)}
.drawer h3{font-size:1.05rem;margin-bottom:.2rem}
.drawer .d-sub{color:var(--muted);font-size:.82rem;margin:0 0 1.1rem}
.drawer label{display:block;font-size:.76rem;font-family:var(--mono);letter-spacing:.03em;text-transform:uppercase;
  color:var(--muted);margin:.9rem 0 .3rem}
.drawer input[type=number],.drawer select{width:100%;padding:.55rem .6rem;border:1px solid var(--line);
  border-radius:9px;background:var(--paper);font-family:var(--mono);font-size:.9rem}
.drawer .two{display:grid;grid-template-columns:1fr 1fr;gap:.6rem}
.drawer .chk{display:flex;align-items:center;gap:.5rem;margin-top:1rem;font-size:.88rem;color:var(--ink);font-family:var(--sans);text-transform:none;letter-spacing:0}
.drawer .actions{display:flex;gap:.6rem;margin-top:1.5rem}
.drawer .actions .btn-primary{flex:1}
.d-close{position:absolute;top:1rem;right:1.1rem;border:0;background:none;font-size:1.3rem;color:var(--faint);cursor:pointer;line-height:1}
.scrim{position:fixed;inset:0;background:rgba(28,27,26,.28);z-index:15;display:none}
.scrim.open{display:block}
.applied{font-family:var(--mono);font-size:.68rem;color:var(--accent-600);background:var(--accent-050);
  border:1px solid #EFDBCC;padding:.28rem .6rem;border-radius:100px}

/* lightbox */
#lb{position:fixed;inset:0;background:rgba(20,18,16,.92);z-index:60;display:none;place-items:center;padding:2.5rem;cursor:zoom-out}
#lb.open{display:grid}
#lb>*{max-width:95vw;max-height:92vh;border-radius:10px;box-shadow:0 20px 60px rgba(0,0,0,.5)}
#lb svg,#lb img{width:auto;height:auto;background:#0c0b0a}
.content figure img,.content figure svg{cursor:zoom-in}
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


def _pdot(v: str) -> str:
    c = VERDICT_COLOURS.get(v, ("#8A8079", ""))[0]
    return f'<span class="pdot" style="background:{c}"></span>'


def _sidebar(subjects: list[Subject], active: str | None, view: str, q: str) -> str:
    plist = ['<div class="plist"><div class="h">Participants</div>']
    for s in subjects:
        on = "on" if s.sid == active else ""
        plist.append(f'<a class="{on}" href="/subject/{esc(s.sid)}">{_pdot(s.overall)}{esc(s.sid)}</a>')
    plist.append("</div>")
    return (
        '<aside class="side">'
        f'<div class="brand">{brand("QC-ToolBox v1.0")}</div>'
        '<nav class="nav">'
        f'<a class="{"on" if view=="overview" else ""}" href="/">&#9638;&nbsp; Overview</a>'
        f'<a class="{"on" if view=="upload" else ""}" href="/upload">&#43;&nbsp; Grade a new scan</a>'
        '</nav>' + "".join(plist) + '</aside>'
    )


def _mtop(cfg: QCConfig, view: str, overrides: dict) -> str:
    applied = ('<span class="applied">custom thresholds</span>' if overrides else "")
    return (
        '<div class="mtop">'
        '<span class="app">QC-ToolBox V1.0</span>'
        f'<span class="chip organ">&#129504; {esc(cfg.organ).title()}</span>'
        '<div class="spacer"></div>'
        f'{applied}'
        f'<nav><a class="{"on" if view=="overview" else ""}" href="/">Dashboard</a>'
        '<a class="off" title="planned">Projects</a><a class="off" title="planned">Archive</a></nav>'
        f'<button type="button" class="btn btn-sm" onclick="openCfg()">{_GEAR}&nbsp;Thresholds</button>'
        '</div>'
    )


def _config_drawer(cfg: QCConfig, back: str) -> str:
    pops = [p for p in _POP_ORDER if p in POPULATIONS]
    pop_opts = "".join(
        f'<option value="{p}"{" selected" if p == cfg.population else ""}>{p.replace("_"," ")}</option>'
        for p in pops)
    fields = []
    for name, label in TUNABLE.items():
        val = getattr(cfg, name)
        fields.append(f'<label>{esc(label)}</label>'
                      f'<input type="number" step="any" name="{name}" value="{val}">')
    # lay the numeric fields two-up
    grid = "".join(f'<div>{fields[i]}{fields[i+1]}</div>'
                   if i + 1 < len(fields) else f'<div>{fields[i]}</div>'
                   for i in range(0, len(fields), 2))
    return (
        f'<div class="scrim" id="scrim" onclick="closeCfg()"></div>'
        f'<form class="drawer" id="drawer" method="get" action="/apply">'
        f'<button type="button" class="d-close" onclick="closeCfg()" aria-label="close">&times;</button>'
        '<h3>Thresholds</h3>'
        '<p class="d-sub">Adjust the grading thresholds and re-grade the whole cohort. '
        'Uncalibrated defaults &mdash; tune to your population.</p>'
        f'<input type="hidden" name="back" value="{esc(back)}">'
        f'<label>Population</label><select name="population">{pop_opts}</select>'
        f'<div class="two" style="margin-top:.2rem">{grid}</div>'
        f'<label class="chk"><input type="checkbox" name="strict" value="on"'
        f'{" checked" if cfg.strict else ""}> strict &mdash; uncalibrated checks may FAIL</label>'
        '<div class="actions"><button type="submit" class="btn btn-primary">Apply &amp; re-grade</button>'
        '<a class="btn" href="/apply?back=' + esc(back) + '">Reset</a></div>'
        '</form>'
    )


_LIGHTBOX_JS = """
<div id="lb" onclick="this.classList.remove('open');this.innerHTML=''"></div>
<script>
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
  document.getElementById('lb').classList.remove('open');closeCfg();}});
</script>
"""


def _shell(subjects, active, view, crumb, content, cfg, overrides, back) -> str:
    crumb_html = f'<p class="crumb">{crumb}</p>' if crumb else ""
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>OSIPI QC-ToolBox</title><style>{BASE_CSS}{REPORT_CSS}{_DASH_CSS}</style></head><body>"
        + _sidebar(subjects, active, view, back)
        + '<main class="main">' + _mtop(cfg, view, overrides)
        + f'<div class="content">{crumb_html}{content}</div></main>'
        + _config_drawer(cfg, back) + _LIGHTBOX_JS
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
    for s in subjects:
        fg, bg = VERDICT_COLOURS.get(s.overall, ("#8A8079", "#F1ECE4"))
        q = s.qei
        if isinstance(q, (int, float)):
            qbar = (f'<span class="qbar"><i style="width:{max(0,min(1,q))*100:.0f}%;background:{fg}"></i></span>')
            qtxt = f'<span class="num">{q:.2f}</span>'
        else:
            qbar, qtxt = "", '<span class="num" style="color:var(--faint)">&mdash;</span>'
        rows.append(
            f'<tr><td class="sid">{esc(s.sid)}</td>'
            f'<td><span class="vpill" style="color:{fg};background:{bg}">{esc(s.overall)}</span></td>'
            f'<td><div class="qei-cell">{qbar}{qtxt}</div></td>'
            f'<td>{esc(s.primary_artifact)}</td>'
            f'<td><a class="viewbtn" href="/subject/{esc(s.sid)}">View report &rarr;</a></td></tr>')
    return ('<table class="ledger"><thead><tr><th>Participant</th><th>Verdict</th>'
            '<th>QEI</th><th>Primary artifact</th><th>Action</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def _artifact_breakdown(summary: BatchSummary) -> str:
    top = summary.artifact_breakdown[:6]
    if not top:
        return '<p class="cap" style="color:var(--muted)">No checks flagged &mdash; a clean cohort.</p>'
    mx = max(n for _, n in top) or 1
    return '<div class="abd">' + "".join(
        f'<div class="row"><div class="lab"><span>{esc(lab)}</span><span class="n">{n}/{summary.total}</span></div>'
        f'<div class="track"><i style="width:{n/mx*100:.0f}%"></i></div></div>' for lab, n in top) + "</div>"


def _insight(summary: BatchSummary, cfg: QCConfig) -> str:
    worst = summary.artifact_breakdown[0][0] if summary.artifact_breakdown else None
    driver = f" The most common flag is <b>{esc(worst)}</b>." if worst else ""
    return (f'<div class="insight">Analysed <b>{summary.total}</b> subjects against the '
            f'<b>{esc(cfg.population)}</b> profile: <b>{summary.pass_rate*100:.0f}%</b> pass, '
            f'{summary.warn_rate*100:.0f}% warn, {summary.fail_rate*100:.0f}% fail.{driver}</div>')


def render_overview(subjects, summary, cfg, dataset="cohort", overrides=None) -> str:
    overrides = overrides or {}
    p, w, f = VERDICT_COLOURS["PASS"][0], VERDICT_COLOURS["WARN"][0], VERDICT_COLOURS["FAIL"][0]

    def tag(rate, good_high):
        if good_high:
            return ("OPTIMAL", p) if rate >= 0.8 else ("MODERATE", w) if rate >= 0.5 else ("LOW", f)
        return ("CRITICAL", f) if rate >= 0.4 else ("ELEVATED", w) if rate >= 0.15 else ("LOW", p)

    stats = (
        _stat_card("Total scans", f"{summary.total}", "Analysed cohort", "AGGREGATE", "var(--muted)", None, "")
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
        '<button class="btn btn-sm btn-primary" onclick="window.print()">Export report</button>'
        '</div></div>'
        f'<div class="stats">{stats}</div>'
        '<div class="cols">'
        '<div class="card panel"><div class="ph"><h3>Participant ledger</h3>'
        '<span class="tag-muted">sorted by import order</span></div>' + _ledger(subjects) + '</div>'
        '<div class="card panel"><h3>Artifact breakdown</h3>'
        '<p class="cap" style="color:var(--muted);margin:.1rem 0 .85rem">How often each check flagged a subject.</p>'
        + _artifact_breakdown(summary) + _insight(summary, cfg) + '</div></div>')
    return _shell(subjects, None, "overview", "", content, cfg, overrides, "/")


def render_subject(subjects, subject, cfg, overrides=None) -> str:
    overrides = overrides or {}
    body = report_body(subject.report, subject.inputs, subject.cfg, with_note=False)
    crumb = f'<a href="/">Overview</a> / <b>{esc(subject.sid)}</b>'
    back = f"/subject/{subject.sid}"
    return _shell(subjects, subject.sid, "subject", crumb, body, subject.cfg, overrides, back)
