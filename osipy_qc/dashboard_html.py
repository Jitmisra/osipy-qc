"""
The cohort dashboard — the proposal's UI, built local and dependency-free.

Three views, one design system:
  * a persistent sidebar with the participant list (verdict-coloured dots)
  * a Batch Overview: cohort stat cards, a participant ledger, an artifact
    breakdown, and a one-line insight
  * a Subject Deep Dive: the full per-scan report (reused wholesale from
    report_html.report_body) inside the dashboard shell

Server-rendered pages, so navigating between subjects is a plain link — no SPA,
no build step. Styling comes from the shared design system (_webassets).
"""

from __future__ import annotations

from ._webassets import BASE_CSS, LOGO_SVG, VERDICT_COLOURS, esc
from .batch import BatchSummary, Subject
from .core.config import QCConfig
from .report_html import REPORT_CSS, report_body

_DASH_CSS = """
body{display:flex;min-height:100vh}
.side{width:248px;flex:none;background:var(--surface);border-right:1px solid var(--line);
  display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto}
.side .brand{padding:1.2rem 1.3rem;border-bottom:1px solid var(--line)}
.side .nav{padding:1rem .8rem .4rem}
.side .nav a{display:flex;align-items:center;gap:.6rem;padding:.55rem .7rem;border-radius:9px;
  color:var(--ink);font-size:.92rem;font-weight:500}
.side .nav a:hover{background:var(--well);text-decoration:none}
.side .nav a.on{background:var(--accent-050);color:var(--accent-600);font-weight:600}
.side .plist{padding:.4rem .8rem 1.2rem}
.side .plist .h{font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--faint);padding:.6rem .7rem .4rem}
.side .plist a{display:flex;align-items:center;gap:.6rem;padding:.42rem .7rem;border-radius:8px;
  color:var(--muted);font-size:.88rem;font-family:var(--mono)}
.side .plist a:hover{background:var(--well);text-decoration:none;color:var(--ink)}
.side .plist a.on{background:var(--well);color:var(--ink);font-weight:600}
.side .pdot{width:8px;height:8px;border-radius:50%;flex:none}

.main{flex:1;min-width:0}
.mtop{display:flex;align-items:center;gap:1rem;padding:1.1rem 2rem;border-bottom:1px solid var(--line);
  position:sticky;top:0;background:rgba(251,247,241,.9);backdrop-filter:blur(6px);z-index:5}
.mtop .title{font-weight:680;font-size:1.02rem;letter-spacing:-.01em}
.mtop .title .v{color:var(--accent-600)}
.mtop .spacer{flex:1}
.content{padding:1.8rem 2rem 3.5rem;max-width:1160px}
.page-h{margin:.2rem 0 .3rem;font-size:2rem;font-weight:730;letter-spacing:-.02em}
.page-sub{color:var(--muted);margin:0 0 1.6rem;font-size:.95rem}

/* stat cards */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem;margin-bottom:1.6rem}
.stat{padding:1.2rem 1.3rem}
.stat .top{display:flex;justify-content:space-between;align-items:center}
.stat .k{color:var(--muted);font-size:.9rem}
.stat .tag{font-family:var(--mono);font-size:.62rem;letter-spacing:.06em;font-weight:700}
.stat .big{font-size:2.6rem;font-weight:760;letter-spacing:-.03em;line-height:1.05;margin:.35rem 0 .1rem}
.stat .big .pct{font-size:1.1rem;color:var(--muted);font-weight:600}
.stat .cap{color:var(--faint);font-size:.78rem}
.bar{height:6px;border-radius:100px;background:var(--well);margin-top:.7rem;overflow:hidden}
.bar > i{display:block;height:100%;border-radius:100px}

/* two-column: ledger + side panel */
.cols{display:grid;grid-template-columns:1fr 320px;gap:1rem;align-items:start}
@media (max-width:900px){.cols{grid-template-columns:1fr}}
.panel{padding:1.2rem 1.3rem}
.panel h3{font-size:1.05rem;margin-bottom:.2rem}
.panel .ph{display:flex;justify-content:space-between;align-items:center;margin-bottom:.8rem}
.tag-muted{font-family:var(--mono);font-size:.66rem;color:var(--muted);background:var(--well);
  padding:.2rem .5rem;border-radius:100px}

/* ledger table */
.ledger{width:100%;border-collapse:collapse}
.ledger th{text-align:left;font-family:var(--mono);font-size:.66rem;letter-spacing:.06em;
  text-transform:uppercase;color:var(--faint);font-weight:600;padding:.4rem .6rem;
  border-bottom:1px solid var(--line)}
.ledger td{padding:.7rem .6rem;border-bottom:1px solid var(--line);vertical-align:middle;font-size:.9rem}
.ledger tr:last-child td{border-bottom:none}
.ledger tr:hover td{background:var(--paper)}
.ledger .sid{font-family:var(--mono);font-weight:600}
.vpill{display:inline-block;font-family:var(--mono);font-size:.66rem;font-weight:700;
  padding:.18rem .55rem;border-radius:100px}
.qei-cell{display:flex;align-items:center;gap:.55rem}
.qei-cell .qbar{width:52px;height:5px;border-radius:100px;background:var(--well);overflow:hidden;flex:none}
.qei-cell .qbar > i{display:block;height:100%}
.view{font-family:var(--mono);font-size:.76rem;color:var(--accent-600);white-space:nowrap}
.view:hover{text-decoration:underline}

/* artifact breakdown */
.abd{display:flex;flex-direction:column;gap:.7rem}
.abd .row .lab{display:flex;justify-content:space-between;font-size:.84rem;margin-bottom:.25rem}
.abd .row .lab .n{font-family:var(--mono);color:var(--muted)}
.abd .row .track{height:7px;border-radius:100px;background:var(--well);overflow:hidden}
.abd .row .track > i{display:block;height:100%;border-radius:100px;background:var(--accent)}
.insight{margin-top:1rem;background:var(--accent-050);border:1px solid #F3D9C4;border-radius:var(--radius-sm);
  padding:.9rem 1rem;font-size:.86rem;color:#5c4632}
.insight b{color:var(--accent-600)}

/* deep dive lives in .content; report_body brings its own .wrap-free styles */
.content .hero{margin-top:0}
"""


def _pdot(v: str) -> str:
    c = VERDICT_COLOURS.get(v, ("#8A8079", ""))[0]
    return f'<span class="pdot" style="background:{c}"></span>'


def _sidebar(subjects: list[Subject], active: str | None, view: str) -> str:
    plist = ['<div class="plist"><div class="h">Participants</div>']
    for s in subjects:
        on = " on" if s.sid == active else ""
        plist.append(f'<a class="{on.strip()}" href="/subject/{esc(s.sid)}">{_pdot(s.overall)}'
                     f'{esc(s.sid)}</a>')
    plist.append("</div>")
    ov_on = " on" if view == "overview" else ""
    up_on = " on" if view == "upload" else ""
    return (
        '<aside class="side">'
        f'<div class="brand">{LOGO_SVG}<b>osipy-qc</b></div>'
        '<nav class="nav">'
        f'<a class="{ov_on.strip()}" href="/">&#9638; Overview</a>'
        f'<a class="{up_on.strip()}" href="/upload">&#43; Grade a new scan</a>'
        '</nav>'
        + "".join(plist) +
        '</aside>'
    )


def _shell(subjects: list[Subject], active: str | None, view: str, title_html: str,
           content: str, cfg: QCConfig) -> str:
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>osipy-qc &mdash; dashboard</title><style>{BASE_CSS}{REPORT_CSS}{_DASH_CSS}</style>"
        "</head><body>"
        + _sidebar(subjects, active, view) +
        '<main class="main">'
        f'<div class="mtop"><div class="title">{title_html}</div><div class="spacer"></div>'
        f'<div class="chip mono">population: {esc(cfg.population)}</div></div>'
        f'<div class="content">{content}</div>'
        '</main></body></html>'
    )


# --------------------------------------------------------------------------- #
# Batch Overview
# --------------------------------------------------------------------------- #
def _stat_card(k: str, big: str, cap: str, tag: str, tag_col: str,
               frac: float | None, bar_col: str) -> str:
    bar = (f'<div class="bar"><i style="width:{max(0.0, min(1.0, frac))*100:.0f}%;'
           f'background:{bar_col}"></i></div>') if frac is not None else ""
    return (
        f'<div class="card stat"><div class="top"><span class="k">{esc(k)}</span>'
        f'<span class="tag" style="color:{tag_col}">{esc(tag)}</span></div>'
        f'<div class="big">{big}</div><span class="cap">{esc(cap)}</span>{bar}</div>'
    )


def _ledger(subjects: list[Subject]) -> str:
    rows = []
    for s in subjects:
        fg, bg = VERDICT_COLOURS.get(s.overall, ("#8A8079", "#F1ECE4"))
        q = s.qei
        if isinstance(q, (int, float)):
            qbar = (f'<span class="qbar"><i style="width:{max(0,min(1,q))*100:.0f}%;'
                    f'background:{fg}"></i></span>')
            qtxt = f'<span class="num">{q:.2f}</span>'
        else:
            qbar, qtxt = "", '<span class="num" style="color:var(--faint)">&mdash;</span>'
        rows.append(
            f'<tr><td class="sid">{esc(s.sid)}</td>'
            f'<td><span class="vpill" style="color:{fg};background:{bg}">{esc(s.overall)}</span></td>'
            f'<td><div class="qei-cell">{qbar}{qtxt}</div></td>'
            f'<td>{esc(s.primary_artifact)}</td>'
            f'<td><a class="view" href="/subject/{esc(s.sid)}">View report &rarr;</a></td></tr>'
        )
    return (
        '<table class="ledger"><thead><tr><th>Participant</th><th>Verdict</th>'
        '<th>QEI</th><th>Primary artifact</th><th>Action</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def _artifact_breakdown(summary: BatchSummary) -> str:
    top = summary.artifact_breakdown[:6]
    if not top:
        rows = '<p class="cap" style="color:var(--muted)">No checks flagged &mdash; a clean cohort.</p>'
    else:
        mx = max(n for _, n in top) or 1
        rows = "".join(
            f'<div class="row"><div class="lab"><span>{esc(lab)}</span>'
            f'<span class="n">{n}/{summary.total}</span></div>'
            f'<div class="track"><i style="width:{n/mx*100:.0f}%"></i></div></div>'
            for lab, n in top
        )
    return f'<div class="abd">{rows}</div>'


def _insight(summary: BatchSummary, cfg: QCConfig) -> str:
    worst = summary.artifact_breakdown[0][0] if summary.artifact_breakdown else None
    driver = (f" The most common flag is <b>{esc(worst)}</b>." if worst else "")
    return (
        f'<div class="insight">Analysed <b>{summary.total}</b> subjects against the '
        f'<b>{esc(cfg.population)}</b> quality profile: '
        f'<b>{summary.pass_rate*100:.0f}%</b> pass, {summary.warn_rate*100:.0f}% warn, '
        f'{summary.fail_rate*100:.0f}% fail.{driver}</div>'
    )


def render_overview(subjects: list[Subject], summary: BatchSummary,
                    cfg: QCConfig, dataset: str = "cohort") -> str:
    pass_col, warn_col, fail_col = (VERDICT_COLOURS["PASS"][0], VERDICT_COLOURS["WARN"][0],
                                    VERDICT_COLOURS["FAIL"][0])

    def tag_for(rate: float, good_high: bool) -> tuple[str, str]:
        if good_high:
            return ("OPTIMAL", pass_col) if rate >= 0.8 else \
                   ("MODERATE", warn_col) if rate >= 0.5 else ("LOW", fail_col)
        return ("CRITICAL", fail_col) if rate >= 0.4 else \
               ("ELEVATED", warn_col) if rate >= 0.15 else ("LOW", pass_col)

    stats = (
        _stat_card("Total scans", f"{summary.total}", "Analysed cohort", "AGGREGATE",
                   "var(--muted)", None, "")
        + _stat_card("Pass rate", f"{summary.pass_rate*100:.0f}<span class='pct'>%</span>",
                     "usable without caveats", *tag_for(summary.pass_rate, True),
                     frac=summary.pass_rate, bar_col=pass_col)
        + _stat_card("Warning", f"{summary.warn_rate*100:.0f}<span class='pct'>%</span>",
                     "review recommended", *tag_for(summary.warn_rate, False),
                     frac=summary.warn_rate, bar_col=warn_col)
        + _stat_card("Fail rate", f"{summary.fail_rate*100:.0f}<span class='pct'>%</span>",
                     "disqualifying problem", *tag_for(summary.fail_rate, False),
                     frac=summary.fail_rate, bar_col=fail_col)
    )
    content = (
        '<h1 class="page-h">Batch overview</h1>'
        f'<p class="page-sub">Dataset: {esc(dataset)} &middot; {summary.total} participants '
        f'&middot; graded against the {esc(cfg.population)} profile</p>'
        f'<div class="stats">{stats}</div>'
        '<div class="cols">'
        '<div class="card panel"><div class="ph"><h3>Participant ledger</h3>'
        '<span class="tag-muted">sorted by import order</span></div>'
        f'{_ledger(subjects)}</div>'
        '<div class="card panel"><h3>Artifact breakdown</h3>'
        '<p class="cap" style="color:var(--muted);margin:.1rem 0 .9rem">How often each check '
        'flagged a subject (FAIL or WARN).</p>'
        f'{_artifact_breakdown(summary)}{_insight(summary, cfg)}</div>'
        '</div>'
    )
    return _shell(subjects, active=None, view="overview",
                  title_html='osipy-qc <span class="v">&middot; cohort QC</span>',
                  content=content, cfg=cfg)


# --------------------------------------------------------------------------- #
# Subject Deep Dive
# --------------------------------------------------------------------------- #
def render_subject(subjects: list[Subject], subject: Subject, cfg: QCConfig) -> str:
    body = report_body(subject.report, subject.inputs, subject.cfg)
    title = (f'<a href="/" style="color:var(--muted)">Overview</a> '
             f'<span style="color:var(--faint)">/</span> '
             f'<span class="v">{esc(subject.sid)}</span>')
    return _shell(subjects, active=subject.sid, view="subject",
                  title_html=title, content=body, cfg=subject.cfg)
