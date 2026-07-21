"""
Shared front-end design system.

One place for the palette, the type scale, the logo and the base CSS, so the
upload console (web.py) and the report (report_html.py) are unmistakably the same
product. Everything here is inline: no webfonts, no CDN, no build step — the UI
works offline and ships inside the wheel.

Design direction — carried over from the project's own prototypes:
  * warm cream paper ground, not flat white
  * a single accent: OSIPI orange (perfusion warmth, and it matches the deck)
  * a system sans for reading, a mono for data/labels/metrics
  * calm, rounded, generously spaced cards; the brain map is the hero
  * semantic PASS / WARN / FAIL colours that are *separate* from the accent
"""

from __future__ import annotations

# Verdict -> (text colour, tint background). Semantic, deliberately not the accent.
VERDICT_COLOURS: dict[str, tuple[str, str]] = {
    "PASS": ("#1F9E5A", "#E7F5EC"),
    "WARN": ("#B87708", "#FBEFD8"),
    "FAIL": ("#C63D2E", "#FBE4E0"),
    "UNKNOWN": ("#8A8079", "#F1ECE4"),
    "N/A": ("#8A8079", "#F1ECE4"),
    "INFO": ("#3B6FA8", "#E7EEF6"),
}

# Provenance level -> (label, colour) for the little threshold badges.
PROVENANCE_COLOURS: dict[str, tuple[str, str]] = {
    "published": ("published", "#1F9E5A"),
    "implementation": ("implementation", "#3B6FA8"),
    "uncalibrated": ("uncalibrated", "#B87708"),
}

# A tasteful mark for osipy-qc — a stylized perfusion "flow" loop in the OSIPI
# orange. Deliberately NOT a copy of OSIPI's real logo; this brands the tool.
LOGO_SVG = (
    '<svg class="mark" width="30" height="30" viewBox="0 0 32 32" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<rect width="32" height="32" rx="9" fill="#E8590C"/>'
    '<path d="M9.5 20.5c0-3.2 2.4-5 4.6-5 3 0 3.8 3.4 6.8 3.4 1.8 0 3-1.4 3-3.2 '
    '0-2-1.4-3.2-3-3.2-2.9 0-3.8 3.4-6.8 3.4-2.2 0-3.4-1.4-3.4-3.1" '
    'stroke="#fff" stroke-width="2.1" stroke-linecap="round" fill="none"/>'
    '<circle cx="21" cy="12" r="1.6" fill="#fff"/></svg>'
)

# Base design system. Page-specific rules are appended by each page.
BASE_CSS = """
:root{
  --paper:#FBF7F1; --well:#F4ECE1; --surface:#FFFFFF;
  --ink:#1B1A18; --muted:#6E6660; --faint:#9C938B; --line:#EBE3D8;
  --accent:#E8590C; --accent-600:#C2571A; --accent-050:#FCEADB;
  --pass:#1F9E5A; --warn:#B87708; --fail:#C63D2E; --info:#3B6FA8;
  --radius:14px; --radius-sm:9px;
  --shadow:0 1px 2px rgba(27,26,24,.04), 0 8px 24px -12px rgba(27,26,24,.14);
  --shadow-lg:0 2px 6px rgba(27,26,24,.05), 0 24px 48px -20px rgba(27,26,24,.22);
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Inter,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);
  line-height:1.55;-webkit-font-smoothing:antialiased;font-feature-settings:"cv02","ss01"}
::selection{background:var(--accent-050)}
a{color:var(--accent-600);text-decoration:none}
a:hover{text-decoration:underline}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.mono{font-family:var(--mono)}
.num{font-variant-numeric:tabular-nums}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

/* top bar */
.topbar{display:flex;align-items:center;gap:.7rem;max-width:1080px;margin:0 auto;
  padding:1.1rem 1.5rem}
.brand{display:flex;align-items:center;gap:.6rem;font-weight:650;letter-spacing:-.01em}
.brand .mark{border-radius:9px;box-shadow:var(--shadow);flex:none}
.brand b{font-size:1.02rem}
.brand span{color:var(--muted);font-weight:450;font-size:.82rem;font-family:var(--mono)}
.topbar .spacer{flex:1}
.topbar .meta{font-family:var(--mono);font-size:.74rem;color:var(--muted);
  background:var(--surface);border:1px solid var(--line);border-radius:100px;
  padding:.3rem .75rem;white-space:nowrap}

/* chips + pills */
.chip{display:inline-flex;align-items:center;gap:.4rem;font-family:var(--mono);
  font-size:.72rem;letter-spacing:.02em;padding:.25rem .6rem;border-radius:100px;
  border:1px solid var(--line);background:var(--surface);color:var(--muted);white-space:nowrap}
.chip .dot{width:7px;height:7px;border-radius:50%;flex:none}
.badge{display:inline-block;font-family:var(--mono);font-size:.63rem;letter-spacing:.02em;
  padding:.08rem .4rem;border-radius:5px;color:#fff;vertical-align:middle}

/* buttons */
.btn{appearance:none;border:1px solid var(--line);background:var(--surface);color:var(--ink);
  font-family:var(--sans);font-size:.95rem;font-weight:550;padding:.7rem 1.1rem;
  border-radius:10px;cursor:pointer;transition:.15s ease;display:inline-flex;
  align-items:center;justify-content:center;gap:.5rem}
.btn:hover{border-color:var(--faint)}
.btn-primary{background:var(--accent);border-color:var(--accent);color:#fff;
  box-shadow:0 1px 2px rgba(232,89,12,.35),0 10px 20px -10px rgba(232,89,12,.6)}
.btn-primary:hover{background:var(--accent-600);border-color:var(--accent-600)}
.btn[disabled]{opacity:.6;cursor:progress}
.btn-ghost{background:transparent;border-color:transparent;color:var(--muted)}
.btn-ghost:hover{background:var(--well);border-color:transparent}

/* cards */
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);
  box-shadow:var(--shadow)}
.note{background:var(--accent-050);border:1px solid #F3D9C4;border-left:3px solid var(--accent);
  border-radius:0 var(--radius-sm) var(--radius-sm) 0;padding:.85rem 1.05rem;font-size:.9rem;color:#5c4632}

h1,h2,h3{margin:0;line-height:1.2;letter-spacing:-.015em}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;
  color:var(--accent-600);font-weight:600}
.section-title{font-size:.8rem;font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;
  color:var(--faint);margin:2.2rem 0 .8rem;font-weight:600}
.footer{max-width:1080px;margin:2.5rem auto 3rem;padding:1.2rem 1.5rem 0;border-top:1px solid var(--line);
  font-family:var(--mono);font-size:.74rem;color:var(--faint)}
"""


def esc(s) -> str:
    import html
    return html.escape(str(s))
