"""
Shared front-end design system.

One place for the palette, the type scale, the OSIPI logo and the base CSS, so
the upload console (web.py), the report (report_html.py) and the cohort
dashboard (dashboard_html.py) are unmistakably one product. Everything is inline:
no webfont, no CDN, no build step — the UI works offline and ships in the wheel.

Design direction — matched to the OSIPI QC-ToolBox proposal:
  * warm off-white paper (#FAF8F5), white cards
  * a terracotta accent (#AC4D2A) — the proposal's primary
  * teal for PASS, amber for WARN, brick-red for FAIL (the proposal's semantics)
  * the OSIPI pink -> purple infinity mark
  * clean, generously spaced, medical-report precision
"""

from __future__ import annotations

# Verdict -> (text colour, tint background). From the proposal palette.
# (ink, tint) per verdict — the same pairs the React app uses, so a printed
# report and the screen it came from are unmistakably one product.
VERDICT_COLOURS: dict[str, tuple[str, str]] = {
    "PASS": ("#1c6b4a", "#dff0e7"),
    "WARN": ("#7a5a08", "#f6ecd3"),
    "FAIL": ("#8f0f26", "#fadfe3"),
    "UNKNOWN": ("#63605a", "#eae8e2"),
    "N/A": ("#63605a", "#eae8e2"),
    "INFO": ("#1f5b96", "#e3ecf6"),
}

# The OSIPI mark — a pink->purple infinity loop, as on the proposal. Rendered
# for the user's own OSIPI project, matching their submitted branding.
LOGO_SVG = (
    '<svg class="mark" width="30" height="30" viewBox="0 0 40 40" fill="none" '
    'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<defs><linearGradient id="osipi_g" x1="4" y1="8" x2="36" y2="32" '
    'gradientUnits="userSpaceOnUse">'
    '<stop stop-color="#F0479B"/><stop offset="1" stop-color="#7C3AED"/></linearGradient></defs>'
    '<path d="M20 20c-3.4-4.6-6.4-7-9.6-7C6.3 13 4 16.1 4 20s2.3 7 6.4 7c3.2 0 '
    '6.2-2.4 9.6-7 3.4-4.6 6.4-7 9.6-7C33.7 13 36 16.1 36 20s-2.3 7-6.4 7c-3.2 '
    '0-6.2-2.4-9.6-7z" stroke="url(#osipi_g)" stroke-width="3.4" '
    'stroke-linecap="round" fill="none"/></svg>'
)


def brand(sub: str = "QC-ToolBox v1.0") -> str:
    """The OSIPI logo + wordmark block used in headers/sidebars."""
    return (
        '<div class="brand">' + LOGO_SVG +
        f'<span class="wm"><b>OSIPI</b><small>{sub}</small></span></div>'
    )


BASE_CSS = """
:root{
  /* One palette across the product. These are the values in
     web/src/index.css; the standalone report is the same thing as the app,
     printed, so it must not have a second look of its own. */
  --paper:#f6f7f9; --well:#f1f3f6; --surface:#ffffff;
  --ink:#1f1d1a; --muted:#57534a; --faint:#6b665c; --line:#e4e7ec;
  --accent:#c15f3c; --accent-600:#a34726; --accent-050:#fbeee8;
  --accent-fill:#bb5b38;
  --pass:#1c6b4a; --warn:#7a5a08; --fail:#8f0f26; --info:#1f5b96;
  --pass-mark:#1f7a54; --warn-mark:#96700c; --fail-mark:#a8142e; --info-mark:#256aad;
  --radius:20px; --radius-sm:12px; --radius-pane:26px;
  --shadow:0 1px 2px rgba(16,24,40,.04), 0 4px 8px -4px rgba(16,24,40,.05),
           0 16px 32px -12px rgba(16,24,40,.08);
  --shadow-lg:0 2px 4px rgba(16,24,40,.06), 0 18px 44px -20px rgba(16,24,40,.24);
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);
  line-height:1.55;-webkit-font-smoothing:antialiased;letter-spacing:0}
::selection{background:var(--accent-050)}
a{color:var(--accent-fill);text-decoration:none}
a:hover{text-decoration:underline}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.mono{font-family:var(--mono)}
.num{font-variant-numeric:tabular-nums}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}

/* brand block (logo + wordmark) */
.brand{display:flex;align-items:center;gap:.6rem}
.brand .mark{flex:none}
.brand .wm{display:flex;flex-direction:column;line-height:1.1}
.brand .wm b{font-size:1rem;font-weight:750;letter-spacing:.02em}
.brand .wm small{font-family:var(--mono);font-size:.62rem;letter-spacing:.08em;
  text-transform:uppercase;color:var(--faint)}

/* generic top bar (report + upload console) */
.topbar{display:flex;align-items:center;gap:.7rem;max-width:1120px;margin:0 auto;padding:1.1rem 1.5rem}
.topbar .spacer{flex:1}
.topbar .meta{font-family:var(--mono);font-size:.72rem;color:var(--muted);
  background:var(--surface);border:1px solid var(--line);border-radius:100px;padding:.3rem .75rem;white-space:nowrap}

/* chips + pills */
.chip{display:inline-flex;align-items:center;gap:.4rem;font-family:var(--mono);
  font-size:.72rem;padding:.25rem .6rem;border-radius:100px;border:1px solid var(--line);
  background:var(--surface);color:var(--muted);white-space:nowrap}
.chip .dot{width:7px;height:7px;border-radius:50%;flex:none}
.badge{display:inline-block;font-family:var(--mono);font-size:.63rem;padding:.08rem .4rem;
  border-radius:5px;color:#fff;vertical-align:middle}

/* buttons */
.btn{appearance:none;border:1px solid var(--line);background:var(--surface);color:var(--ink);
  font-family:var(--sans);font-size:.95rem;font-weight:550;padding:.7rem 1.1rem;border-radius:10px;
  cursor:pointer;transition:.15s ease;display:inline-flex;align-items:center;justify-content:center;gap:.5rem}
.btn:hover{border-color:var(--faint)}
.btn-primary{background:var(--accent);border-color:var(--accent);color:#fff;
  box-shadow:0 1px 2px rgba(172,77,42,.3),0 12px 22px -12px rgba(172,77,42,.6)}
.btn-primary:hover{background:var(--accent-600);border-color:var(--accent-600)}
.btn[disabled]{opacity:.6;cursor:progress}

/* cards */
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}
.note{background:var(--accent-050);border:1px solid #EFDBCC;border-left:3px solid var(--accent);
  border-radius:0 var(--radius-sm) var(--radius-sm) 0;padding:.85rem 1.05rem;font-size:.9rem;color:#5c4028}

h1,h2,h3{margin:0;line-height:1.18;letter-spacing:-.02em}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;
  color:var(--accent-600);font-weight:600}
.section-title{font-size:.76rem;font-family:var(--mono);letter-spacing:.09em;text-transform:uppercase;
  color:var(--faint);margin:2.2rem 0 .8rem;font-weight:600}
.footer{max-width:1120px;margin:2.5rem auto 3rem;padding:1.2rem 1.5rem 0;border-top:1px solid var(--line);
  font-family:var(--mono);font-size:.74rem;color:var(--faint)}
"""


def esc(s) -> str:
    import html
    return html.escape(str(s))
