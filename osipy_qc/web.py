"""
The frontend: upload a CBF map, get a report.

Why
---
Raised in the 2026-07 review: "even for uploading the data — how are they
uploading it right now? Just writing the Python command with flags?" A researcher
should not need the CLI to QC a scan.

`osipy-qc serve` starts a local web app: drag in a CBF map (and, if you have
them, the tissue maps), pick the population, get the full visual report back.

How, without dependencies
-------------------------
Built on `http.server` + `email` from the standard library, so the package stays
numpy + nibabel only — no Flask, no FastAPI, no build step, no node_modules. The
page is served as a single self-contained string; the report it returns is the
The self-contained HTML report is written by `osipy-qc --html`; this
server answers JSON and lets React render it.

Scope, honestly
---------------
This is a LOCAL tool. It binds to 127.0.0.1 by default and is deliberately not
hardened for public hosting: it accepts file uploads and hands them to a NIfTI
parser, and it has no authentication, no rate limiting and no sandboxing.
Deploying it to the open internet is a separate job (a real WSGI/ASGI server, a
job queue, upload limits, and an isolated worker) and is not what this module is
for.
"""

from __future__ import annotations

import http.server
import math
import os
import re
import socketserver
import tempfile
import traceback
import webbrowser

from ._webassets import BASE_CSS, brand, esc
from .core.config import POPULATIONS, for_population
from .report import run_qc

# Refuse absurd uploads outright rather than trying to parse them.
# The free Render container has 512 MB of RAM in total, and a request is copied
# more than once on its way through: read into memory, then split by the
# multipart boundary. A 512 MB ceiling therefore guaranteed an out-of-memory kill
# from a single request that was inside the advertised limit. Real inputs are
# 8-16 MB a file, so 64 MB is generous; the env var lets a bigger host raise it.
MAX_UPLOAD_BYTES = int(os.environ.get("OSIPY_MAX_UPLOAD_MB", "64")) * 1024 * 1024

# youngest -> oldest, for the segmented population control, with short display
# labels so nothing clips in the chips (the group heading already says "Population")
_POP_ORDER = ["adult", "neonate"]
_POP_LABELS = {"adult": "adult", "neonate": "neonate"}

_CONSOLE_CSS = """
.thr-note{font-size:.8rem;color:var(--muted);margin:.5rem 0 .9rem}
.thr-g{margin-bottom:1rem}
.thr-g h4{margin:0 0 .45rem;font-size:.74rem;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;color:var(--accent-600)}

.thr{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.7rem;margin:.7rem 0}
.thr-f{display:flex;flex-direction:column;gap:.25rem}
.thr-f span{font-size:.76rem;color:var(--muted)}
.thr-f input{border:1px solid var(--line);border-radius:var(--radius-sm);padding:.45rem .6rem;
  font:inherit;font-family:var(--mono);font-size:.85rem;background:var(--surface);color:var(--ink)}
.thr-f input:focus{outline:2px solid var(--accent);outline-offset:1px}
.strict{display:flex;gap:.55rem;align-items:flex-start;font-size:.82rem;color:var(--muted)}
.strict input{margin-top:.2rem}

.dropall{display:flex;align-items:center;gap:1rem;padding:1.5rem 1.4rem;min-height:132px;
  border:2px dashed var(--line);border-radius:var(--radius);background:var(--surface);
  transition:border-color .15s,background .15s}
.dropall.over{border-color:var(--accent);background:var(--accent-050)}
.dropbtns{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:.75rem}
.dbtn{display:inline-flex;align-items:center;min-height:38px;padding:0 1rem;
  font:inherit;font-size:.9rem;font-weight:600;cursor:pointer;
  border-radius:100px;border:1px solid transparent;
  background:var(--accent-fill);color:#fff;
  box-shadow:0 1px 2px rgba(16,24,40,.16)}
.dbtn:hover{background:var(--accent-600)}
.dbtn-alt{background:var(--surface);color:var(--accent-600);border-color:var(--line);
  box-shadow:none}
.dbtn-alt:hover{background:var(--accent-050);border-color:var(--accent);color:var(--accent-600)}
.dropall .ico{font-size:1.7rem;color:var(--accent);line-height:1}
.dropall .txt b{display:block;font-size:1.05rem}
.dropall .txt small{color:var(--muted)}
.picked{margin-left:auto;display:flex;flex-direction:column;gap:.2rem;max-width:46%}
.picked span{font-family:var(--mono);font-size:.72rem;color:var(--muted);
  display:flex;justify-content:space-between;gap:.6rem}
.picked span b{color:var(--accent-600);font-weight:600}
.manual{margin-top:1rem;border-top:1px solid var(--line);padding-top:.9rem}
.manual summary{cursor:pointer;font-size:.85rem;color:var(--muted)}
.manual summary:hover{color:var(--ink)}

.stage{max-width:760px;margin:0 auto;padding:1rem 1.5rem 4rem}
.lede{text-align:center;margin:1.5rem 0 2rem}
.lede h1{font-size:clamp(1.9rem,5vw,2.7rem);font-weight:730;margin:.6rem 0 .4rem}
.lede p{color:var(--muted);font-size:1.05rem;max-width:48ch;margin:0 auto}
.err{max-width:640px;margin:0 auto 1.4rem;background:#FBE4E0;border:1px solid #F1C7C0;
  border-left:3px solid var(--fail);border-radius:0 var(--radius-sm) var(--radius-sm) 0;
  padding:.8rem 1rem;font-size:.9rem;color:#7a2a20}
form{margin-top:.5rem}
.field-label{display:flex;align-items:baseline;gap:.5rem;margin:1.4rem 0 .5rem;font-weight:600}
.field-label .req{font-family:var(--mono);font-size:.66rem;color:var(--accent-600);
  background:var(--accent-050);padding:.1rem .4rem;border-radius:5px;letter-spacing:.03em}
.field-label .opt{font-family:var(--mono);font-size:.66rem;color:var(--faint)}
.drop{position:relative;display:flex;align-items:center;gap:.9rem;padding:1.1rem 1.2rem;
  border:1.5px dashed var(--line);border-radius:var(--radius);background:var(--surface);
  cursor:pointer;transition:.15s ease}
.drop:hover{border-color:var(--accent);background:#FFFDFB}
.drop.drag{border-color:var(--accent);background:var(--accent-050)}
.drop.filled{border-style:solid;border-color:#CFE7D8;background:#F4FBF7}
.drop input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer}
.drop .ic{width:38px;height:38px;border-radius:10px;background:var(--well);flex:none;
  display:grid;place-items:center;color:var(--accent-600)}
.drop.filled .ic{background:#E1F3E9;color:var(--pass)}
.drop .txt{flex:1;min-width:0}
.drop .txt b{font-size:.94rem}
.drop .txt small{display:block;color:var(--muted);font-size:.8rem;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.drop .clear{font-family:var(--mono);font-size:.72rem;color:var(--muted);border:0;background:none;
  cursor:pointer;padding:.2rem .4rem;display:none;z-index:2}
.drop.filled .clear{display:inline}
.tissue-grid{display:grid;grid-template-columns:1fr 1fr;gap:.7rem}
@media (max-width:560px){.tissue-grid{grid-template-columns:1fr}}
.tissue-grid .drop{padding:.85rem 1rem}
.tissue-grid .ic{width:30px;height:30px}
.seg{display:flex;flex-wrap:wrap;gap:.4rem}
.seg label{flex:1;min-width:86px;position:relative}
.seg input{position:absolute;opacity:0;width:0;height:0}
.seg span{display:block;text-align:center;padding:.5rem .3rem;border:1px solid var(--line);
  border-radius:9px;background:var(--surface);cursor:pointer;font-size:.8rem;
  font-family:var(--mono);color:var(--muted);transition:.12s ease;white-space:nowrap}
.seg span:hover{border-color:var(--faint)}
.seg input:checked + span{background:var(--accent);border-color:var(--accent);color:#fff;
  box-shadow:0 6px 14px -8px rgba(232,89,12,.7)}
.seg input:focus-visible + span{outline:2px solid var(--accent);outline-offset:2px}
.submit-row{margin-top:1.8rem}
.submit-row .btn-primary{width:100%;padding:.9rem;font-size:1.02rem}
.hint{font-size:.82rem;color:var(--muted);margin:.4rem 0 0}
#overlay{position:fixed;inset:0;background:rgba(251,247,241,.86);backdrop-filter:blur(3px);
  display:none;place-items:center;z-index:50}
#overlay.on{display:grid}
.spin{width:42px;height:42px;border:4px solid var(--accent-050);border-top-color:var(--accent);
  border-radius:50%;animation:spin .8s linear infinite;margin:0 auto .9rem}
@keyframes spin{to{transform:rotate(360deg)}}
#overlay .msg{text-align:center;color:var(--muted);font-family:var(--mono);font-size:.85rem}
"""

_FILE_IC = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>'
            '<polyline points="13 2 13 9 20 9"/></svg>')


def _threshold_groups() -> tuple[str, str]:
    """Every tunable threshold, grouped, plus both populations' defaults as JSON.

    The defaults are shipped to the page so choosing adult or neonate can update
    what each field shows. Ten of them genuinely differ — a neonate's normal GM
    CBF is around 16, where an adult's band starts at 40 — so a form that showed
    adult numbers under a neonate heading would be telling the reader something
    untrue about the grading they are about to get.
    """
    import json as _json

    from .batch import TUNABLE_GROUPS
    from .core.config import for_population

    defaults = {pop: {name: getattr(for_population(pop), name)
                      for _g, fields in TUNABLE_GROUPS for name, _l in fields}
                for pop in ("adult", "neonate")}

    out: list[str] = []
    for group, fields in TUNABLE_GROUPS:
        cells = "".join(
            f'<label class="thr-f"><span>{esc(label)}</span>'
            f'<input type="number" name="thr_{name}" step="any" inputmode="decimal" '
            f'data-thr="{name}" placeholder="{defaults["adult"][name]:g}"></label>'
            for name, label in fields
        )
        out.append(f'<div class="thr-g"><h4>{esc(group)}</h4><div class="thr">{cells}</div></div>')
    return "".join(out), _json.dumps(defaults)


def _dropzone(field: str, title: str, hint: str, required: bool = False) -> str:
    req = " required" if required else ""
    return (
        f'<div class="drop" data-field="{field}">'
        f'<div class="ic">{_FILE_IC}</div>'
        f'<div class="txt"><b>{esc(title)}</b><small data-hint>{esc(hint)}</small></div>'
        f'<button type="button" class="clear" aria-label="remove">clear</button>'
        f'<input type="file" name="{field}" accept=".nii,.gz,application/gzip,application/x-gzip,application/octet-stream"{req}></div>'
    )


def _upload_page(error: str = "") -> str:
    pops = [p for p in _POP_ORDER if p in POPULATIONS]
    seg = "".join(
        f'<label><input type="radio" name="population" value="{p}"'
        f'{" checked" if p == "adult" else ""}>'
        f'<span title="{p}">{esc(_POP_LABELS.get(p, p))}</span></label>'
        for p in pops
    )
    thr_groups, thr_defaults = _threshold_groups()
    err = f'<div class="err"><b>Could not grade that.</b> {esc(error)}</div>' if error else ""
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>osipy-qc &mdash; ASL quality control</title>
<style>{BASE_CSS}{_CONSOLE_CSS}</style></head><body>
<div class="topbar">{brand("ASL quality control")}</div>

<div class="stage">
  <div class="lede">
    <div class="eyebrow">CBF map &rarr; PASS / WARN / FAIL</div>
    <h1>Grade an ASL scan</h1>
    <p>Drop in a CBF map and get an interpretable quality report &mdash; a verdict per
       check, with the reason and the reference behind every number.</p>
  </div>
  {err}
  <form id="qc" method="post" action="/run" enctype="multipart/form-data">
    <div class="field-label">CBF map <span class="req">required</span></div>
    {_dropzone("cbf", "Choose or drop a CBF map", "quantified CBF (mL/100g/min), NIfTI")}

    <div class="field-label">Tissue maps
      <span class="opt">optional &mdash; needed for QEI &amp; level checks</span></div>
    <div class="grid2">
      {_dropzone("gm", "Grey matter", "GM probability map")}
      {_dropzone("wm", "White matter", "WM probability map")}
      {_dropzone("csf", "CSF", "derived if omitted")}
      <div class="drop" style="border:none;background:transparent;box-shadow:none;cursor:default">
        <div class="txt"><small>Must share the <b>same voxel grid</b> as the CBF map.</small></div></div>
    </div>

    <div class="field-label">Raw acquisition
      <span class="opt">optional &mdash; unlocks the schema, M0, motion and data-type checks</span></div>
    <div class="drop dropall" id="zone">
      <input id="files" name="files" type="file" accept=".nii,.gz,application/gzip,application/x-gzip,application/octet-stream" multiple hidden>
      <input id="folder" name="files" type="file" webkitdirectory directory multiple hidden>
      <div class="ico">&#8615;</div>
      <div class="txt">
        <b>Drop the raw files here</b>
        <small>The ASL series, M0 and structural. Each is recognised by its filename;
        if a name is unusual, use the boxes underneath instead.</small>
        <div class="dropbtns">
          <button type="button" id="pickfiles" class="dbtn">Choose files&hellip;</button>
          <button type="button" id="pickdir" class="dbtn dbtn-alt">Choose a folder&hellip;</button>
        </div>
      </div>
      <div class="picked" id="picked"></div>
    </div>

    <details class="manual" id="byrole">
      <summary>Or say what each file is &mdash; use this if a name is not recognised</summary>
      <p class="thr-note">Nothing here depends on the filename. Whatever you put in a box is
        treated as that kind of file.</p>
      <div class="grid2">
        {_dropzone("raw_asl", "ASL series", "4D control/label, pairs, or pre-subtracted dM")}
        {_dropzone("raw_m0", "M0", "the calibration scan")}
        {_dropzone("raw_t1", "Structural", "T1 / MPRAGE")}
      </div>
    </details>

    <details class="manual">
      <summary>Thresholds &mdash; change what counts as a pass</summary>
      <p class="thr-note">Every field is optional. Leave one blank and the packaged value for
        the chosen population is used &mdash; the greyed number is what that would be.</p>
      {thr_groups}
      <label class="strict">
        <input type="checkbox" name="strict" value="1" checked>
        <span>Strict &mdash; an uncalibrated cut-off may raise a FAIL. Turn off to
        demote those to a WARN.</span>
      </label>
    </details>

    <div class="field-label">Population <span class="opt">newborn CBF is far lower than adult</span></div>
    <div class="seg">{seg}</div>
    <p class="hint">A neonate's normal GM CBF (~16) would look abnormal against the adult
       40&ndash;100 band, so pick <b>neonate</b> for newborn scans.</p>

    <div class="submit-row">
      <button type="submit" class="btn btn-primary">Grade scan &rarr;</button>
    </div>
  </form>
</div>

<div id="overlay"><div class="msg"><div class="spin"></div>Grading the scan&hellip;</div></div>

<script>
(function(){{
  document.querySelectorAll('.drop[data-field]').forEach(function(dz){{
    var input = dz.querySelector('input[type=file]');
    var hint  = dz.querySelector('[data-hint]');
    var base  = hint.textContent;
    function refresh(){{
      if(input.files && input.files.length){{
        dz.classList.add('filled'); hint.textContent = input.files[0].name;
      }} else {{ dz.classList.remove('filled'); hint.textContent = base; }}
    }}
    input.addEventListener('change', refresh);
    ['dragenter','dragover'].forEach(function(e){{
      dz.addEventListener(e, function(ev){{ ev.preventDefault(); dz.classList.add('drag'); }}); }});
    ['dragleave','drop'].forEach(function(e){{
      dz.addEventListener(e, function(ev){{ ev.preventDefault(); dz.classList.remove('drag'); }}); }});
    dz.addEventListener('drop', function(ev){{
      if(ev.dataTransfer && ev.dataTransfer.files.length){{ input.files = ev.dataTransfer.files; refresh(); }} }});
    dz.querySelector('.clear').addEventListener('click', function(ev){{
      ev.stopPropagation(); input.value=''; refresh(); }});
  }});

  // progressive enhancement: fetch + render in place with a loading overlay.
  // With JS off, the form does a normal POST and still works.
  var form = document.getElementById('qc');
  var overlay = document.getElementById('overlay');
  // Choosing a population rewrites what each threshold field shows, because ten
  // of them genuinely differ: a neonate's normal GM CBF is about 16, where the
  // adult band starts at 40. Showing adult numbers under a neonate heading would
  // misdescribe the grading about to happen.
  var THR = {thr_defaults};
  function applyPopulation(){{
    var pop = (document.querySelector('input[name="population"]:checked') || {{}}).value || 'adult';
    var d = THR[pop] || THR.adult;
    document.querySelectorAll('input[data-thr]').forEach(function(el){{
      var v = d[el.getAttribute('data-thr')];
      if(v !== undefined) el.placeholder = String(v);
    }});
  }}
  document.querySelectorAll('input[name="population"]').forEach(function(r){{
    r.addEventListener('change', applyPopulation);
  }});
  applyPopulation();

  var multi = document.getElementById('files');
  var picked = document.getElementById('picked');
  var zone = document.getElementById('zone');
  // Mirrors classify_role in checks/schema.py. When these two disagree the page
  // tells the reader their file is unusable while the server would have taken
  // it — which is exactly what happened to a reviewer's conAFeHe73_1_pairs.nii.
  function role(n){{
    n = n.toLowerCase();
    if(/m0|calib/.test(n)) return 'M0';
    if(/pcasl|pasl|asl|perf|cbf|deltam|delta_m|pair|control|ctrl|label|tag|subtract/.test(n)
       || /^(con|tag|diff|dm)/.test(n)) return 'ASL series';
    if(/mprage|t1|anat|struct/.test(n)) return 'structural';
    return 'name unclear \u2014 use the boxes below';
  }}
  function show(src){{
    picked.innerHTML = '';
    for(var i=0;i<src.files.length;i++){{
      var f = src.files[i], s = document.createElement('span');
      s.innerHTML = '<span>'+f.name+'</span><b>'+role(f.name)+'</b>';
      picked.appendChild(s);
    }}
  }}
  var folder = document.getElementById('folder');
  document.getElementById('pickfiles').addEventListener('click', function(ev){{
    ev.preventDefault(); multi.click();
  }});
  document.getElementById('pickdir').addEventListener('click', function(ev){{
    ev.preventDefault(); folder.click();
  }});
  // a folder carries everything in it, so only the NIfTIs are kept
  folder.addEventListener('change', function(){{
    var keep = [];
    for(var i=0;i<folder.files.length;i++){{
      var n = folder.files[i].name.toLowerCase();
      if(n.endsWith('.nii') || n.endsWith('.nii.gz')) keep.push(folder.files[i]);
    }}
    var dt = new DataTransfer();
    keep.forEach(function(f){{ dt.items.add(f); }});
    folder.files = dt.files;
    show(folder);
  }});
  multi.addEventListener('change', function(){{ show(multi); }});
  ['dragenter','dragover'].forEach(function(e){{
    zone.addEventListener(e, function(ev){{ ev.preventDefault(); zone.classList.add('over'); }});
  }});
  ['dragleave','drop'].forEach(function(e){{
    zone.addEventListener(e, function(ev){{ ev.preventDefault(); zone.classList.remove('over'); }});
  }});
  zone.addEventListener('drop', function(ev){{ multi.files = ev.dataTransfer.files; show(multi); }});

  form.addEventListener('submit', function(ev){{
    if(!window.fetch){{ return; }}
    ev.preventDefault();
    overlay.classList.add('on');
    fetch('/run', {{method:'POST', body:new FormData(form)}})
      .then(function(r){{ return r.text(); }})
      .then(function(html){{
        // give the report a real address: reloadable, shareable, and the back
        // button returns to the form rather than to nothing
        history.pushState({{}}, '', '/result');
        document.open(); document.write(html); document.close();
      }})
      .catch(function(){{ overlay.classList.remove('on'); form.submit(); }});
  }});
}})();
</script>
</body></html>"""


def _parse_multipart(body: bytes, content_type: str) -> dict[str, tuple[str, bytes]]:
    """{field: (filename, bytes)} from a multipart/form-data body.

    Deliberately a byte-level parser. The obvious shortcut — feeding the body to
    the stdlib `email` module — silently CORRUPTS binary uploads: it decodes the
    payload to str internally, so a gzipped NIfTI comes back out unparseable
    ("zlib.error: invalid literal/lengths set"). Never let a MIME text parser
    near a .nii.gz.

    So the bytes are sliced directly and never decoded: only the headers are
    turned into text, and the payload is passed through untouched.
    """
    m = re.search(r'boundary=(?:"([^"]+)"|([^;,\s]+))', content_type or "")
    if not m:
        raise ValueError("malformed upload: no multipart boundary in Content-Type")
    boundary = (m.group(1) or m.group(2)).encode()
    delim = b"--" + boundary

    out: dict[str, tuple[str, bytes]] = {}
    for chunk in body.split(delim):
        if chunk[:2] == b"\r\n":
            chunk = chunk[2:]              # strip only the separator's own CRLF
        if not chunk or chunk[:2] == b"--":
            continue                        # preamble or the closing '--'
        split = chunk.find(b"\r\n\r\n")
        if split < 0:
            continue
        headers = chunk[:split].decode("utf-8", "replace")
        data = chunk[split + 4:]
        if data[-2:] == b"\r\n":
            data = data[:-2]                # exactly the CRLF before the next boundary
        name = filename = None
        for line in headers.split("\r\n"):
            if line.lower().startswith("content-disposition"):
                nm = re.search(r'name="([^"]*)"', line)
                fn = re.search(r'filename="([^"]*)"', line)
                name = nm.group(1) if nm else None
                filename = fn.group(1) if fn else ""
        if name:
            # A multi-file input sends one part per file, all under the same
            # name, so those are collected rather than overwriting each other.
            # Single-file fields keep their (filename, bytes) shape so every
            # existing caller is unaffected.
            if name == "files":
                out.setdefault("files_multi", []).append((filename or "", data))
            else:
                out[name] = (filename or "", data)
    return out


# The last few graded uploads, kept so their figures can be rendered on demand.
# Bounded on purpose: the arrays are large and this is a single-process server.
#
# Keyed by an unguessable token, NEVER by the uploaded filename. Keying by
# filename made every upload readable by anyone who guessed the name, and
# "perfusion_calib.nii.gz" is what half the world's oxford_asl output is called:
# one visitor could read another's report and their brain images.
_UPLOADS: dict[str, object] = {}
_UPLOAD_KEEP = 4


def _remember_upload(subject) -> str:
    import secrets
    token = secrets.token_urlsafe(18)
    _UPLOADS[token] = subject
    while len(_UPLOADS) > _UPLOAD_KEEP:
        _UPLOADS.pop(next(iter(_UPLOADS)))
    return token


def _grade_upload(fields: dict[str, tuple[str, bytes]]) -> dict:
    """Write the uploaded NIfTIs to a temp dir, grade them, return the payload.

    Returns the same JSON shape as /api/subject/<id>, so the React report page
    renders an upload with exactly the components it uses for a cohort scan.
    The self-contained HTML report still exists — it is what `osipy-qc --html`
    writes — but it is no longer what the browser gets back from an upload.
    """
    from .io import load_cbf_inputs

    from .checks.schema import classify_upload
    cbf_name, cbf_bytes = fields.get("cbf", ("", b""))

    population = (fields.get("population", ("", b"adult"))[1] or b"adult").decode()
    cfg = for_population(population)

    # Threshold overrides from the form. Anything left blank keeps the packaged
    # value, and a field that is not a number is ignored rather than silently
    # becoming zero — a zero cut-off would pass everything.
    for key, value in fields.items():
        if not key.startswith("thr_") or not isinstance(value, tuple):
            continue
        raw = (value[1] or b"").decode(errors="replace").strip()
        if not raw:
            continue
        name = key[4:]
        if not hasattr(cfg, name):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        # "nan" and "inf" parse happily, and then every `x >= cutoff` is False,
        # so a clean scan falls through to FAIL against a cut-off of nan. The
        # documented contract is that a field which is not a number is ignored.
        if not math.isfinite(val):
            continue
        setattr(cfg, name, val)
    if "strict" in fields or "thr_qei_pass" in fields:
        # only a form that actually carries the control may change it; an upload
        # that says nothing about strictness keeps the packaged default rather
        # than silently grading leniently
        cfg.strict = bool(fields.get("strict", ("", b""))[1])

    with tempfile.TemporaryDirectory(prefix="osipy_qc_") as tmp:
        def _save(field: str) -> str | None:
            fname, data = fields.get(field, ("", b""))
            if not data:
                return None
            # Never trust an uploaded filename; keep only the extension we need.
            ext = ".nii.gz" if fname.endswith(".gz") else ".nii"
            path = os.path.join(tmp, field + ext)
            with open(path, "wb") as fh:
                fh.write(data)
            return path

        paths = {k: _save(k) for k in ("cbf", "gm", "wm", "csf")}
        if not paths["cbf"]:
            raise ValueError("No CBF map was uploaded.")
        inputs = load_cbf_inputs(paths["cbf"], gm=paths["gm"], wm=paths["wm"],
                                 csf=paths["csf"])

        # Raw acquisition files, if any were sent. These are what Stream A needs:
        # without them the schema, M0, motion and data-type checks have nothing
        # to look at and correctly return UNKNOWN.
        #
        # Their FILENAMES carry the signal here — classify_role reads them to tell
        # an ASL series from an M0 from a T1 — so unlike the CBF map they cannot
        # simply be renamed to the form field. They are sanitised instead: the
        # basename is stripped of any path and reduced to safe characters, which
        # keeps the role legible without trusting the client.
        # Name the folder after the client's own, when a folder upload told us
        # what it was. detect_dataset reads it for the vendor and for whether
        # background suppression was on, and BS decides whether the control/label
        # swap check applies at all — so losing the name changed the verdict.
        client_dir = ""
        for fname, _data in list(fields.get("files_multi", [])):
            head = os.path.dirname(fname.replace("\\", "/")).strip("/")
            if head:
                client_dir = head.split("/")[0]
                break
        safe_dir = re.sub(r"[^A-Za-z0-9._-]", "_", client_dir)[:60] or "raw"
        raw_dir = os.path.join(tmp, safe_dir)
        saved_raw = os.path.isdir(raw_dir)
        raw_parts = list(fields.get("files_multi", []))
        for field, value in fields.items():          # the per-role fields still work
            if field.startswith("raw") and isinstance(value, tuple) and value[1]:
                raw_parts.append(value)

        for fname, data in raw_parts:
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(fname))[-80:]
            if not safe.endswith((".nii", ".nii.gz")):
                safe += ".nii.gz"
            os.makedirs(raw_dir, exist_ok=True)
            with open(os.path.join(raw_dir, safe), "wb") as fh:
                fh.write(data)
            saved_raw = True

        if saved_raw:
            from .io import load_folder
            # the CBF-derived inputs win where the two overlap; the raw folder
            # only adds what it alone can know
            inputs = {**load_folder(raw_dir), **inputs}
        # Grade only what was actually supplied. Running the whole registry on a
        # CBF map alone leaves ten Stream-A checks with nothing to look at; each
        # returns UNKNOWN, and the overall verdict is dragged to WARN — so a
        # flawless map could never PASS through this console. Verified: the clean
        # synthetic case is WARN with every check and PASS with the CBF set.
        from .batch import cbf_map_checks
        if saved_raw:
            from .core.registry import all_checks
            # 4.1 needs an ASL mask and a structural mask, and no loader in the
            # package produces either, so including it pins even a full raw
            # upload at WARN.
            checks = [n for n in all_checks() if n != "4.1.coregistration"]
        else:
            checks = cbf_map_checks()
        report = run_qc(inputs, cfg=cfg, checks=checks)

        from .api import subject_payload
        from .batch import Subject
        sid = (os.path.basename(cbf_name)
               or os.path.basename(paths["cbf"] or "")
               or "uploaded scan")
        subject = Subject(sid=sid, report=report, inputs=inputs, cfg=cfg)
        # hold the graded arrays briefly so /figure/ can render from them; the
        # temp dir is gone by then, but the loaded arrays are not
        token = _remember_upload(subject)
        payload = subject_payload(subject)
        payload["uploaded"] = True
        payload["token"] = token          # figure URLs address the token, not the name
        return payload


def _grade_upload_html(fields: dict[str, tuple[str, bytes]]) -> tuple[str, str]:
    """The same grading, rendered as the self-contained HTML report.

    Used by the no-build console and by any client that did not ask for JSON.
    Both paths call _grade_upload first, so the grading cannot differ between
    them — only the presentation does.
    """
    from .report_html import render_html
    payload = _grade_upload(fields)             # grades and remembers the subject
    subject = _UPLOADS[payload["token"]]
    html = render_html(subject.report, inputs=subject.inputs, cfg=subject.cfg,
                       served=True, title=f"ASL QC report — {subject.sid}")
    return html, payload["token"]


def _safe_back(raw: str) -> str:
    """Sanitise a user-supplied redirect target. Only a local, single-slash-rooted
    path is allowed — never an absolute URL (open redirect) and never one carrying
    CR/LF (HTTP response splitting into the Location header)."""
    if not raw or not raw.startswith("/") or raw.startswith("//"):
        return "/"
    if any(c in raw for c in "\r\n"):
        return "/"
    return raw


def _error_page(code: int, msg: str) -> str:
    """A styled, self-contained error page (charset + head), not a bare <h1>."""
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{code}</title><style>{BASE_CSS}</style></head><body>"
        "<div style='max-width:560px;margin:14vh auto;padding:0 1.5rem;text-align:center'>"
        "<div style='display:inline-flex'>" + brand("ASL quality control") + "</div>"
        f"<h1 style='font-size:3rem;margin:1.4rem 0 .3rem'>{code}</h1>"
        f"<p style='color:var(--muted)'>{esc(msg)}</p>"
        "<p style='margin-top:1.4rem'><a class='btn' href='/'>Back to overview</a></p>"
        "</div></body></html>"
    )


class QCHandler(http.server.BaseHTTPRequestHandler):
    server_version = "osipy-qc"
    timeout = 60                    # drop a stalled/slow-loris connection

    def _host_ok(self) -> bool:
        """Reject cross-origin Host headers when bound to loopback (DNS-rebinding
        defence). If the operator explicitly bound a non-loopback address, don't
        second-guess them."""
        bound = self.server.server_address[0]
        if bound not in ("127.0.0.1", "::1", "localhost"):
            return True
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in ("127.0.0.1", "localhost", "::1", "")

    def log_message(self, fmt, *args):          # quieter than the noisy default
        print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    def _send(self, body: str, status: int = 200, extra_headers=None) -> None:
        raw = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        for name, value in (extra_headers or []):
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(raw)

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    # ---- JSON API plumbing ------------------------------------------------ #
    def _send_json(self, obj, status: int = 200) -> None:
        import json
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_bytes(self, raw: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _serve_spa(self, path: str) -> bool:
        """Serve the built React app from web/dist, if it has been built.

        Returns True if the request was handled.

        A path that looks like a file (it has an extension) but does not exist is
        a genuinely missing asset, so it 404s. Anything else is treated as a
        client-side route and gets index.html, which is what makes a deep link
        survive a hard refresh.
        """
        import mimetypes
        import os
        import posixpath

        root = _spa_root()
        if root is None:
            return False
        index = os.path.join(root, "index.html")
        rel = path.lstrip("/") or "index.html"
        target = os.path.normpath(os.path.join(root, rel))
        # never serve outside the build directory
        if not target.startswith(root) or not os.path.isfile(target):
            if "." in posixpath.basename(path):
                return False            # a missing asset, not a route
            target = index
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        with open(target, "rb") as fh:
            raw = fh.read()
        if ctype.startswith("text/") or "javascript" in ctype or "json" in ctype:
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        # hashed asset filenames are immutable; index.html must not be cached
        self.send_header("Cache-Control",
                         "no-store" if target == index else "public, max-age=31536000, immutable")
        self.end_headers()
        self.wfile.write(raw)
        return True

    def _api(self, path: str, query: dict, batch: dict | None) -> bool:
        """Handle an /api/... GET. Returns True if the route matched."""
        from . import api as apimod

        if path == "/api/health":
            self._send_json({"ok": True, "hasCohort": batch is not None})
            return True

        if path == "/api/provenance":
            self._send_json(apimod.provenance_payload())
            return True

        if path == "/api/checks":
            self._send_json(apimod.checks_catalogue())
            return True

        # A graded upload is addressable whether or not a cohort is loaded — the
        # upload console runs with no cohort at all, and its report still needs
        # its figures.
        if path.startswith("/api/subject/") and _UPLOADS:
            import urllib.parse as up
            rest = up.unquote(path[len("/api/subject/"):])
            token, _, fig = rest.partition("/figure/")
            sub = _UPLOADS.get(token if fig else rest)
            if sub is not None:
                if fig:
                    try:
                        raw, ctype = apimod.figure_bytes(sub, fig.rsplit(".", 1)[0])
                    except KeyError:
                        self._send_json({"error": f"unknown figure {fig}"}, 404)
                        return True
                    self._send_bytes(raw, ctype)
                else:
                    self._send_json(apimod.subject_payload(sub))
                return True

        if batch is None:
            # single-scan mode: no cohort loaded
            if path.startswith("/api/"):
                self._send_json({"error": "no cohort loaded; start with --dashboard"}, 404)
                return True
            return False

        if path == "/api/cohort":
            subjects, summary, cfg = self._effective(batch)
            self._send_json(apimod.cohort_payload(
                subjects, summary, cfg, batch.get("dataset", "cohort"),
                batch.get("overrides") or {}))
            return True

        if path == "/api/config":
            _subjects, _summary, cfg = self._effective(batch)
            self._send_json(apimod.config_payload(cfg, batch.get("overrides") or {}))
            return True

        if path.startswith("/api/subject/"):
            import urllib.parse as up
            rest = up.unquote(path[len("/api/subject/"):])
            subjects, _summary, _cfg = self._effective(batch)
            if "/figure/" in rest:
                sid, fig = rest.split("/figure/", 1)
                sub = next((s for s in subjects if s.sid == sid), None) or _UPLOADS.get(sid)
                if sub is None:
                    self._send_json({"error": f"unknown subject {sid}"}, 404)
                    return True
                try:
                    raw, ctype = apimod.figure_bytes(sub, fig.rsplit(".", 1)[0])
                except KeyError:
                    self._send_json({"error": f"unknown figure {fig}"}, 404)
                    return True
                self._send_bytes(raw, ctype)
                return True
            sub = next((s for s in subjects if s.sid == rest), None) or _UPLOADS.get(rest)
            if sub is None:
                self._send_json({"error": f"unknown subject {rest}"}, 404)
                return True
            self._send_json(apimod.subject_payload(sub))
            return True

        self._send_json({"error": "unknown endpoint"}, 404)
        return True

    def _effective(self, batch: dict):
        """(subjects, summary, cfg) for the currently applied overrides, memoised
        so images are only re-rendered when the config actually changes."""
        import urllib.parse as up

        from .batch import cfg_from_params, regrade, summarise

        overrides = batch.get("overrides") or {}
        key = tuple(sorted(overrides.items()))
        cache = batch.get("_cache")
        if cache and cache[0] == key:
            return cache[1]
        cfg = cfg_from_params(batch["base_cfg"], dict(overrides))
        subjects = regrade(batch["base_subjects"], cfg) if overrides else batch["base_subjects"]
        result = (subjects, summarise(subjects), cfg)
        batch["_cache"] = (key, result)
        return result

    def do_GET(self):
        import urllib.parse as up

        if not self._host_ok():
            self._send(_error_page(403, "Cross-origin request refused."), 403)
            return
        try:
            batch = getattr(self.server, "batch", None)     # set in dashboard mode
            parsed = up.urlparse(self.path)
            path, query = parsed.path, up.parse_qs(parsed.query)

            # the JSON API the React client consumes
            if path.startswith("/api/") and self._api(path, query, batch):
                return

            # apply a config change, then bounce back to where the user was
            if batch and path == "/apply":
                batch["overrides"] = {k: v[0] for k, v in query.items() if k != "back"}
                self._redirect(_safe_back(query.get("back", ["/"])[0]))
                return

            # React owns the UI. There is no second, server-rendered copy of it
            # to drift out of step; the only page rendered here is the upload
            # console, which must work before any build exists.
            if getattr(self.server, "spa", True) and self._serve_spa(path):
                return

            if path == "/result":
                token = ""
                for part in (self.headers.get("Cookie") or "").split(";"):
                    k, _, v = part.strip().partition("=")
                    if k == "osipy_result":
                        token = v
                sub = _UPLOADS.get(token)
                if sub is not None:
                    from .report_html import render_html
                    self._send(render_html(sub.report, inputs=sub.inputs, cfg=sub.cfg,
                                           served=True, title=f"ASL QC report — {sub.sid}"))
                else:
                    self._send(_upload_page())
                return
            if path in ("/", "/index.html", "/upload"):
                self._send(_upload_page())
            else:
                self._send(_error_page(404, "That page does not exist."), 404)
        except Exception as exc:            # a render error must not leak a traceback
            traceback.print_exc()
            self._send(_error_page(500, f"{type(exc).__name__}: {exc}"), 500)

    def do_POST(self):
        if not self._host_ok():
            self._send(_error_page(403, "Cross-origin request refused."), 403)
            return

        # apply threshold overrides from the React client and re-grade
        if self.path == "/api/config":
            import json

            from . import api as apimod
            batch = getattr(self.server, "batch", None)
            if batch is None:
                self._send_json({"error": "no cohort loaded"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b"{}"
                payload = json.loads(body or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object")
                # only known config fields may be set, and only as scalars
                allowed = set(apimod.config_fields())
                overrides = {k: str(v) for k, v in payload.items()
                             if k in allowed and not isinstance(v, (dict, list))}
                batch["overrides"] = overrides
                subjects, summary, cfg = self._effective(batch)
                self._send_json(apimod.cohort_payload(
                    subjects, summary, cfg, batch.get("dataset", "cohort"), overrides))
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 400)
            return

        if self.path != "/run":
            self._send(_error_page(404, "That page does not exist."), 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                raise ValueError("Empty request.")
            if length > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"Upload is {length/1e6:.0f} MB; the limit is "
                    f"{MAX_UPLOAD_BYTES/1e6:.0f} MB."
                )
            body = self.rfile.read(length)
            fields = _parse_multipart(body, self.headers.get("Content-Type", ""))
            wants_json = "application/json" in (self.headers.get("Accept") or "")
            if wants_json:
                self._send_json(_grade_upload(fields))
            else:
                html, token = _grade_upload_html(fields)
                # the cookie is what makes /result yours and not the next
                # visitor's; HttpOnly so script cannot read it, Strict so it is
                # not sent from another site
                self._send(html, extra_headers=[
                    ("Set-Cookie",
                     f"osipy_result={token}; Path=/; HttpOnly; SameSite=Strict")])
        except Exception as exc:                # a bad upload must not kill the server
            traceback.print_exc()
            if "application/json" in (self.headers.get("Accept") or ""):
                self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 400)
            else:
                self._send(_upload_page(error=f"{type(exc).__name__}: {exc}"), 400)


def _spa_root() -> str | None:
    """Locate the built React app, or None if it has not been built.

    Three places, because an installed package and a source checkout do not put
    it in the same one — and the difference is invisible locally, where the venv
    usually points straight at the source tree:

      1. inside the installed package (osipy_qc/webui), which is how a wheel
         ships it;
      2. beside the package in a source checkout (../web/dist);
      3. under the current working directory, which is what a platform that
         builds and then runs from the repo root ends up with.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "webui"),
                 os.path.join(os.path.dirname(here), "web", "dist"),
                 os.path.join(os.getcwd(), "web", "dist")):
        if os.path.isfile(os.path.join(cand, "index.html")):
            return cand
    return None


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    batch = None                # set to a dict{subjects,summary,cfg,dataset} for the dashboard
    spa = True                  # False in --serve mode: the HTML console owns the root


def _run(httpd, url: str, banner: str, open_browser: bool) -> None:
    print(banner)
    print(f"  {url}   (Ctrl-C to stop)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Run the single-scan upload console, which answers with the HTML report.

    The built React app is deliberately not served here: this mode exists to be
    the thing that works with no build step, and letting the SPA claim the root
    would make the mode indistinguishable from --dashboard.
    """
    with _Server((host, port), QCHandler) as httpd:
        httpd.spa = False
        _run(httpd, f"http://{host}:{port}/", "osipy-qc upload console running at", open_browser)


def serve_dashboard(subjects, cfg=None, dataset: str = "cohort",
                    host: str = "127.0.0.1", port: int = 8000,
                    open_browser: bool = True) -> None:
    """Run the cohort dashboard over an already-graded list of Subjects."""
    from .batch import summarise
    from .core.config import QCConfig

    cfg = cfg or QCConfig()
    with _Server((host, port), QCHandler) as httpd:
        httpd.batch = {"base_subjects": subjects, "base_cfg": cfg,
                       "dataset": dataset, "overrides": {}, "_cache": None}
        _run(httpd, f"http://{host}:{port}/",
             f"osipy-qc dashboard running ({len(subjects)} subjects) at", open_browser)
