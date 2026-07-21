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
same `report_html.render_html` the CLI writes.

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
import os
import re
import socketserver
import tempfile
import traceback
import webbrowser

from ._webassets import BASE_CSS, LOGO_SVG, esc
from .core.config import POPULATIONS, for_population
from .report import run_qc
from .report_html import render_html

# Refuse absurd uploads outright rather than trying to parse them.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024      # 512 MB across all files in one request

# youngest -> oldest, for the segmented population control, with short display
# labels so nothing clips in the chips (the group heading already says "Population")
_POP_ORDER = ["neonate_preterm", "neonate_term", "infant", "child",
              "adolescent", "adult", "elderly"]
_POP_LABELS = {
    "neonate_preterm": "preterm", "neonate_term": "neonate", "infant": "infant",
    "child": "child", "adolescent": "teen", "adult": "adult", "elderly": "elderly",
}

_CONSOLE_CSS = """
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


def _dropzone(field: str, title: str, hint: str, required: bool = False) -> str:
    req = " required" if required else ""
    return (
        f'<div class="drop" data-field="{field}">'
        f'<div class="ic">{_FILE_IC}</div>'
        f'<div class="txt"><b>{esc(title)}</b><small data-hint>{esc(hint)}</small></div>'
        f'<button type="button" class="clear" aria-label="remove">clear</button>'
        f'<input type="file" name="{field}" accept=".nii,.gz"{req}></div>'
    )


def _upload_page(error: str = "") -> str:
    pops = [p for p in _POP_ORDER if p in POPULATIONS]
    seg = "".join(
        f'<label><input type="radio" name="population" value="{p}"'
        f'{" checked" if p == "adult" else ""}>'
        f'<span title="{p}">{esc(_POP_LABELS.get(p, p))}</span></label>'
        for p in pops
    )
    err = f'<div class="err"><b>Could not grade that.</b> {esc(error)}</div>' if error else ""
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>osipy-qc &mdash; ASL quality control</title>
<style>{BASE_CSS}{_CONSOLE_CSS}</style></head><body>
<div class="topbar"><div class="brand">{LOGO_SVG}<b>osipy-qc</b>
  <span>ASL quality control</span></div></div>

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
    {_dropzone("cbf", "Choose or drop a CBF map", "quantified CBF (mL/100g/min), NIfTI", required=True)}

    <div class="field-label">Tissue maps <span class="opt">optional &mdash; needed for QEI &amp; level checks</span></div>
    <div class="tissue-grid">
      {_dropzone("gm", "Grey matter", "GM probability map")}
      {_dropzone("wm", "White matter", "WM probability map")}
      {_dropzone("csf", "CSF", "derived if omitted")}
      <div class="drop" style="border:none;background:transparent;box-shadow:none;cursor:default">
        <div class="txt"><small>Must share the <b>same voxel grid</b> as the CBF map.</small></div></div>
    </div>

    <div class="field-label">Population <span class="opt">CBF norms shift across the lifespan</span></div>
    <div class="seg">{seg}</div>
    <p class="hint">A child's normal GM CBF (~97) would look abnormal against adult bands;
       a neonate's (~16) far more so.</p>

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
  form.addEventListener('submit', function(ev){{
    if(!window.fetch){{ return; }}
    ev.preventDefault();
    overlay.classList.add('on');
    fetch('/run', {{method:'POST', body:new FormData(form)}})
      .then(function(r){{ return r.text(); }})
      .then(function(html){{ document.open(); document.write(html); document.close(); }})
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
            out[name] = (filename or "", data)
    return out


def _grade_upload(fields: dict[str, tuple[str, bytes]]) -> str:
    """Write the uploaded NIfTIs to a temp dir, grade them, render the report."""
    from .io import load_cbf_inputs

    cbf_name, cbf_bytes = fields.get("cbf", ("", b""))
    if not cbf_bytes:
        raise ValueError("No CBF map was uploaded.")

    population = (fields.get("population", ("", b"adult"))[1] or b"adult").decode()
    cfg = for_population(population)

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
        inputs = load_cbf_inputs(paths["cbf"], gm=paths["gm"], wm=paths["wm"],
                                 csf=paths["csf"])
        report = run_qc(inputs, cfg=cfg)
        title = f"ASL QC report — {os.path.basename(cbf_name) or 'uploaded scan'}"
        return render_html(report, inputs=inputs, cfg=cfg, title=title)


class QCHandler(http.server.BaseHTTPRequestHandler):
    server_version = "osipy-qc"

    def log_message(self, fmt, *args):          # quieter than the noisy default
        print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    def _send(self, body: str, status: int = 200) -> None:
        raw = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        batch = getattr(self.server, "batch", None)     # set in dashboard mode
        path = self.path.split("?", 1)[0]

        if batch and path in ("/", "/index.html"):
            from .dashboard_html import render_overview
            self._send(render_overview(batch["subjects"], batch["summary"],
                                       batch["cfg"], batch["dataset"]))
        elif batch and path.startswith("/subject/"):
            from .dashboard_html import render_subject
            sid = path[len("/subject/"):]
            sub = next((s for s in batch["subjects"] if s.sid == sid), None)
            if sub is None:
                self._send("<h1>404</h1><p><a href='/'>Back to overview</a></p>", 404)
            else:
                self._send(render_subject(batch["subjects"], sub, batch["cfg"]))
        elif path in ("/", "/index.html", "/upload"):
            self._send(_upload_page())
        else:
            dest = "/" if batch else "/"
            self._send(f"<h1>404</h1><p><a href='{dest}'>Start over</a></p>", 404)

    def do_POST(self):
        if self.path != "/run":
            self._send("<h1>404</h1>", 404)
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
            self._send(_grade_upload(fields))
        except Exception as exc:                # a bad upload must not kill the server
            traceback.print_exc()
            self._send(_upload_page(error=f"{type(exc).__name__}: {exc}"), 400)


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    batch = None                # set to a dict{subjects,summary,cfg,dataset} for the dashboard


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
    """Run the single-scan upload console. Binds to localhost (see module docstring)."""
    with _Server((host, port), QCHandler) as httpd:
        _run(httpd, f"http://{host}:{port}/", "osipy-qc upload console running at", open_browser)


def serve_dashboard(subjects, cfg=None, dataset: str = "cohort",
                    host: str = "127.0.0.1", port: int = 8000,
                    open_browser: bool = True) -> None:
    """Run the cohort dashboard over an already-graded list of Subjects."""
    from .batch import summarise
    from .core.config import QCConfig

    cfg = cfg or QCConfig()
    with _Server((host, port), QCHandler) as httpd:
        httpd.batch = {"subjects": subjects, "summary": summarise(subjects),
                       "cfg": cfg, "dataset": dataset}
        _run(httpd, f"http://{host}:{port}/",
             f"osipy-qc dashboard running ({len(subjects)} subjects) at", open_browser)
