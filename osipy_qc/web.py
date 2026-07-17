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

from .core.config import POPULATIONS, for_population
from .report import run_qc
from .report_html import render_html

# Refuse absurd uploads outright rather than trying to parse them.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024      # 512 MB across all files in one request

_PAGE_CSS = """
:root{--paper:#fff;--warm:#FDF9F5;--ink:#241C15;--soft:#6B5D4F;--line:#EDE2D5;--accent:#C2571A;}
*{box-sizing:border-box}
body{margin:0;background:var(--warm);color:var(--ink);
     font-family:ui-serif,"Iowan Old Style",Georgia,serif;line-height:1.55}
.wrap{max-width:640px;margin:0 auto;padding:3rem 1.25rem}
h1{font-size:1.8rem;margin:0 0 .3rem}
.sub{color:var(--soft);margin:0 0 2rem}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
form{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:1.5rem}
label{display:block;font-size:.82rem;font-family:ui-monospace,Menlo,monospace;
      letter-spacing:.04em;text-transform:uppercase;color:var(--soft);margin:1.1rem 0 .3rem}
label:first-of-type{margin-top:0}
input[type=file],select{width:100%;padding:.6rem;border:1px solid var(--line);
      border-radius:4px;background:var(--warm);font-family:inherit;font-size:.95rem}
.req{color:var(--accent)}
.hint{font-size:.78rem;color:var(--soft);margin:.25rem 0 0}
button{margin-top:1.6rem;width:100%;padding:.8rem;border:0;border-radius:4px;
       background:var(--accent);color:#fff;font-size:1rem;font-weight:600;cursor:pointer;
       font-family:inherit}
button:hover{filter:brightness(1.08)}
button:disabled{opacity:.6;cursor:progress}
.note{background:#FBE4D0;border-left:3px solid var(--accent);padding:.8rem 1rem;
      border-radius:0 4px 4px 0;font-size:.86rem;margin:1.5rem 0 0}
.err{background:#fdecea;border-left:3px solid #C0392B;padding:.8rem 1rem;
     border-radius:0 4px 4px 0;font-size:.9rem;margin-bottom:1.5rem}
a{color:var(--accent)}
"""


def _upload_page(error: str = "") -> str:
    opts = "".join(
        f'<option value="{p}"{" selected" if p == "adult" else ""}>{p}</option>'
        for p in sorted(POPULATIONS)
    )
    err = f'<div class="err"><b>Could not grade that.</b><br>{error}</div>' if error else ""
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>osipy-qc</title><style>{_PAGE_CSS}</style></head><body><div class="wrap">
<h1>ASL Quality Control</h1>
<p class="sub">Upload a CBF map and get a PASS / WARN / FAIL report, with reasons.</p>
{err}
<form method="post" action="/run" enctype="multipart/form-data"
      onsubmit="this.querySelector('button').disabled=true;
                this.querySelector('button').textContent='Grading\\u2026';">

  <label for="cbf">CBF map <span class="req">*required</span></label>
  <input id="cbf" type="file" name="cbf" accept=".nii,.gz" required>
  <p class="hint">The quantified CBF map (mL/100g/min), NIfTI.</p>

  <label for="gm">Grey-matter map</label>
  <input id="gm" type="file" name="gm" accept=".nii,.gz">
  <label for="wm">White-matter map</label>
  <input id="wm" type="file" name="wm" accept=".nii,.gz">
  <label for="csf">CSF map</label>
  <input id="csf" type="file" name="csf" accept=".nii,.gz">
  <p class="hint">Tissue probability maps, <b>on the same voxel grid as the CBF map</b>.
     Without GM/WM the QEI and the level checks cannot run and will report UNKNOWN.
     CSF is derived as 1&minus;GM&minus;WM if you leave it out.</p>

  <label for="population">Population</label>
  <select id="population" name="population">{opts}</select>
  <p class="hint">CBF norms move a lot across the lifespan &mdash; a child's normal
     GM CBF (~97) would look abnormal against adult bands, and a neonate's (~16)
     far more so. Pick the right one.</p>

  <button type="submit">Grade this scan</button>
</form>
<div class="note"><b>Running locally.</b> Files are graded in a temporary folder on
this machine and deleted immediately afterwards &mdash; nothing is uploaded anywhere
or kept. This server is meant for local use and is not hardened for public hosting.</div>
</div></body></html>"""


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
        if self.path in ("/", "/index.html"):
            self._send(_upload_page())
        else:
            self._send("<h1>404</h1><p><a href='/'>Start over</a></p>", 404)

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


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Run the local QC web app. Binds to localhost by default (see module docstring)."""
    with _Server((host, port), QCHandler) as httpd:
        url = f"http://{host}:{port}/"
        print(f"osipy-qc web UI running at {url}")
        print("  upload a CBF map (+ tissue maps) and get a report. Ctrl-C to stop.")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped.")
