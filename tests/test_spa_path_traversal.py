"""The static file server must not serve anything outside the build directory.

`_serve_spa` guarded itself with `target.startswith(root)`, which is a comparison
of CHARACTERS where a comparison of PATH SEGMENTS was meant. With a build root of
`web/dist`, every sibling directory whose name merely begins with `dist` passed
the guard - so `GET /../dist-secrets/creds.js` normalised to `web/dist-secrets/
creds.js`, satisfied `startswith`, and was served.

This was verified reachable over a raw socket before the fix, not argued from
reading the code: a browser rewrites `..` before sending, but nothing obliges any
other client to. The service is deployed publicly on 0.0.0.0, so the guard is
load-bearing.

These tests drive the real handler over a real socket for the same reason.
"""

from __future__ import annotations

import os
import socket
import threading
from http.server import HTTPServer

import pytest

from osipy_qc import web as W


@pytest.fixture
def spa(tmp_path, monkeypatch):
    """A built SPA, a same-prefixed sibling holding a secret, and a symlink out."""
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<h1>app</h1>")
    (root / "assets" / "app.js").write_text("console.log(1)")

    secrets = tmp_path / "dist-secrets"          # note: shares the "dist" prefix
    secrets.mkdir()
    (secrets / "creds.js").write_text("SECRET_TOKEN=hunter2")
    os.symlink(secrets / "creds.js", root / "escape.js")

    monkeypatch.setattr(W, "_spa_root", lambda: str(root))
    srv = HTTPServer(("127.0.0.1", 0), W.QCHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def _get(port: int, target: str) -> str:
    """Send `target` as the raw request line, without client-side normalisation."""
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(f"GET {target} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
              f"Connection: close\r\n\r\n".encode())
    chunks = []
    while True:
        d = s.recv(65536)
        if not d:
            break
        chunks.append(d)
    s.close()
    return b"".join(chunks).decode("utf8", "replace")


@pytest.mark.parametrize("target", [
    "/../dist-secrets/creds.js",              # the sibling-prefix bypass itself
    "/./../dist-secrets/creds.js",
    "/assets/../../dist-secrets/creds.js",
    "/escape.js",                             # a symlink pointing out of the root
    "/../../etc/passwd",
])
def test_nothing_outside_the_build_directory_is_ever_served(spa, target):
    body = _get(spa, target)
    assert "hunter2" not in body, f"{target} leaked a file outside the build root"
    assert "root:x:" not in body, f"{target} leaked a system file"


@pytest.mark.parametrize("target,needle", [
    ("/index.html", "app"),
    ("/assets/app.js", "console.log"),
    ("/some/client/route", "app"),   # deep links fall through to index.html
    ("/", "app"),
])
def test_the_fix_does_not_break_ordinary_requests(spa, target, needle):
    """A guard that also blocks the app is not a fix."""
    response = _get(spa, target)
    assert " 200 " in response.splitlines()[0], f"{target} should be served"
    assert needle in response


def test_within_compares_segments_not_characters():
    """The unit behind the fix, stated directly."""
    assert W._within("/srv/web/dist/index.html", "/srv/web/dist")
    assert W._within("/srv/web/dist", "/srv/web/dist")
    assert not W._within("/srv/web/dist-secrets/creds.js", "/srv/web/dist")
    assert not W._within("/srv/web/distant", "/srv/web/dist")
    assert not W._within("/etc/passwd", "/srv/web/dist")
