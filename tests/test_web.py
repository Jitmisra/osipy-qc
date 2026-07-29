"""Tests for the local web UI (`osipy-qc --serve`).

Answers the review question "how are they uploading the data right now? Just
writing the Python command with flags?" — a researcher should not need the CLI.

The multipart parser gets the most attention here, because the obvious
implementation is silently wrong on exactly the files this tool exists to read.
"""

import gzip
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from osipy_qc.web import (MAX_UPLOAD_BYTES, QCHandler, _parse_multipart,
                          _upload_page, _Server)


def _part(name, filename, data):
    h = f'--B\r\nContent-Disposition: form-data; name="{name}"'
    if filename:
        h += f'; filename="{filename}"'
    return h.encode() + b"\r\n\r\n" + data + b"\r\n"


# --------------------------------------------------------------------------- #
# multipart parsing
# --------------------------------------------------------------------------- #
def test_parses_a_simple_field():
    body = _part("population", None, b"neonate") + b"--B--\r\n"
    got = _parse_multipart(body, "multipart/form-data; boundary=B")
    assert got["population"] == ("", b"neonate")


def test_binary_payload_survives_byte_for_byte():
    """THE regression test.

    The tempting implementation — hand the body to the stdlib `email` module —
    corrupts binary uploads, because it decodes the payload to str internally. A
    gzipped NIfTI then fails to inflate ('zlib.error: invalid literal/lengths
    set'). This asserts the bytes come back exactly as they went in, gzip magic
    and all.
    """
    blob = gzip.compress(np.arange(4096, dtype=np.uint8).tobytes())
    assert blob[:2] == b"\x1f\x8b"                     # real gzip
    body = _part("cbf", "x.nii.gz", blob) + b"--B--\r\n"

    filename, data = _parse_multipart(body, "multipart/form-data; boundary=B")["cbf"]
    assert filename == "x.nii.gz"
    assert data == blob                                 # byte-for-byte
    assert gzip.decompress(data)                        # and it still inflates


def test_payload_containing_crlf_is_not_truncated():
    """Binary data can contain \\r\\n; only the boundary's own CRLF may be stripped."""
    blob = b"\x00\r\n\x01\r\n\r\n\x02"
    body = _part("cbf", "x.nii", blob) + b"--B--\r\n"
    assert _parse_multipart(body, "multipart/form-data; boundary=B")["cbf"][1] == blob


def test_quoted_boundary_is_accepted():
    body = _part("population", None, b"adult") + b"--B--\r\n"
    got = _parse_multipart(body, 'multipart/form-data; boundary="B"')
    assert got["population"][1] == b"adult"


def test_multiple_fields_round_trip():
    body = (_part("cbf", "a.nii.gz", b"\x1f\x8bAAA")
            + _part("gm", "b.nii.gz", b"\x1f\x8bBBB")
            + _part("population", None, b"neonate")
            + b"--B--\r\n")
    got = _parse_multipart(body, "multipart/form-data; boundary=B")
    assert set(got) == {"cbf", "gm", "population"}
    assert got["gm"] == ("b.nii.gz", b"\x1f\x8bBBB")


def test_missing_boundary_raises():
    try:
        _parse_multipart(b"junk", "multipart/form-data")
    except ValueError:
        return
    raise AssertionError("expected ValueError when the boundary is missing")


# --------------------------------------------------------------------------- #
# the upload page
# --------------------------------------------------------------------------- #
def test_upload_page_offers_the_v1_populations():
    page = _upload_page()
    for p in ("adult", "neonate"):
        assert f'value="{p}"' in page
    # the un-shipped age groups must not appear
    for gone in ("child", "elderly", "neonate_term"):
        assert f'value="{gone}"' not in page


def test_upload_page_requires_the_cbf_map_only():
    page = _upload_page()
    assert 'name="cbf"' in page and "required" in page
    for optional in ("gm", "wm", "csf"):
        assert f'name="{optional}"' in page


def test_upload_page_renders_an_error_when_given_one():
    page = _upload_page(error="something broke")
    assert "Could not grade" in page and "something broke" in page


def test_upload_page_is_self_contained():
    page = _upload_page()
    assert "<form" in page and "http://" not in page.split("</head>")[0]


# --------------------------------------------------------------------------- #
# the server, driven over real HTTP
# --------------------------------------------------------------------------- #
def _serve():
    srv = _Server(("127.0.0.1", 0), QCHandler)          # port 0 -> free port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.2)
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_get_serves_the_upload_page():
    """/upload is answered.

    When the React bundle is built it owns the route; the server-rendered
    console below is the no-build fallback, so both are acceptable here and the
    fallback itself is asserted directly in the next test."""
    srv, base = _serve()
    try:
        body = urllib.request.urlopen(base + "/upload").read().decode()
        assert '<div id="root">' in body or "<form" in body
    finally:
        srv.shutdown()


def test_the_no_build_upload_console_still_renders():
    """The package must be usable straight from a wheel, with no bundler run."""
    from osipy_qc.web import _upload_page
    body = _upload_page()
    assert "<form" in body and 'action="/run"' in body and 'method="post"' in body
    assert 'name="cbf"' in body


def test_unknown_path_is_404():
    srv, base = _serve()
    try:
        # a path that looks like a file but does not exist is a missing asset
        urllib.request.urlopen(base + "/nope.js")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    else:
        raise AssertionError("expected 404")
    finally:
        srv.shutdown()


def test_upload_with_no_file_shows_an_error_and_keeps_serving():
    """A bad upload must never take the server down."""
    srv, base = _serve()
    try:
        req = urllib.request.Request(
            base + "/run", data=b"--B--\r\n",
            headers={"Content-Type": "multipart/form-data; boundary=B"})
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            assert e.code == 400
            assert b"Could not grade" in e.read()
        else:
            raise AssertionError("expected 400")
        # still alive
        assert urllib.request.urlopen(base + "/").status == 200
    finally:
        srv.shutdown()


def test_end_to_end_upload_produces_a_report(tmp_path):
    """Synthetic CBF + tissue maps over real HTTP -> a real report."""
    import nibabel as nib

    from osipy_qc.synth import synthetic_case

    c = synthetic_case(quality="clean", seed=0)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])

    def blob(arr):
        p = tmp_path / "x.nii.gz"
        nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), p)
        return p.read_bytes()

    body = (_part("cbf", "cbf.nii.gz", blob(c.cbf))
            + _part("gm", "gm.nii.gz", blob(c.gm))
            + _part("wm", "wm.nii.gz", blob(c.wm))
            + _part("population", None, b"adult")
            + b"--B--\r\n")

    srv, base = _serve()
    try:
        req = urllib.request.Request(
            base + "/run", data=body,
            headers={"Content-Type": "multipart/form-data; boundary=B"})
        html = urllib.request.urlopen(req).read().decode()
        assert html.startswith("<!DOCTYPE html>")
        assert "1.qei" in html                       # the QEI actually ran
        assert "data:image/png;base64," in html      # images rendered
        assert "<svg" in html                        # the histogram
    finally:
        srv.shutdown()


def test_upload_limit_is_declared():
    assert MAX_UPLOAD_BYTES > 0
