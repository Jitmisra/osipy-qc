"""Tests for the local web UI (`osipy-qc --serve`).

Answers the review question "how are they uploading the data right now? Just
writing the Python command with flags?" — a researcher should not need the CLI.

The multipart parser gets the most attention here, because the obvious
implementation is silently wrong on exactly the files this tool exists to read.
"""

import gzip
import re
import threading
import time
import urllib.error
import os
import urllib.parse
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


def test_upload_page_states_the_minimum_inputs():
    """"Which are the minimum inputs?" was a review question, and the page's answer
    was wrong: it marked the CBF map "required". Stream A grades the raw acquisition
    with no CBF map at all, so either input alone is enough and the page has to say
    which."""
    page = _upload_page()
    assert 'name="cbf"' in page and 'name="files"' in page
    assert "Minimum inputs" in page
    assert page.count("one of the two") == 2, "both alternatives must be marked as such"
    for optional in ("gm", "wm", "csf"):
        assert f'name="{optional}"' in page


def _css_rules(page: str) -> list[tuple[str, str]]:
    """(selector, declarations) for every flat rule in the page's <style>."""
    css = page.split("<style>", 1)[1].split("</style>", 1)[0]
    return re.findall(r"([^{}]+)\{([^{}]*)\}", css)


def test_only_the_filename_hint_is_clipped_to_an_ellipsis():
    """A reviewer's screenshot showed the raw zone reading "Each is recog…".

    `.drop .txt small` set white-space:nowrap + text-overflow:ellipsis so that the
    filename the JS swaps into the hint line cannot wrap. The zone's description
    and the "same voxel grid" note sit in the same slot, so they were clipped to
    one line too. The rule is scoped to the hint line rather than deleted - the
    filename still must not wrap.
    """
    clipping = [sel.strip() for sel, decl in _css_rules(_upload_page())
                if "white-space:nowrap" in decl.replace(" ", "").replace("\n", "")
                and ".drop" in sel and "small" in sel]
    assert clipping, "the filename hint lost its no-wrap protection"
    for sel in clipping:
        assert "[data-hint]" in sel, f"{sel} clips descriptive text as well"


def test_an_unrecognised_filename_opens_the_per_role_boxes():
    """role() tells the reader "use the boxes below". Those boxes are inside
    <details id="byrole">, and "byrole" appeared nowhere in the script - so the
    message pointed at something the reader could not see."""
    script = _upload_page().split("<script>", 1)[1]
    assert "byrole.open = true" in script
    assert "classList.add('needed')" in script


def test_the_page_states_which_filenames_are_recognised():
    """"Indicate format of file name" - the zone said files are "recognised by
    their filename" and then named not one token."""
    page = _upload_page()
    assert "Which filenames are recognised" in page
    for token in ("pcasl", "m0", "calib", "mprage"):
        assert token in page
    # BIDS is recommended as canonical. That is only honest because BIDS names pass
    # the classifier unchanged, which tests/test_filename_vocabulary.py asserts.
    assert "sub-01_m0scan.nii.gz" in page


def test_the_per_role_boxes_are_at_least_as_prominent_as_the_token_list():
    """The vocabulary is a convenience and is necessarily incomplete, so the boxes
    that need no filename at all must not read as the lesser option: they come
    first, and they are styled up rather than down."""
    page = _upload_page()
    assert page.index('id="byrole"') < page.index('id="names"')
    assert 'class="manual strong" id="byrole"' in page
    assert ".manual.strong summary" in page


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
            headers={"Content-Type": "multipart/form-data; boundary=B",
                     "Accept": "application/json"})
        try:
            urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            assert e.code == 400
            import json as _json
            assert "error" in _json.loads(e.read())
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
            headers={"Content-Type": "multipart/form-data; boundary=B",
                     "Accept": "application/json"})
        import json as _json
        res = urllib.request.urlopen(req)
        assert res.headers.get("Content-Type", "").startswith("application/json")
        data = _json.loads(res.read())
        # the same payload a cohort scan returns, so React renders it with the
        # same components instead of receiving a foreign HTML document
        assert data["verdict"] in ("PASS", "WARN", "FAIL", "UNKNOWN")
        assert any(c["id"] == "1.qei" for c in data["checks"])   # the QEI ran
        assert data["kpis"], "headline metrics are present"
        assert data["figures"], "figures survive an upload"
        # and a figure actually renders from the retained arrays
        fig = data["figures"][0]
        # addressed by the upload's token, never by its filename
        url = base + f"/api/subject/{urllib.parse.quote(data['token'])}/figure/{fig['name']}.png"
        img = urllib.request.urlopen(url)
        assert img.status == 200 and int(img.headers["Content-Length"]) > 500
    finally:
        srv.shutdown()


def test_upload_limit_is_declared():
    assert MAX_UPLOAD_BYTES > 0


def test_run_serves_html_when_json_was_not_requested():
    """The no-build console cannot render a payload, so it gets the report.

    /run answers whichever the caller asked for: React sends
    Accept: application/json and renders the payload itself; anything else —
    the console's own form, or a plain POST with JavaScript off — gets the
    self-contained HTML report it can actually display.
    """
    import nibabel as nib

    from osipy_qc.synth import synthetic_case

    c = synthetic_case(quality="clean", seed=0)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    import tempfile
    tmp = tempfile.mkdtemp()

    def blob(arr):
        p = os.path.join(tmp, "x.nii.gz")
        nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), p)
        with open(p, "rb") as fh:
            return fh.read()

    body = (_part("cbf", "cbf.nii.gz", blob(c.cbf))
            + _part("gm", "gm.nii.gz", blob(c.gm))
            + b"--B--\r\n")
    srv, base = _serve()
    try:
        req = urllib.request.Request(
            base + "/run", data=body,
            headers={"Content-Type": "multipart/form-data; boundary=B"})
        res = urllib.request.urlopen(req)
        assert res.headers.get("Content-Type", "").startswith("text/html")
        html = res.read().decode()
        assert html.startswith("<!DOCTYPE html>")
        assert "1.qei" in html
    finally:
        srv.shutdown()


def test_an_upload_is_not_readable_by_its_filename():
    """One visitor must not be able to read another's scan.

    Uploads used to be cached under the client's own filename, so anyone who
    guessed it — and "perfusion_calib.nii.gz" is what half of oxford_asl's
    output is called — could fetch a stranger's report and their brain images
    from a public instance. They are keyed by an unguessable token now.
    """
    import nibabel as nib

    from osipy_qc.synth import synthetic_case

    c = synthetic_case(quality="clean", seed=0)
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    import tempfile
    tmp = tempfile.mkdtemp()

    def blob(arr):
        p = os.path.join(tmp, "x.nii.gz")
        nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), p)
        with open(p, "rb") as fh:
            return fh.read()

    body = (_part("cbf", "perfusion_calib.nii.gz", blob(c.cbf))
            + _part("gm", "gm.nii.gz", blob(c.gm))
            + b"--B--\r\n")
    srv, base = _serve()
    try:
        req = urllib.request.Request(
            base + "/run", data=body,
            headers={"Content-Type": "multipart/form-data; boundary=B",
                     "Accept": "application/json"})
        import json as _json
        data = _json.loads(urllib.request.urlopen(req).read())
        assert data["token"] and data["token"] != data["sid"]

        # the token works
        ok = urllib.request.urlopen(
            base + f"/api/subject/{urllib.parse.quote(data['token'])}")
        assert ok.status == 200

        # the filename does not
        try:
            urllib.request.urlopen(base + "/api/subject/perfusion_calib.nii.gz")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        else:
            raise AssertionError("an upload was readable by its filename")
    finally:
        srv.shutdown()
