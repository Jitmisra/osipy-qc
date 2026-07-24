"""Tests for the JSON API that the web frontend consumes.

The browser client renders whatever these payloads contain and recomputes
nothing, so these tests pin the contract: the shapes the UI depends on, and the
guarantee that a payload is always JSON-serialisable (NumPy scalars and
non-finite values are a real hazard when the source is a NumPy pipeline).
"""

import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from osipy_qc import api
from osipy_qc.batch import demo_cohort, summarise
from osipy_qc.core.config import QCConfig
from osipy_qc.core.result import CheckResult, Verdict
from osipy_qc.web import QCHandler, _Server


def _cohort(n=6):
    subs = demo_cohort(n)
    return subs, summarise(subs), QCConfig()


# --------------------------------------------------------------------------- #
# payload shapes
# --------------------------------------------------------------------------- #
def test_cohort_payload_has_everything_the_overview_renders():
    subs, summ, cfg = _cohort()
    p = api.cohort_payload(subs, summ, cfg, "demo")
    for key in ("total", "counts", "rates", "medianQei", "qeiCutoff",
                "flagBreakdown", "subjects", "matrix", "population"):
        assert key in p, f"missing {key}"
    assert len(p["subjects"]) == summ.total
    # the ledger is sorted worst-first, which is the triage order
    sev = [r["severity"] for r in p["subjects"]]
    assert sev == sorted(sev)


def test_cohort_matrix_is_rectangular_and_covers_every_subject():
    subs, summ, cfg = _cohort()
    p = api.cohort_payload(subs, summ, cfg, "demo")
    sids = [s["sid"] for s in p["subjects"]]
    assert p["matrix"], "expected at least one check row"
    for row in p["matrix"]:
        assert [c["sid"] for c in row["cells"]] == sids


def test_subject_payload_shape_and_ordering():
    subs, _summ, _cfg = _cohort()
    p = api.subject_payload(subs[0])
    for key in ("sid", "verdict", "summary", "checks", "kpis", "figures", "drivers"):
        assert key in p
    # Stream B (the CBF map) first, then worst-first inside each stream
    streams = [c["stream"] for c in p["checks"]]
    assert streams == sorted(streams, key=lambda s: s != "B")
    b = [c["severity"] for c in p["checks"] if c["stream"] == "B"]
    assert b == sorted(b)


def test_every_payload_is_json_serialisable():
    """NumPy floats and NaN would both break json.dumps; the adapter must clean them."""
    subs, summ, cfg = _cohort()
    for payload in (
        api.cohort_payload(subs, summ, cfg, "demo"),
        api.subject_payload(subs[0]),
        api.config_payload(cfg),
        api.provenance_payload(),
        api.checks_catalogue(),
    ):
        json.dumps(payload)          # must not raise


def test_non_finite_metrics_become_null_not_nan():
    """NaN/Infinity are not valid JSON; they have to arrive as null."""
    res = CheckResult("x.test", Verdict.PASS,
                      metric={"bad": float("nan"), "worse": np.float64("inf"),
                              "fine": np.float32(1.5), "count": np.int64(3)})
    payload = api.check_payload(res.to_dict(), {"x.test": "B"})
    assert payload["metric"]["bad"] is None
    assert payload["metric"]["worse"] is None
    assert payload["metric"]["fine"] == 1.5
    assert payload["metric"]["count"] == 3
    assert "NaN" not in json.dumps(payload)


def test_provisional_only_marks_non_passing_verdicts():
    """A provisional flag on a PASS would be meaningless noise in the UI."""
    passing = CheckResult("x.a", Verdict.PASS, provisional=True).to_dict()
    failing = CheckResult("x.b", Verdict.FAIL, provisional=True).to_dict()
    assert api.check_payload(passing, {})["provisional"] is False
    assert api.check_payload(failing, {})["provisional"] is True


def test_config_payload_carries_provenance_per_field():
    p = api.config_payload(QCConfig())
    fields = [f for g in p["groups"] for f in g["fields"]]
    assert fields, "expected tunable fields"
    for f in fields:
        assert f["provenance"] in ("published", "implementation", "uncalibrated")
    # the published QEI cut-off must be labelled as such
    qei = next(f for f in fields if f["name"] == "qei_warn")
    assert qei["provenance"] == "published" and "doi" in qei["citation"].lower()


def test_config_payload_exposes_each_population_preset():
    """The client resets the bands when a population is picked, so it needs the
    full value set for every population up front."""
    p = api.config_payload(QCConfig())
    for pop in p["populations"]:
        assert pop in p["populationValues"]
        assert "gm_cbf_lo" in p["populationValues"][pop]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def test_figures_render_as_real_png_and_svg_bytes():
    subs, _summ, _cfg = _cohort()
    png, ctype = api.figure_bytes(subs[0], "cbf")
    assert ctype == "image/png" and png[:8] == b"\x89PNG\r\n\x1a\n"
    svg, ctype = api.figure_bytes(subs[0], "gm_hist")
    assert ctype == "image/svg+xml" and svg.lstrip().startswith(b"<svg")


def test_unknown_figure_raises_keyerror_so_the_server_can_404():
    subs, _summ, _cfg = _cohort()
    try:
        api.figure_bytes(subs[0], "not-a-figure")
    except KeyError:
        return
    raise AssertionError("expected KeyError for an unknown figure")


# --------------------------------------------------------------------------- #
# over real HTTP
# --------------------------------------------------------------------------- #
def _serve():
    subs = demo_cohort(5)
    srv = _Server(("127.0.0.1", 0), QCHandler)
    srv.batch = {"base_subjects": subs, "base_cfg": QCConfig(),
                 "dataset": "demo", "overrides": {}, "_cache": None}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.2)
    return srv, f"http://127.0.0.1:{srv.server_address[1]}", subs


def _json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode())


def test_api_routes_serve_json():
    srv, base, subs = _serve()
    try:
        assert _json(base + "/api/health")["ok"] is True
        assert _json(base + "/api/cohort")["total"] == 5
        assert _json(base + f"/api/subject/{subs[0].sid}")["sid"] == subs[0].sid
        assert _json(base + "/api/config")["groups"]
        assert _json(base + "/api/provenance")["total"] > 0
        assert len(_json(base + "/api/checks")) > 0
    finally:
        srv.shutdown()


def test_api_404s_an_unknown_subject_as_json():
    srv, base, _ = _serve()
    try:
        urllib.request.urlopen(base + "/api/subject/nope")
    except urllib.error.HTTPError as e:
        assert e.code == 404
        assert "error" in json.loads(e.read().decode())
    else:
        raise AssertionError("expected a 404")
    finally:
        srv.shutdown()


def test_posting_config_regrades_the_cohort():
    """The threshold panel posts overrides; the server must re-grade and return
    the fresh cohort rather than the stale one."""
    srv, base, _ = _serve()
    try:
        before = _json(base + "/api/cohort")
        req = urllib.request.Request(
            base + "/api/config",
            data=json.dumps({"qei_warn": 0.999, "qei_pass": 0.999}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            after = json.loads(r.read().decode())
        assert after["customThresholds"] is True
        assert after["counts"].get("PASS", 0) < before["counts"].get("PASS", 1)
    finally:
        srv.shutdown()


def test_posting_config_ignores_unknown_keys():
    """A hand-crafted request must not be able to set arbitrary attributes."""
    srv, base, _ = _serve()
    try:
        req = urllib.request.Request(
            base + "/api/config",
            data=json.dumps({"not_a_real_field": 1, "qei_warn": 0.4}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            body = json.loads(r.read().decode())
        assert body["qeiCutoff"] == 0.4          # the known field applied
    finally:
        srv.shutdown()


# --------------------------------------------------------------------------- #
# serving the built single-page app
# --------------------------------------------------------------------------- #
def _dist_built():
    import os
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "dist")
    return os.path.isfile(os.path.join(root, "index.html"))


def test_built_app_is_served_and_deep_links_survive_a_refresh():
    """A client-side route must return the app shell, or a hard refresh on
    /subject/<id> would 404. Skipped when the frontend has not been built."""
    if not _dist_built():
        return
    srv, base, subs = _serve()
    try:
        root = urllib.request.urlopen(base + "/")
        assert root.status == 200
        html = root.read().decode()
        assert "<div id=\"root\">" in html
        deep = urllib.request.urlopen(base + f"/subject/{subs[0].sid}")
        assert deep.status == 200 and "<div id=\"root\">" in deep.read().decode()
    finally:
        srv.shutdown()


def test_a_missing_asset_404s_rather_than_returning_the_app_shell():
    """Serving index.html for a missing .js would turn a broken build into a
    confusing runtime error instead of an obvious 404."""
    if not _dist_built():
        return
    srv, base, _ = _serve()
    try:
        urllib.request.urlopen(base + "/assets/does-not-exist.js")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    else:
        raise AssertionError("expected 404 for a missing asset")
    finally:
        srv.shutdown()


def test_the_server_rendered_pages_remain_available_as_a_fallback():
    """The no-build path still works, so the package is usable straight from a
    wheel without running a bundler."""
    srv, base, subs = _serve()
    try:
        body = urllib.request.urlopen(base + "/legacy/").read().decode()
        assert "Batch overview" in body
    finally:
        srv.shutdown()
