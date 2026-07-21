"""Tests for the cohort dashboard: the batch engine and the dashboard pages.

The dashboard is the proposal's UI (batch overview + participant ledger +
per-subject deep dive), built local and dependency-free.
"""

import re
import threading
import time
import urllib.error
import urllib.request

import nibabel as nib
import numpy as np

from osipy_qc.batch import (Subject, check_label, demo_cohort, grade_folder,
                            stream_b_checks, summarise)
from osipy_qc.core.config import QCConfig
from osipy_qc.dashboard_html import render_overview, render_subject
from osipy_qc.web import QCHandler, _Server, serve_dashboard  # noqa: F401


# --------------------------------------------------------------------------- #
# batch engine
# --------------------------------------------------------------------------- #
def test_demo_cohort_has_a_realistic_spread():
    subs = demo_cohort(14)
    assert len(subs) == 14
    s = summarise(subs)
    assert s.total == 14
    # the pattern is designed to yield both passes and fails
    assert s.counts.get("PASS", 0) > 0
    assert s.counts.get("FAIL", 0) > 0
    assert abs(sum(s.rates.values()) - 1.0) < 1e-6 or s.rates["PASS"] + s.rates["WARN"] + s.rates["FAIL"] <= 1.0


def test_demo_cohort_is_deterministic():
    a = [(x.sid, x.overall) for x in demo_cohort(10)]
    b = [(x.sid, x.overall) for x in demo_cohort(10)]
    assert a == b


def test_subject_reports_qei_and_primary_artifact():
    subs = demo_cohort(14)
    a_fail = next(s for s in subs if s.overall == "FAIL")
    assert isinstance(a_fail.qei, float)
    assert a_fail.primary_artifact != "-"          # a FAIL must name its artifact
    a_pass = next(s for s in subs if s.overall == "PASS")
    assert a_pass.primary_artifact == "-"


def test_summarise_breakdown_is_sorted_worst_first():
    s = summarise(demo_cohort(14))
    counts = [n for _, n in s.artifact_breakdown]
    assert counts == sorted(counts, reverse=True)
    assert all(n <= s.total for _, n in s.artifact_breakdown)


def test_stream_b_checks_excludes_raw_data_checks():
    b = stream_b_checks()
    assert "1.qei" in b and "3.1.cbf_level" in b
    assert "7.1.motion" not in b and "5.1.schema" not in b


def test_check_label_is_human():
    assert check_label("2.1.spatial_cov") == "sCoV"
    assert check_label("7.1.motion") == "Motion"


def test_grade_folder_reads_subject_subdirs(tmp_path):
    from osipy_qc.synth import synthetic_case

    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    for i, quality in enumerate(("clean", "garbage")):
        d = tmp_path / f"sub-{i:02d}"
        d.mkdir()
        c = synthetic_case(quality=quality, seed=i)
        nib.save(nib.Nifti1Image(c.cbf.astype(np.float32), aff), d / "perfusion_calib.nii.gz")
        nib.save(nib.Nifti1Image(c.gm.astype(np.float32), aff), d / "pvgm_inasl.nii.gz")
        nib.save(nib.Nifti1Image(c.wm.astype(np.float32), aff), d / "pvwm_inasl.nii.gz")

    subs = grade_folder(str(tmp_path))
    assert [s.sid for s in subs] == ["sub-00", "sub-01"]
    assert subs[0].overall == "PASS"               # clean
    assert subs[1].overall == "FAIL"               # garbage


def test_grade_folder_skips_dirs_without_a_cbf_map(tmp_path):
    (tmp_path / "not_a_subject").mkdir()
    (tmp_path / "not_a_subject" / "readme.txt").write_text("nothing here")
    assert grade_folder(str(tmp_path)) == []


# --------------------------------------------------------------------------- #
# dashboard pages
# --------------------------------------------------------------------------- #
def _cohort():
    cfg = QCConfig()
    subs = demo_cohort(14)
    return subs, summarise(subs), cfg


def test_overview_is_self_contained_and_lists_every_subject():
    subs, summ, cfg = _cohort()
    h = render_overview(subs, summ, cfg, "demo")
    assert h.startswith("<!DOCTYPE html>")
    assert re.findall(r'(?:src|href)="(?!data:|#|/)', h) == []   # only data:/#/local links
    for s in subs:
        assert f"/subject/{s.sid}" in h            # ledger + sidebar link to each
    assert "Batch overview" in h
    assert "Artifact breakdown" in h


def test_overview_shows_the_rates():
    subs, summ, cfg = _cohort()
    h = render_overview(subs, summ, cfg, "demo")
    assert f"{summ.pass_rate*100:.0f}" in h
    assert "Pass rate" in h and "Fail rate" in h


def test_subject_page_embeds_the_full_report():
    subs, _summ, cfg = _cohort()
    fail = next(s for s in subs if s.overall == "FAIL")
    h = render_subject(subs, fail, cfg)
    assert h.startswith("<!DOCTYPE html>")
    assert "OVERALL VERDICT" in h                  # the report hero
    assert "1.qei" in h                            # the checks
    assert "data:image/png;base64," in h           # the brain images
    assert fail.sid in h


def test_subject_page_escapes_and_stays_local():
    subs, _summ, cfg = _cohort()
    h = render_subject(subs, subs[0], cfg)
    assert re.findall(r'(?:src|href)="(?!data:|#|/)', h) == []


# --------------------------------------------------------------------------- #
# the dashboard server, over real HTTP
# --------------------------------------------------------------------------- #
def _serve_dashboard():
    from osipy_qc.batch import summarise as _sum

    subs = demo_cohort(14)
    srv = _Server(("127.0.0.1", 0), QCHandler)
    srv.batch = {"subjects": subs, "summary": _sum(subs),
                 "cfg": QCConfig(), "dataset": "demo"}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.2)
    return srv, f"http://127.0.0.1:{srv.server_address[1]}", subs


def test_server_serves_overview_at_root():
    srv, base, _ = _serve_dashboard()
    try:
        body = urllib.request.urlopen(base + "/").read().decode()
        assert "Batch overview" in body
    finally:
        srv.shutdown()


def test_server_serves_each_subject():
    srv, base, subs = _serve_dashboard()
    try:
        body = urllib.request.urlopen(base + f"/subject/{subs[0].sid}").read().decode()
        assert subs[0].sid in body and "OVERALL VERDICT" in body
    finally:
        srv.shutdown()


def test_server_404s_unknown_subject():
    srv, base, _ = _serve_dashboard()
    try:
        urllib.request.urlopen(base + "/subject/sub-999")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    else:
        raise AssertionError("expected 404 for an unknown subject")
    finally:
        srv.shutdown()
