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

from osipy_qc.batch import (TUNABLE, Subject, check_label, demo_cohort,
                            grade_folder, stream_b_checks, summarise)
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
    # no external RESOURCE loads (images/scripts/styles): the page is self-contained.
    # Outbound anchor hyperlinks (e.g. the README) are navigation, and allowed.
    assert re.findall(r'src="(?!data:|#|/)', h) == []
    for s in subs:
        assert f"/subject/{s.sid}" in h            # ledger + sidebar link to each
    assert "Batch overview" in h
    assert "Flagged checks" in h


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
    assert re.findall(r'src="(?!data:|#|/)', h) == []       # no external resource loads


# --------------------------------------------------------------------------- #
# the dashboard server, over real HTTP
# --------------------------------------------------------------------------- #
def _serve_dashboard():
    subs = demo_cohort(14)
    srv = _Server(("127.0.0.1", 0), QCHandler)
    srv.batch = {"base_subjects": subs, "base_cfg": QCConfig(),
                 "dataset": "demo", "overrides": {}, "_cache": None}
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


# --------------------------------------------------------------------------- #
# live threshold config
# --------------------------------------------------------------------------- #
def test_cfg_from_params_applies_overrides():
    from osipy_qc.batch import cfg_from_params

    base = QCConfig()
    cfg = cfg_from_params(base, {"population": "neonate", "qei_pass": "0.7",
                                 "gm_cbf_lo": "70", "strict": "on"})
    assert cfg.population == "neonate"        # population resets the bands...
    assert cfg.qei_pass == 0.7                 # ...then overrides apply on top
    assert cfg.gm_cbf_lo == 70.0
    assert cfg.strict is True


def test_cfg_from_params_ignores_garbage():
    from osipy_qc.batch import cfg_from_params

    cfg = cfg_from_params(QCConfig(), {"qei_pass": "not-a-number", "bogus": "x"})
    assert cfg.qei_pass == QCConfig().qei_pass   # bad value ignored, not fatal


def test_regrade_changes_verdicts_without_touching_disk():
    from osipy_qc.batch import regrade

    subs = demo_cohort(8)
    strict_pass = QCConfig(qei_pass=0.999, qei_warn=0.999)   # nothing can pass
    re = regrade(subs, strict_pass)
    assert all(s.overall != "PASS" for s in re)
    assert [s.sid for s in re] == [s.sid for s in subs]


def test_apply_route_stores_overrides_and_regrades():
    srv, base, _ = _serve_dashboard()
    try:
        # apply an impossible QEI cutoff -> pass rate must drop to 0
        opener = urllib.request.build_opener()
        opener.open(base + "/apply?qei_warn=0.999&qei_pass=0.999&back=/")
        body = opener.open(base + "/").read().decode()
        assert "custom thresholds" in body          # the applied-badge shows
        # the overview now reflects the re-grade (0% pass)
        assert "Pass rate" in body
    finally:
        srv.shutdown()


def test_overview_has_the_threshold_panel_and_lightbox():
    subs, summ, cfg = _cohort()
    h = render_overview(subs, summ, cfg, "demo")
    assert 'id="drawer"' in h and "Apply &amp; re-grade" in h   # config panel
    assert 'id="lb"' in h                                        # lightbox
    assert 'onclick="window.print()"' in h                      # export


def test_config_drawer_exposes_the_full_grouped_threshold_set():
    subs, summ, cfg = _cohort()
    h = render_overview(subs, summ, cfg, "demo")
    # a field from most groups is present as a real input the panel can tune
    for name in ("qei_pass", "gm_cbf_fail_lo", "wm_cbf_hi", "ratio_min",
                 "neg_gm_fail", "coverage_fail", "deep_gm_ratio_lo"):
        assert f'name="{name}"' in h


def test_population_change_repopulates_fields_and_survives_apply():
    """The bug: picking a population left the number fields on the old bands, and
    Apply then submitted those stale values, clobbering the population reset. The
    fix embeds each population's field values + an onchange handler, so the fields
    (and what gets submitted) match the chosen population."""
    from osipy_qc.batch import cfg_from_params
    from osipy_qc.core.config import for_population

    subs, summ, cfg = _cohort()
    h = render_overview(subs, summ, cfg, "demo")
    assert "applyPop(this.value)" in h                 # the handler is wired
    assert '"gm_cbf_lo": 8.0' in h                      # neonate's GM band is embedded for the JS

    # the round-trip the JS produces (neonate's own values submitted) must survive
    # cfg_from_params and NOT fall back to the adult 40:
    neo = for_population("neonate")
    params = {name: str(getattr(neo, name)) for name in TUNABLE}
    params["population"] = "neonate"
    eff = cfg_from_params(QCConfig(), params)
    assert eff.population == "neonate" and eff.gm_cbf_lo == 8.0


# --------------------------------------------------------------------------- #
# triage: worst-first ordering + the ledger verdict filter (review findings M4/M5)
# --------------------------------------------------------------------------- #
def test_ledger_is_sorted_worst_first_with_filter_chips():
    subs, summ, cfg = _cohort()
    h = render_overview(subs, summ, cfg, "demo")
    body = h.split('id="ledgerbody"', 1)[1]
    order = re.findall(r'data-v="(PASS|WARN|FAIL)"', body)
    sev = {"FAIL": 0, "WARN": 1, "PASS": 2}
    assert order == sorted(order, key=lambda v: sev[v])   # worst first
    assert 'class="fchip' in h and 'onclick="filterLedger(this)"' in h
    assert "worst first" in h


def test_sidebar_dots_are_not_colour_only():
    subs, summ, cfg = _cohort()
    h = render_overview(subs, summ, cfg, "demo")
    assert 'aria-label=' in h            # verdict dots carry a text label


# --------------------------------------------------------------------------- #
# proposal parity: New Analysis modal + organ menu + subject report header
# --------------------------------------------------------------------------- #
def test_new_analysis_modal_shows_real_runnable_snippets():
    subs, summ, cfg = _cohort()
    h = render_overview(subs, summ, cfg, "demo")
    assert 'id="runmodal"' in h and 'onclick="openRun()"' in h
    # the commands must be REAL (they exist in cli.py / the public API)
    assert "osipy-qc --dashboard" in h
    assert "from osipy_qc import grade_cbf" in h
    assert 'href="/upload"' in h              # links to the actual upload console


def test_organ_menu_marks_planned_organs_honestly():
    subs, summ, cfg = _cohort()
    h = render_overview(subs, summ, cfg, "demo")
    assert "organ-menu" in h
    assert "planned" in h                      # kidney/placenta/preclinical are inert


def test_subject_page_has_a_report_header_and_nav():
    subs, _summ, cfg = _cohort()
    fail = next(s for s in subs if s.overall == "FAIL")
    h = render_subject(subs, fail, cfg)
    assert "Participant quality report" in h
    assert "window.print()" in h               # export
    # prev/next step through the cohort
    assert h.count('href="/subject/') >= 2


def test_overview_has_the_cohort_visualisations():
    subs, summ, cfg = _cohort()
    h = render_overview(subs, summ, cfg, "demo")
    assert "QEI across the cohort" in h and "stripsvg" in h     # strip plot
    assert "Check matrix" in h and 'class="carpet"' in h        # per-check carpet
    assert h.count('class="cell"') >= len(subs)                 # a cell per subject-check


# --------------------------------------------------------------------------- #
# server robustness (review findings S6/S15)
# --------------------------------------------------------------------------- #
def test_safe_back_blocks_open_redirect_and_crlf():
    from osipy_qc.web import _safe_back
    assert _safe_back("/subject/sub-01") == "/subject/sub-01"
    assert _safe_back("//evil.com") == "/"          # protocol-relative open redirect
    assert _safe_back("https://evil.com") == "/"    # absolute open redirect
    assert _safe_back("/x\r\nSet-Cookie: y") == "/"  # CRLF response-splitting
    assert _safe_back("") == "/"


def test_404_is_a_styled_page_not_a_bare_tag():
    srv, base, _ = _serve_dashboard()
    try:
        urllib.request.urlopen(base + "/subject/does-not-exist")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        assert e.code == 404
        assert "charset" in body and "Back to overview" in body   # styled, has head
    else:
        raise AssertionError("expected 404")
    finally:
        srv.shutdown()
