"""Tests for the visual report and the dependency-free image encoders.

The report answers the review point that the CLI's numbers alone are not enough
("for somebody who doesn't know what they're looking at, it would be difficult")
and Maria's "are the histograms going to be displayed?".

The images are encoded with the standard library only (zlib/struct for PNG,
f-strings for SVG), so these tests also pin the encoders themselves.
"""

import base64
import re
import struct

import numpy as np

from osipy_qc import run_qc
from osipy_qc.core.config import QCConfig, for_population
from osipy_qc.report_html import render_html
from osipy_qc.synth import synthetic_case
from osipy_qc.utils.imaging import (colorise, encode_png, histogram_svg,
                                    png_data_uri, slice_mosaic)


# --------------------------------------------------------------------------- #
# PNG encoder
# --------------------------------------------------------------------------- #
def test_encode_png_is_a_valid_png():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    raw = encode_png(rgb)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"           # magic
    w, h = struct.unpack(">II", raw[16:24])          # IHDR width/height
    assert (w, h) == (6, 4)
    assert raw[-8:-4] == b"IEND"


def test_png_data_uri_round_trips():
    rgb = np.full((3, 3, 3), 128, dtype=np.uint8)
    uri = png_data_uri(rgb)
    assert uri.startswith("data:image/png;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_encode_png_rejects_wrong_shape():
    try:
        encode_png(np.zeros((4, 4)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a non-(H,W,3) array")


# --------------------------------------------------------------------------- #
# Colour mapping
# --------------------------------------------------------------------------- #
def test_negative_voxels_are_painted_a_distinct_colour_not_hidden():
    """Negative CBF is physically impossible; the report must show it in the
    dedicated negative colour, not clip it into the bottom of the ramp."""
    from osipy_qc.utils.imaging import _NEG

    sl = np.array([[-5.0, 50.0]])
    rgb = colorise(sl, vmin=0, vmax=100)
    assert tuple(rgb[0, 0]) == _NEG                # the dedicated negative colour
    assert tuple(rgb[0, 1]) != _NEG                # a positive value uses the ramp


def test_slice_mosaic_shape_and_type():
    vol = np.random.default_rng(0).random((8, 8, 6)) * 60
    m = slice_mosaic(vol, n=6, cols=3)
    assert m.dtype == np.uint8 and m.ndim == 3 and m.shape[2] == 3
    assert m.shape[0] == 2 * 8 and m.shape[1] == 3 * 8   # 2 rows x 3 cols of 8x8


def test_slice_mosaic_accepts_4d_by_averaging():
    vol = np.random.default_rng(0).random((6, 6, 4, 3)) * 60
    assert slice_mosaic(vol, n=4, cols=2).ndim == 3


def test_slice_mosaic_survives_an_all_zero_volume():
    """A degenerate volume must not raise — a figure should never break a report."""
    assert slice_mosaic(np.zeros((4, 4, 3))).shape[2] == 3


# --------------------------------------------------------------------------- #
# SVG histogram
# --------------------------------------------------------------------------- #
def test_histogram_svg_is_well_formed():
    svg = histogram_svg(np.random.default_rng(0).normal(55, 8, 400))
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert svg.count("<rect") > 5


def test_histogram_svg_marks_negatives_blue():
    svg = histogram_svg(np.array([-20.0] * 50 + [50.0] * 50))
    assert "#3B6FA8" in svg      # negative bars
    assert "#C2571A" in svg      # positive bars


def test_histogram_svg_degenerate_input_does_not_raise():
    assert histogram_svg(np.array([1.0])).startswith("<svg")
    assert histogram_svg(np.array([])).startswith("<svg")


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def _demo_report():
    c = synthetic_case(quality="clean", seed=0)
    inputs = {"cbf": c.cbf, "gm": c.gm, "wm": c.wm, "csf": c.csf,
              "brain": c.brain, "voxel_mm": c.voxel_mm}
    return run_qc(inputs), inputs


def test_report_is_self_contained():
    """One file you can email: no external CSS, JS, fonts or images."""
    report, inputs = _demo_report()
    h = render_html(report, inputs=inputs)
    assert h.startswith("<!DOCTYPE html>")
    external = re.findall(r'(?:src|href)="(?!data:|#)', h)
    assert external == [], f"report reaches out to {external}"


def test_report_embeds_images_and_plots():
    report, inputs = _demo_report()
    h = render_html(report, inputs=inputs)
    assert "data:image/png;base64," in h     # brain slices
    assert "<svg" in h                        # the histogram Maria asked for


def test_report_shows_every_check():
    report, inputs = _demo_report()
    h = render_html(report, inputs=inputs)
    for r in report.results:
        assert r.check in h


def test_report_is_honest_about_uncalibrated_provenance():
    """The direct answer to 'how did you get this number?' — the report explains
    that some verdicts are *provisional* (decided by an uncalibrated cutoff, not a
    published threshold) and points to the full sourcing."""
    report, inputs = _demo_report()
    h = render_html(report, inputs=inputs)
    assert ">provisional<" in h
    assert ">uncalibrated<" in h
    assert "THRESHOLD_PROVENANCE.md" in h


def test_report_states_the_population_whose_bands_were_used():
    report, inputs = _demo_report()
    h = render_html(report, inputs=inputs, cfg=for_population("neonate_term"))
    assert "neonate_term" in h


def test_report_works_without_any_images():
    """A Stream-A run has no CBF map; the report must still render."""
    report = run_qc({"m0_type": "separate"})
    h = render_html(report, inputs={})
    assert h.startswith("<!DOCTYPE html>")
    assert "OVERALL" in h


def test_report_escapes_html_in_reasons():
    """A check reason must never be able to inject markup."""
    from osipy_qc.core.result import CheckResult, Verdict
    from osipy_qc.report import QCReport

    evil = CheckResult("x.evil", Verdict.PASS, reason="<script>alert(1)</script>")
    h = render_html(QCReport(Verdict.PASS, [evil]), inputs={})
    assert "<script>alert(1)</script>" not in h
    assert "&lt;script&gt;" in h


def test_provisional_uncalibrated_fail_renders_a_marker_not_a_hard_fail():
    """A FAIL from an uncalibrated cutoff must be visibly provisional in the report,
    so it never reads as a hard, evidence-backed failure (2026-07 review, M1)."""
    from osipy_qc.core.config import QCConfig
    from osipy_qc.core.result import CheckResult, Verdict
    from osipy_qc.report import QCReport

    prov = CheckResult("4.2.coverage", Verdict.FAIL, reason="only 40% covered",
                       provisional=True)
    h = render_html(QCReport(Verdict.FAIL, [prov]), inputs={})
    assert "provisional" in h                       # the marker shows
    assert "repeating-linear-gradient" in h          # the striped (not solid) rail


def test_check_cards_use_human_names_and_keep_the_id():
    """Cards show a friendly name (QEI, Coverage) but keep the dotted id for traceability."""
    from osipy_qc.core.result import CheckResult, Verdict
    from osipy_qc.report import QCReport

    r = CheckResult("1.qei", Verdict.PASS, metric={"qei": 0.8}, reason="ok")
    h = render_html(QCReport(Verdict.PASS, [r]), inputs={})
    assert ">QEI<" in h and "1.qei" in h
