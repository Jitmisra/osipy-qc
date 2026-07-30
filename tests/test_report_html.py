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
from osipy_qc.utils.imaging import (colorbar_svg, colorise, encode_png,
                                    format_level, histogram_svg, mosaic_window,
                                    negative_colour, png_data_uri, slice_mosaic)


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
# Colour bar
# --------------------------------------------------------------------------- #
def test_colorbar_states_its_range_and_the_unit():
    """From the 2026-07 review: "Color bar with units (ml/100g/min) is missing".
    An unlabelled ramp cannot tell 60 from 600 — a healthy scan from a broken M0."""
    svg = colorbar_svg(0.0, 80.0)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "mL/100 g/min" in svg
    assert ">0</text>" in svg and ">80</text>" in svg


def test_colorbar_swatches_come_from_the_image_mapping():
    """The bar is coloured by `colorise`, the same call that paints the mosaic, so
    it cannot advertise a colour the image does not use: a window reaching below
    zero shows the negative colour there, and one that does not, does not."""
    assert negative_colour() in colorbar_svg(-8.0, 73.0)
    assert negative_colour() not in colorbar_svg(0.0, 73.0)


def test_colorbar_ticks_zero_when_the_window_reaches_below_it():
    """Where impossible values stop is the most informative point on the scale."""
    assert ">0</text>" in colorbar_svg(-8.0, 73.0)


def test_colorbar_uses_literal_colours_not_css_variables():
    """The bar is also served on its own as image/svg+xml, where a CSS variable
    resolves to nothing — and a literal survives a print path that drops
    backgrounds and re-declares every colour token."""
    assert "var(--" not in colorbar_svg(-8.0, 73.0)


def test_colorbar_degenerate_window_does_not_raise():
    """A figure should never break a report."""
    for lo, hi in [(0.0, 0.0), (5.0, 1.0), (float("nan"), 1.0)]:
        assert colorbar_svg(lo, hi).startswith("<svg")


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
    h = render_html(report, inputs=inputs, cfg=for_population("neonate"))
    assert "neonate" in h


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


def test_report_labels_the_mosaic_scale_with_this_scans_own_window():
    """The window is per-scan, so the bar has to show *this* scan's range and the
    report has to say it is autoscaled — otherwise two reports' colours look
    comparable and are not."""
    report, inputs = _demo_report()
    lo, hi = mosaic_window(inputs["cbf"])
    h = render_html(report, inputs=inputs)
    assert "mL/100 g/min" in h
    assert f"{format_level(lo)}&ndash;{format_level(hi)}" in h
    assert "autoscaled to this scan" in h


def test_every_mosaic_carries_a_scale():
    """A figure copied out of the report still has to state what its colours mean,
    so the bar goes on each mosaic rather than once per page."""
    report, inputs = _demo_report()
    h = render_html(report, inputs=inputs)
    mosaics = len(set(re.findall(r'src="(data:image/png;base64,[^"]+)"', h)))
    assert mosaics >= 2
    assert h.count("<title>Colour scale,") == mosaics


def test_mosaic_caption_names_the_colour_the_code_actually_paints():
    """The caption said "Blue = negative CBF" while `colorise` paints cyan
    (#3cc8e6) — and the ramp's own low end is a dark violet a reader reads as
    blue, so the caption pointed at ordinary low-perfusion voxels."""
    report, inputs = _demo_report()
    h = render_html(report, inputs=inputs)
    assert "Blue = negative CBF" not in h
    assert f"cyan ({negative_colour()})" in h


def test_both_mosaics_share_one_window_so_they_can_be_compared():
    """The GM-masked mosaic used to autoscale itself, so one report showed the same
    scan at two different apparent perfusion levels."""
    report, inputs = _demo_report()
    cbf, gm = np.asarray(inputs["cbf"], float), np.asarray(inputs["gm"], float)
    lo, hi = mosaic_window(cbf)
    overlay = np.where(gm > QCConfig().tissue_thresh, cbf, 0.0)
    h = render_html(report, inputs=inputs)
    assert png_data_uri(slice_mosaic(overlay, vmin=lo, vmax=hi)) in h


def test_figures_do_not_leak_a_stray_angle_bracket():
    """Wrapping a mosaic in its zoom link left the image tag's own closing ">"
    behind, so a literal > rendered between every mosaic and its caption."""
    report, inputs = _demo_report()
    h = render_html(report, inputs=inputs)
    assert "</a>>" not in h


def test_report_uses_one_spelling_of_the_cbf_unit():
    """Two spellings on one page reads as two different quantities."""
    report, inputs = _demo_report()
    h = render_html(report, inputs=inputs)
    assert "mL/100g/min" not in h


def test_report_shows_acquisition_params_or_says_not_provided():
    report, inputs = _demo_report()
    h = render_html(report, inputs=inputs)      # no acq params supplied
    assert "Acquisition" in h and "PLD" in h
    assert "not provided" in h                  # honest about absent facts
