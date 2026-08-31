"""End-to-end integration across every organ and every entry point.

The unit tests check each piece; this checks that the pieces are WIRED. Most of
the defects this file exists to catch were wiring, not logic: a control rendered
on the form with no server-side reader, a map arriving under a key its own
organ's checks do not read, an import naming a module that does not exist.
"""

from __future__ import annotations

import contextlib
import io as _io
import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

from osipy_qc import web
from osipy_qc.batch import tunable_groups
from osipy_qc.cli import main as cli_main
from osipy_qc.core.config import ORGANS, QCConfig, THRESHOLD_PROVENANCE, for_organ
from osipy_qc.core.registry import organs_covered
from osipy_qc.report import run_qc
from osipy_qc.synth import (synthetic_case, synthetic_kidney_case,
                            synthetic_placenta_case)

ORGAN_LIST = ("brain", "kidney", "placenta")
BOUNDARY = "----osipyqcintegration"


def _part(name, filename=None, data=b""):
    head = f'form-data; name="{name}"'
    if filename is not None:
        head += f'; filename="{filename}"'
    return (b"--" + BOUNDARY.encode() + b"\r\nContent-Disposition: "
            + head.encode() + b"\r\n\r\n" + data + b"\r\n")


def _nii(arr, voxel=(3.0, 3.0, 3.0)):
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "x.nii.gz")
        nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32),
                                 np.diag(list(voxel) + [1.0])), p)
        with open(p, "rb") as fh:
            return fh.read()


def _grade(body):
    fields = web._parse_multipart(body + b"--" + BOUNDARY.encode() + b"--\r\n",
                                  f"multipart/form-data; boundary={BOUNDARY}")
    out = web._grade_upload(fields)
    return out[0] if isinstance(out, tuple) else out


# --------------------------------------------------------------------------- #
# registry / config wiring
# --------------------------------------------------------------------------- #
def test_every_organ_is_registered_profiled_and_tunable():
    assert organs_covered() == {"brain": 20, "kidney": 19, "placenta": 15}
    assert set(ORGANS) == set(ORGAN_LIST)
    cfg = QCConfig()
    for organ in ORGAN_LIST:
        assert for_organ(organ).organ == organ
        missing = [f for _g, fields in tunable_groups(organ) for f, _l in fields
                   if not hasattr(cfg, f)]
        assert not missing, f"{organ} exposes thresholds that do not exist: {missing}"


def test_no_provenance_row_names_a_field_that_does_not_exist():
    """--provenance and the report both read this table; a stale name there
    prints a threshold nobody can set."""
    cfg = QCConfig()
    ghosts = [k for k in THRESHOLD_PROVENANCE if not hasattr(cfg, k)]
    assert not ghosts, ghosts


@pytest.mark.parametrize("organ", ORGAN_LIST)
def test_an_organ_runs_with_no_inputs_at_all_and_never_crashes(organ):
    """run_qc swallows exceptions into 'check error: ...', a stack-trace fragment
    where a reader needs an instruction. No check may reach that path."""
    rep = run_qc({}, cfg=QCConfig(organ=organ))
    assert len(rep.results) == organs_covered()[organ]
    assert not [r for r in rep.results if "check error" in r.reason]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("organ", ORGAN_LIST)
def test_organ_demo_runs_from_the_cli(organ):
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli_main(["--organ-demo", organ])
    assert rc == 0 and "OVERALL" in buf.getvalue()


def test_organ_demo_honours_html_and_json(tmp_path):
    """--organ-demo returned before the --html block, and the import it then
    gained named a module (osipy_qc.render) that does not exist."""
    out = str(tmp_path / "r.html")
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli_main(["--organ-demo", "kidney", "--html", out])
    assert os.path.getsize(out) > 5000
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli_main(["--organ-demo", "placenta", "--json"])
    assert '"overall_verdict"' in buf.getvalue()


# --------------------------------------------------------------------------- #
# the console
# --------------------------------------------------------------------------- #
def test_the_form_offers_each_organ_its_own_controls():
    page = web._upload_page()
    for organ in ORGAN_LIST:
        assert f'name="organ" value="{organ}"' in page
    # masks, per side, because R10.1 reports the two kidneys separately
    for kind in ("kidney", "cortex", "medulla"):
        for side in ("left", "right"):
            assert f'name="kidney__{kind}_{side}"' in page
    assert 'name="placenta__placenta_mask"' in page
    # the facts that gate whole checks
    for field in ("kidney__units", "kidney__pld_or_ti_s", "placenta__declared_units",
                  "placenta__gestational_age_wk", "placenta__labelling_scheme"):
        assert f'name="{field}"' in page, field
    # thresholds are organ-scoped, not the brain's for everyone
    assert page.count('data-thr="kidney_') == 20
    assert page.count('data-thr="placenta_') == 19
    assert "function applyOrgan" in page


def test_the_error_page_still_renders():
    assert len(web._upload_page("something went wrong")) > 40_000


def test_brain_upload_is_unaffected_by_the_organ_work():
    c = synthetic_case(quality="clean", seed=0)
    body = (_part("cbf", "cbf.nii.gz", _nii(c.cbf))
            + _part("gm", "gm.nii.gz", _nii(c.gm))
            + _part("wm", "wm.nii.gz", _nii(c.wm))
            + _part("organ", data=b"brain") + _part("population", data=b"adult")
            + _part("files", "", b"") * 2)
    data = _grade(body)
    assert data["verdict"] == "PASS"
    assert not [x for x in data["checks"] if x["id"][0] in "kp"]


def test_kidney_upload_carries_masks_and_facts_all_the_way_through():
    """Before the organ sections existed this reached ONE of nine CBF-map checks,
    because the masks had nowhere to arrive and the facts had no fields."""
    c = synthetic_kidney_case(quality="clean", seed=0)
    vox = (2.0, 2.0, 8.0)
    body = _part("cbf", "rbf.nii.gz", _nii(c.rbf, vox))
    for side in ("left", "right"):
        for kind, masks in (("kidney", c.kidney_masks), ("cortex", c.cortex_masks),
                            ("medulla", c.medulla_masks)):
            body += _part(f"kidney__{kind}_{side}", f"{kind}.nii.gz", _nii(masks[side], vox))
    for key, val in (("units", "mL/100g/min"), ("labelling", "pCASL"),
                     ("pld_or_ti_s", "1.4"), ("field_strength_t", "3"),
                     ("breathing_strategy", "free breathing"), ("readout", "3D")):
        body += _part(f"kidney__{key}", data=val.encode())
    body += (_part("organ", data=b"kidney") + _part("population", data=b"adult")
             + _part("files", "", b"") * 2)
    data = _grade(body)
    ids = {x["id"]: x for x in data["checks"]}
    assert data["coverage"]["graded"] >= 4
    # the consensus quantity, per kidney, actually computed from the uploaded map
    assert ids["k3.1.cortical_rbf"]["verdict"] == "INFO"
    assert "300" in ids["k3.1.cortical_rbf"]["reason"]
    assert ids["k4.1.mask_integrity"]["verdict"] == "PASS"
    assert "6 mask" in ids["k4.1.mask_integrity"]["reason"]
    assert not [x for x in data["checks"] if x["id"][0].isdigit()]


def test_placenta_upload_can_declare_units_and_context():
    """P2.1 gates every magnitude check on a declared unit; with no field for it
    a placenta upload could never get past."""
    c = synthetic_placenta_case(quality="clean", seed=0)
    body = (_part("cbf", "perf.nii.gz", _nii(c.perfusion))
            + _part("placenta__placenta_mask", "pm.nii.gz", _nii(c.placenta_mask)))
    for key, val in (("declared_units", "mL/100g/min"), ("labelling_scheme", "VSASL"),
                     ("gestational_age_wk", "30"), ("maternal_position", "lateral"),
                     ("field_strength_T", "3"), ("mask_source", "manual_on_m0"),
                     ("normalisation_mode", "scalar"),
                     ("registration_model", "non-rigid (DSVR)"),
                     ("lambda", "0.9"), ("alpha", "0.767"), ("t1_blood_ms", "1650")):
        body += _part(f"placenta__{key}", data=val.encode())
    body += (_part("organ", data=b"placenta") + _part("population", data=b"adult")
             + _part("files", "", b"") * 2)
    data = _grade(body)
    ids = {x["id"]: x for x in data["checks"]}
    assert ids["p2.1.units_declaration"]["verdict"] == "PASS"
    assert "manual_on_m0" in ids["p3.1.mask_integrity"]["reason"]
    assert data["verdict"] == "PASS"
    assert data["coverage"]["complete"] is True


def test_a_mask_on_the_wrong_grid_is_actionable_not_a_stack_trace():
    c = synthetic_kidney_case(quality="clean", seed=0)
    vox = (2.0, 2.0, 8.0)
    body = (_part("cbf", "rbf.nii.gz", _nii(c.rbf, vox))
            + _part("kidney__cortex_left", "c.nii.gz", _nii(np.ones((4, 4, 4)), vox))
            + _part("kidney__units", data=b"mL/100g/min")
            + _part("organ", data=b"kidney") + _part("population", data=b"adult")
            + _part("files", "", b"") * 2)
    data = _grade(body)
    assert any("resample" in x["reason"] for x in data["checks"])
    assert not [x for x in data["checks"] if "check error" in x["reason"]]


# --------------------------------------------------------------------------- #
# branding
# --------------------------------------------------------------------------- #
def test_every_page_carries_the_favicon():
    """Without it the browser tab shows a generic globe next to the title."""
    from osipy_qc._webassets import favicon_link
    from osipy_qc.report_html import render_html

    link = favicon_link()
    assert link.startswith('<link rel="icon" type="image/svg+xml"')
    assert 'data:image/svg+xml,' in link

    c = synthetic_case(quality="clean", seed=0)
    report = run_qc(dict(cbf=c.cbf, gm=c.gm, wm=c.wm, csf=c.csf,
                         brain=c.brain, voxel_mm=c.voxel_mm))
    for page in (web._upload_page(), web._upload_page("an error"),
                 web._error_page(404, "nope"), render_html(report)):
        assert '<link rel="icon"' in page


def test_the_favicon_data_uri_decodes_to_valid_svg():
    """A data URI with an unescaped '#' or quote silently yields a blank tab,
    and nothing else in the page would look wrong."""
    import re
    from urllib.parse import unquote

    from osipy_qc._webassets import favicon_link
    uri = re.search(r'href="data:image/svg\+xml,([^"]+)"', favicon_link()).group(1)
    svg = unquote(uri)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert 'viewBox="0 0 40 40"' in svg
    assert "linearGradient" in svg          # the gradient def travels with it
    assert '"' not in uri                   # nothing that would close the attribute


# --------------------------------------------------------------------------- #
# defects the parallel probes found in the console
# --------------------------------------------------------------------------- #
def test_the_uploaded_map_wins_over_a_lookalike_in_the_raw_drop():
    """The worst defect of the session. load_organ_folder set rbf_map from the
    raw folder, and the alias guard was `if alias not in inputs`, so it never
    fired - the console silently graded a DIFFERENT file from the one the user
    put in the box, and reported a confident number about it."""
    c = synthetic_kidney_case(quality="clean", seed=0)
    vox = (2.0, 2.0, 8.0)
    decoy = np.full_like(c.rbf, 999.0)
    body = (_part("cbf", "MY_MAP.nii.gz", _nii(c.rbf, vox))
            + _part("files", "ASL_RBF.nii.gz", _nii(decoy, vox)))
    for side in ("left", "right"):
        body += _part(f"kidney__cortex_{side}", "c.nii.gz", _nii(c.cortex_masks[side], vox))
    body += (_part("kidney__units", data=b"mL/100g/min")
             + _part("organ", data=b"kidney") + _part("population", data=b"adult")
             + _part("files", "", b""))
    data = _grade(body)
    reason = next(x["reason"] for x in data["checks"] if x["id"] == "k3.1.cortical_rbf")
    assert "300" in reason and "999" not in reason, reason


def test_strict_can_be_turned_off_for_every_organ():
    """The marker for 'this form carried the control' was thr_qei_pass, a
    brain-only threshold, so unchecking Strict was silently ignored for kidney
    and placenta - the two organs where it matters most, since almost all their
    thresholds are uncalibrated."""
    g = synthetic_kidney_case(quality="garbage", seed=0)
    vox = (2.0, 2.0, 8.0)
    base = _part("cbf", "r.nii.gz", _nii(g.rbf, vox))
    for side in ("left", "right"):
        base += _part(f"kidney__cortex_{side}", "c.nii.gz", _nii(g.cortex_masks[side], vox))
    base += (_part("kidney__units", data=b"mL/100g/min")
             + _part("organ", data=b"kidney") + _part("population", data=b"adult")
             + _part("files", "", b"") * 2)
    assert _grade(base + _part("strict", data=b"1"))["verdict"] == "FAIL"
    assert _grade(base)["verdict"] == "WARN"      # unchecked = not submitted


def test_a_map_in_arbitrary_units_is_not_graded_against_a_per_mass_band():
    """Accepting any non-empty string meant a caller who correctly declared
    'a.u.' got a confident verdict from a band stated per 100 g."""
    from osipy_qc.checks.kidney import _units_declared
    assert _units_declared("mL/100g/min") and _units_declared("mL/min/100mL")
    assert not _units_declared("a.u.") and not _units_declared("banana")
    assert not _units_declared("") and not _units_declared(None)

    c = synthetic_kidney_case(quality="clean", seed=0)
    from osipy_qc.checks.kidney import cortical_rbf_check
    r = cortical_rbf_check(rbf_map=c.rbf, cortex_masks=c.cortex_masks, units="a.u.")
    assert r.verdict.value == "UNKNOWN"
    # and it must say what actually happened, not "not declared"
    assert "declared as 'a.u.'" in r.reason


def test_declaring_arbitrary_units_is_not_a_physiological_claim():
    """`quantified` was set from the mere presence of a units string, so a map
    declared as arbitrary units was treated as a physiological claim - the
    opposite of what the caller said."""
    c = synthetic_placenta_case(quality="clean", seed=0)
    body = (_part("cbf", "p.nii.gz", _nii(c.perfusion))
            + _part("placenta__placenta_mask", "m.nii.gz", _nii(c.placenta_mask))
            + _part("placenta__declared_units", data="a.u.".encode())
            + _part("organ", data=b"placenta") + _part("population", data=b"adult")
            + _part("files", "", b"") * 2)
    data = _grade(body)
    units = next(x for x in data["checks"] if x["id"] == "p2.1.units_declaration")
    assert units["verdict"] == "PASS"            # declaring a.u. is legitimate
    assert units["metric"]["quantified"] is False


def test_organ_folder_loading_works_from_the_cli():
    """`--organ kidney <folder>` raised NameError: the import named only
    load_folder while the branch called load_organ_folder."""
    from osipy_qc.io import load_folder, load_organ_folder   # noqa: F401
    import osipy_qc.cli as cli
    src = open(cli.__file__).read()
    assert "from .io import load_folder, load_organ_folder" in src


def test_the_per_role_boxes_beat_the_filename():
    """The escape hatch that exists SOLELY to defeat filename guessing was itself
    pure filename guessing: the field name was discarded and the role re-derived
    from the name. A scan called anon_0042.nii.gz - what an anonymiser emits, in
    the multi-centre setting this toolbox is for - was dropped, and the report
    said "no M0" about a file the user had put in the box marked M0."""
    from osipy_qc.checks.schema import classify_role
    from osipy_qc.synth import synthetic_control_label

    # these are all names classify_role cannot place, and all realistic
    for name in ("anon_0042.nii.gz", "IM_0001.nii.gz", "proton_density.nii.gz",
                 "PD.nii.gz", "FAIR.nii.gz", "Q2TIPS.nii.gz"):
        assert classify_role(name) == "other", name

    asl = synthetic_control_label(n_pairs=8)
    body = (_part("raw_asl", "anon_0042.nii.gz", _nii(asl))
            + _part("raw_m0", "anon_0043.nii.gz", _nii(asl[..., 0]))
            + _part("organ", data=b"brain") + _part("population", data=b"adult")
            + _part("files", "", b"") * 2)
    ids = {x["id"]: x["verdict"] for x in _grade(body)["checks"]}
    assert ids["6.1.m0_present"] == "PASS", "the M0 box was ignored"
    assert ids["5.2.volume_integrity"] == "PASS"
    assert ids["5.3.swap"] == "PASS"


def test_role_overrides_reach_the_loader():
    """The override has to survive into detect_dataset too, or 8.2 reports a
    dataset with no ASL series while the checks grade one."""
    import tempfile as _tf

    from osipy_qc.io import load_folder
    from osipy_qc.synth import synthetic_control_label
    with _tf.TemporaryDirectory() as d:
        p = os.path.join(d, "anon_0042.nii.gz")
        nib.save(nib.Nifti1Image(synthetic_control_label(n_pairs=4).astype(np.float32),
                                 np.diag([3., 3., 3., 1.])), p)
        plain = load_folder(d, load_arrays=False)
        assert plain["detected"]["structure"] == "unknown"
        fixed = load_folder(d, load_arrays=False,
                            role_overrides={"anon_0042.nii.gz": "asl"})
        assert "control/label" in fixed["detected"]["structure"]
        assert fixed["asl_shape"][3] == 8


def test_the_outlier_thresholds_the_console_renders_actually_grade():
    """kidney_outlier_sd and kidney_outlier_voxel_frac were rendered as editable
    but read by nothing - the rule parameters came from a hard-coded table, so a
    user could type any value and the rejection count did not move."""
    from osipy_qc.checks.kidney import kidney_outlier_rate_check
    rng = np.random.default_rng(11)
    series = rng.normal(10, 1.0, (6, 6, 3, 12))
    series[..., 5] += 20.0
    masks = {"left": np.ones((6, 6, 3), bool)}

    loose = kidney_outlier_rate_check(delta_m_4d=series, kidney_masks=masks,
                                      cfg=QCConfig(organ="kidney", kidney_outlier_sd=6.0))
    tight = kidney_outlier_rate_check(delta_m_4d=series, kidney_masks=masks,
                                      cfg=QCConfig(organ="kidney", kidney_outlier_sd=2.0))
    assert loose.metric["per_kidney"]["left"]["n_rejected"] < \
        tight.metric["per_kidney"]["left"]["n_rejected"]
    # and a report must say when the applied rule is no longer the published one
    assert loose.metric["parameters_customised"] is True
    assert "overridden" in loose.metric["rule"]
    assert tight.metric["parameters_customised"] is False
    assert tight.metric["published_parameters"] == {"k": 2.0, "limit": 0.20}


def test_no_html_entity_renders_as_literal_text():
    """_dropzone escapes its title, so an HTML entity written there is
    double-escaped and the user sees the literal text "&mdash;". The page is
    UTF-8, so labels must carry real characters."""
    import re
    page = web._upload_page()
    assert not re.findall(r"&amp;[a-z]+;", page), re.findall(r"&amp;[a-z]+;", page)
    assert "Kidney — left" in page and "Cortex — right" in page


def test_masks_supplied_through_the_boxes_are_counted_as_masks():
    """k5.2 counted only files it could name from the raw drop, so a user who
    supplied six masks through the per-organ boxes was told '0 mask file(s)' -
    which reads as a failed upload when the masks graded perfectly well."""
    from osipy_qc.checks.kidney import kidney_data_type_check
    c = synthetic_kidney_case(quality="clean", seed=0)
    r = kidney_data_type_check(files=[{"name": "M0.nii.gz"}],
                               kidney_masks=c.kidney_masks,
                               cortex_masks=c.cortex_masks,
                               medulla_masks=c.medulla_masks)
    assert r.metric["n_masks"] == 6
    assert "0 mask" not in r.reason
    assert "6 mask(s) supplied directly" in r.reason


def test_the_fields_that_gate_whole_checks_are_marked_on_the_form():
    """An empty units box turns two kidney checks into UNKNOWN and every
    placenta magnitude check into a refusal to grade. Nothing on the form said
    so, so the first sign was an unexplained UNKNOWN in the report."""
    page = web._upload_page()
    assert page.count("gates checks") == 3      # kidney units, placenta units, GA


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
def test_the_mosaic_slices_an_axis_it_chose_not_one_it_assumed():
    """slice_mosaic hard-coded axis 2. On a two-kidney volume whose organs are
    separated along that axis it sliced BETWEEN them: half the panels came out
    empty and no panel ever showed both kidneys, which is the one thing a
    per-kidney report has to show."""
    from osipy_qc.utils.imaging import slice_axis_of
    # brain-shaped and real renal data keep the conventional axis
    assert slice_axis_of((36, 36, 28)) == 2
    assert slice_axis_of((64, 32, 16)) == 2
    assert slice_axis_of((64, 64, 36)) == 2
    # the kidney phantom does not
    assert slice_axis_of((32, 40, 40)) == 0
    # a thick-slice acquisition is decided by the voxel size, not the shape
    assert slice_axis_of((32, 40, 40), (2.0, 2.0, 8.0)) == 2


def test_the_mosaic_shows_both_kidneys():
    from osipy_qc.utils.imaging import slice_axis_of
    from osipy_qc.utils.roi import component_sizes
    c = synthetic_kidney_case(quality="clean", seed=0)
    axis = slice_axis_of(c.rbf.shape)
    idx = np.linspace(0, c.rbf.shape[axis] - 1, 12).round().astype(int)
    both = sum(1 for k in idx
               if len(component_sizes((np.take(c.rbf, k, axis=axis) > 0)[..., None])) == 2)
    assert both >= 6, f"only {both}/12 panels show both kidneys"


def test_no_data_and_the_lowest_measured_value_look_different():
    """The ramp's own floor is (0,0,4) - visually black - and voxels outside the
    organ are black too, so genuine low-perfusion tissue rendered identically to
    a hole. On a kidney that made the medulla look like a hole punched through
    the organ."""
    from osipy_qc.utils.imaging import colorise
    vmin, vmax = 110.0, 313.0
    bg = colorise(np.array([[0.0]]), vmin, vmax)[0, 0].astype(int)
    low = colorise(np.array([[120.0]]), vmin, vmax)[0, 0].astype(int)
    assert tuple(bg) == (0, 0, 0)                  # no data stays black
    assert np.linalg.norm(bg - low) > 80           # was 27 - invisible


def test_the_colour_bar_shows_the_colours_the_image_actually_uses():
    """A legend that misstates its own image is worse than no legend."""
    from osipy_qc.utils.imaging import colorise, ramp_stops
    vmin, vmax = 110.0, 313.0
    for pos, hexc in ramp_stops():
        val = vmin + pos * (vmax - vmin)
        rgb = colorise(np.array([[val]]), vmin, vmax)[0, 0]
        assert hexc == "#%02x%02x%02x" % tuple(int(c) for c in rgb), pos


def test_the_figure_is_named_for_its_organ():
    from osipy_qc.api import _figure_list
    for organ, title in (("brain", "CBF map"), ("kidney", "RBF map"),
                         ("placenta", "Perfusion map")):
        c = synthetic_kidney_case(quality="clean", seed=0)
        figs = _figure_list({"cbf": c.rbf}, QCConfig(organ=organ))
        assert figs[0]["title"] == title
        assert "axial" not in figs[0]["caption"]   # an orientation we never measured


def test_the_colour_bar_text_is_not_blown_up_by_the_stretch():
    """The report stretches this SVG to the figure width, so everything inside
    scales by that ratio. At the old 280 px viewBox a font-size of 10 came out
    about six times the caption beneath it, with a colour bar taller than the
    thing it explained. Text size here is a FRACTION of the viewBox."""
    import re
    from osipy_qc.utils.imaging import colorbar_svg
    svg = colorbar_svg(110.0, 313.0)
    width = int(re.search(r'width="(\d+)"', svg).group(1))
    size = int(re.search(r'font-size="(\d+)"', svg).group(1))
    ratio = size / width
    # the report caption is 0.78rem in a ~900 px figure, about 1.4%
    assert 0.010 < ratio < 0.020, f"tick text is {ratio:.1%} of the bar width"
    # and the bar must be wide relative to its height, or it dominates the figure
    height = int(re.search(r'height="(\d+)"', svg).group(1))
    assert width / height > 10


def test_the_report_names_the_map_after_its_organ():
    """A placenta report headed 'The CBF map', over a figure captioned 'CBF map -
    evenly spaced axial slices', names a quantity and an orientation it never
    had."""
    from osipy_qc.report_html import render_html
    for organ, section, title in (("brain", "The CBF map", "CBF map"),
                                  ("kidney", "The RBF map", "RBF map"),
                                  ("placenta", "The perfusion map", "Perfusion map")):
        cfg = QCConfig(organ=organ)
        c = synthetic_kidney_case(quality="clean", seed=0)
        html = render_html(run_qc({}, cfg=cfg), inputs={"cbf": c.rbf}, cfg=cfg)
        assert f"Checks &mdash; {section}" in html, organ
        assert title in html, organ
        assert "axial" not in html.lower(), organ


def test_the_documented_strict_contract_is_the_real_one():
    """README and USAGE both claimed "uncalibrated thresholds never drive a FAIL
    on their own". That was never true of the code: strict=True is the default
    and five FAIL branches are gated on it. The real contract is weaker and is
    what the docs now state - an uncalibrated FAIL is marked provisional, and
    --no-strict demotes every provisional FAIL, leaving the published ones."""
    from osipy_qc.synth import synthetic_case
    c = synthetic_case(quality="garbage", seed=0)
    inputs = dict(cbf=c.cbf, gm=c.gm, wm=c.wm, csf=c.csf,
                  brain=c.brain, voxel_mm=c.voxel_mm)

    strict = run_qc(inputs, cfg=QCConfig(strict=True))
    lenient = run_qc(inputs, cfg=QCConfig(strict=False))
    f_strict = {r.check: r.provisional for r in strict.results if r.verdict.value == "FAIL"}
    f_lenient = {r.check: r.provisional for r in lenient.results if r.verdict.value == "FAIL"}

    # an uncalibrated cut-off DOES reach a FAIL by default - the docs must not
    # claim otherwise
    assert any(prov for prov in f_strict.values())
    # every surviving failure is non-provisional, i.e. published-backed
    assert f_lenient and not any(f_lenient.values())
    assert set(f_lenient) < set(f_strict)


def test_the_docs_do_not_claim_uncalibrated_never_fails():
    """The claim was in two shipped files. A future edit must not reintroduce
    it, because it is the one guarantee a reviewer would check first."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    for name in ("README.md", "USAGE.md"):
        text = (root / name).read_text()
        assert "never drive a FAIL on their own" not in text, name
        assert "provisional" in text, f"{name} must explain the real contract"


def test_no_strict_is_reachable_from_the_command_line():
    """The documented remedy was QCConfig(strict=False) - Python-API only - so a
    CLI or website user had no way to opt out of a grading rule the docs told
    them they could turn off."""
    import osipy_qc.cli as cli
    src = open(cli.__file__).read()
    assert '"--no-strict"' in src
    assert "strict=not args.no_strict" in src


def test_the_provenance_counts_in_the_docs_are_current():
    """README and USAGE both printed 11/9/16 while the config held 12/16/32 - a
    board member running `osipy-qc --provenance` would see a 2x discrepancy."""
    import pathlib
    from osipy_qc.core.config import THRESHOLD_PROVENANCE, uncalibrated_fields
    counts = {}
    for _k, (level, _c, _n) in THRESHOLD_PROVENANCE.items():
        counts[level.value] = counts.get(level.value, 0) + 1
    assert len(uncalibrated_fields()) == counts["uncalibrated"]
    # THRESHOLD_PROVENANCE.md is generated, so IT is where the tallies must be
    # current; the README quotes them in prose and is checked by the generator's
    # own --check mode rather than by string matching here.
    doc = (pathlib.Path(__file__).resolve().parent.parent / "THRESHOLD_PROVENANCE.md").read_text()
    for level in ("published", "implementation", "uncalibrated"):
        assert f"{level} | {counts[level]} " in doc, level


def test_a_single_volume_in_a_4d_container_is_graded_not_rejected():
    """(X, Y, Z, 1) is how several vendors write one pre-subtracted deltaM, and
    NIfTI permits it. Unsqueezed it reached the 5 mm smoother, which raised, and
    the report showed "check error: smooth_fwhm expects a 3-D volume" - a
    stack-trace fragment where a reader needs an instruction."""
    from osipy_qc.synth import synthetic_case
    c = synthetic_case(quality="clean", seed=0)
    flat = dict(cbf=c.cbf, gm=c.gm, wm=c.wm, csf=c.csf,
                brain=c.brain, voxel_mm=c.voxel_mm)
    fourd = {k: (v[..., None] if isinstance(v, np.ndarray) else v)
             for k, v in flat.items()}

    a, b = run_qc(flat, cfg=QCConfig()), run_qc(fourd, cfg=QCConfig())
    assert not [r for r in b.results if "check error" in r.reason]
    qa = next(r for r in a.results if r.check == "1.qei")
    qb = next(r for r in b.results if r.check == "1.qei")
    assert qa.verdict is qb.verdict
    assert qa.metric["qei"] == pytest.approx(qb.metric["qei"])


def test_the_memory_guard_fits_inside_the_container_it_protects():
    """MAX_ARRAY_BYTES was 805 MB while the free Render container has 512 MB for
    the whole process, so the guard could never have protected the deployment:
    a request declaring 700 MB passed it and then OOM-killed the server."""
    from osipy_qc.io import MAX_ARRAY_BYTES
    assert MAX_ARRAY_BYTES < 512 * 1024 * 1024
    # and still clears the largest real input by a wide margin (a 208x300x320
    # T1 is about 160 MB as float64)
    assert MAX_ARRAY_BYTES > 170 * 1024 * 1024


def test_an_empty_folder_is_unknown_not_a_warning():
    """An empty folder produced WARN on two checks: 5.1 said "no BIDS sidecar,
    fields inferred from NIfTI + filenames" when there were no NIfTIs to infer
    from, and 6.1 said "no M0" about a dataset that did not exist. Both invent a
    finding out of nothing - the same class as the phantom-folder bug."""
    import tempfile

    from osipy_qc.io import load_folder
    with tempfile.TemporaryDirectory() as d:
        rep = run_qc(load_folder(d), cfg=QCConfig())
    assert rep.overall.value == "UNKNOWN"
    assert rep.coverage["graded"] == 0
    assert not [r for r in rep.results if r.verdict.value in ("PASS", "WARN", "FAIL")]


def test_a_folder_with_data_still_grades_normally():
    """The empty-folder guard must not silence a real one."""
    import tempfile

    import nibabel as nib

    from osipy_qc.io import load_folder
    from osipy_qc.synth import synthetic_control_label
    with tempfile.TemporaryDirectory() as d:
        nib.save(nib.Nifti1Image(synthetic_control_label(n_pairs=4).astype(np.float32),
                                 np.diag([3., 3., 3., 1.])),
                 os.path.join(d, "PCASL.nii.gz"))
        rep = run_qc(load_folder(d), cfg=QCConfig())
    assert rep.coverage["graded"] > 0
    ids = {r.check: r.verdict.value for r in rep.results}
    assert ids["5.1.schema"] == "WARN"          # genuinely no sidecar
    assert ids["6.1.m0_present"] == "WARN"      # genuinely no M0


def test_a_total_fov_mismatch_fails_rather_than_reporting_an_empty_mask():
    """coverage_fraction returned 0.0 both for an empty ROI and for an ROI the
    CBF map reaches none of, so 4.2 reported a TOTAL FOV mismatch - the one
    failure it exists to catch - as "empty tissue mask, cannot assess coverage".
    Wrong verdict, and a false statement about the mask."""
    from osipy_qc.checks.coreg import coverage_check
    full_gm = np.full((10, 10, 10), 0.9)

    no_overlap = coverage_check(cbf=np.zeros((10, 10, 10)), gm=full_gm, cfg=QCConfig())
    assert no_overlap.verdict.value == "FAIL"
    assert "covers NONE" in no_overlap.reason

    empty_roi = coverage_check(cbf=np.ones((10, 10, 10)), gm=np.zeros((10, 10, 10)),
                               cfg=QCConfig())
    assert empty_roi.verdict.value == "UNKNOWN"
    assert "mask is empty" in empty_roi.reason


def test_a_non_positive_m0_tr_is_refused_not_corrected():
    """1/(1 - exp(-TR/T1)) is infinite at TR=0 and negative below it, so a bad
    value produced a confident "correct by xinf"."""
    from osipy_qc.checks.m0 import m0_tr_check
    for bad in (0.0, -3.0):
        r = m0_tr_check(m0_tr_s=bad)
        assert r.verdict.value == "UNKNOWN"
        assert "not a repetition time" in r.reason
    assert m0_tr_check(m0_tr_s=6.0).verdict.value == "PASS"


def test_a_4d_singleton_mask_does_not_crash_the_kidney_checks():
    """(X, Y, Z, 1) is a legal mask shape that segmentation tools emit. Five
    checks reported "check error: operands could not be broadcast" about it."""
    from osipy_qc.synth import synthetic_kidney_case
    c = synthetic_kidney_case(quality="clean", seed=0)
    fourd = {s: m[..., None] for s, m in c.cortex_masks.items()}
    rep = run_qc(dict(rbf_map=c.rbf, cortex_masks=fourd, units="mL/100g/min"),
                 cfg=QCConfig(organ="kidney"))
    assert not [r for r in rep.results if "check error" in r.reason]
    lvl = next(r for r in rep.results if r.check == "k3.1.cortical_rbf")
    assert "300" in lvl.reason


# --------------------------------------------------------------------------- #
# deployment posture
# --------------------------------------------------------------------------- #
def test_binding_off_loopback_must_be_explicit():
    """--host defaulted from a HOST environment variable. A stray HOST in the
    shell moved the server off loopback, and _host_ok stops enforcing its
    DNS-rebinding check the moment the bind address is not loopback - so an
    ambient variable silently downgraded the security posture."""
    import osipy_qc.cli as cli
    src = open(cli.__file__).read()
    assert 'os.environ.get("HOST"' not in src
    assert 'default="127.0.0.1"' in src


def test_error_pages_do_not_disclose_server_paths():
    """Library exceptions embed absolute paths - nibabel names the temp file it
    could not read - and the message is rendered into a page the uploader sees."""
    from osipy_qc.web import _client_safe
    msg = _client_safe(Exception(
        "File /var/folders/px/T/osipy_qc_ab12/raw/x.nii.gz is not a gzip file"))
    assert "/var/folders" not in msg and "osipy_qc_ab12" not in msg
    assert "x.nii.gz" in msg          # the useful part survives
    # a message with no path is left alone
    plain = _client_safe(ValueError("mask shape (4,4,4) != volume shape (8,8,4)"))
    assert "(4,4,4)" in plain
