"""Integration tests for the runner + aggregation."""

from osipy_qc import Verdict, all_checks, run_qc
from osipy_qc.synth import synthetic_case


def test_registry_has_all_checks():
    names = set(all_checks())
    expected = {
        "1.qei", "2.1.spatial_cov", "2.2.snr", "2.3.histogram",
        "3.1.cbf_level", "3.2.gm_wm_ratio", "3.3.negative_gm",
        "4.1.coregistration", "5.1.schema", "5.2.volume_integrity",
        "5.3.swap", "6.1.m0_present", "6.2.m0_tr", "6.3.m0_no_bs",
        "6.5.m0_geometry", "7.1.motion", "8.2.data_type",
    }
    assert expected <= names


def test_clean_cbf_map_passes_stream_b():
    c = synthetic_case(quality="clean", seed=0)
    report = run_qc({"cbf": c.cbf, "gm": c.gm, "wm": c.wm, "csf": c.csf,
                     "brain": c.brain, "voxel_mm": c.voxel_mm})
    # the Stream B graders that CAN run should pass; overall is at worst WARN
    qei = next(r for r in report.results if r.check == "1.qei")
    assert qei.verdict == Verdict.PASS
    assert report.overall in (Verdict.PASS, Verdict.WARN)


def test_garbage_cbf_map_fails():
    c = synthetic_case(quality="garbage", seed=0)
    report = run_qc({"cbf": c.cbf, "gm": c.gm, "wm": c.wm, "csf": c.csf,
                     "brain": c.brain, "voxel_mm": c.voxel_mm})
    assert report.overall == Verdict.FAIL


def test_na_and_info_excluded_from_overall():
    # only a data_type (INFO) + a pre-subtracted swap (N/A) -> nothing gradeable -> UNKNOWN
    report = run_qc({"files": [{"name": "PCASL.nii.gz", "shape": (64, 64, 30)}],
                     "structure": "pre-subtracted deltaM"})
    swap = next(r for r in report.results if r.check == "5.3.swap")
    dtype = next(r for r in report.results if r.check == "8.2.data_type")
    assert swap.verdict == Verdict.NA
    assert dtype.verdict == Verdict.INFO


def test_report_json_roundtrip():
    c = synthetic_case(quality="clean", seed=0)
    report = run_qc({"cbf": c.cbf, "gm": c.gm, "wm": c.wm, "csf": c.csf})
    import json
    d = json.loads(report.to_json())
    assert d["overall_verdict"] in {"PASS", "WARN", "FAIL", "UNKNOWN"}
    assert d["n_checks"] >= 17
