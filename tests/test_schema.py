"""Known-answer tests for Module 5 + 8.2 (detection, schema, integrity, swap)."""

from osipy_qc.checks.schema import (
    data_type_check,
    detect_dataset,
    schema_check,
    swap_check,
    volume_integrity_check,
)
from osipy_qc.core import Verdict
from osipy_qc.synth import synthetic_control_label


def test_detect_ge_presubtracted():
    files = [
        {"name": "PCASL_999_S_9999.nii.gz", "shape": (128, 128, 40), "voxel_mm": (1.88, 1.88, 4.0)},
        {"name": "M0_999_S_9999.nii.gz", "shape": (128, 128, 40), "voxel_mm": (1.88, 1.88, 4.0)},
        {"name": "MPRAGE_999_S_9999.nii.gz", "shape": (256, 256, 196), "voxel_mm": (1.0, 1.0, 1.0)},
    ]
    d = detect_dataset(files, context="GE_PCASL_Product_Sequence")
    assert d["vendor"] == "GE"
    assert d["structure"].startswith("pre-subtracted")
    assert d["m0"] == "separate"
    assert d["t1_structural"] is True


def test_detect_siemens_2d_no_m0():
    files = [
        {"name": "PCASL.nii.gz", "shape": (80, 80, 20, 80), "voxel_mm": (2.75, 2.75, 6.0)},
        {"name": "MPRAGE.nii.gz", "shape": (170, 256, 256), "voxel_mm": (1.0, 1.0, 1.0)},
    ]
    d = detect_dataset(files)
    assert d["readout"] == "2D"
    assert d["n_volumes"] == 80
    assert d["m0"] == "absent"


def test_detect_bs_3d():
    files = [
        {"name": "PCASL.nii.gz", "shape": (88, 88, 52, 16), "voxel_mm": (2.5, 2.5, 2.5)},
        {"name": "M0.nii.gz", "shape": (88, 88, 52), "voxel_mm": (2.5, 2.5, 2.5)},
        {"name": "T1.nii.gz", "shape": (208, 300, 320), "voxel_mm": (0.8, 0.8, 0.8)},
    ]
    # folder/context flags BS via the foldername passed in a file name token
    files[0]["name"] = "Siemens_BS3DPCASL_PCASL.nii.gz"
    d = detect_dataset(files)
    assert d["readout"] == "3D"
    assert d["n_volumes"] == 16
    assert d["background_suppression"] is True
    assert d["m0"] == "separate"


def test_data_type_check_is_info():
    files = [{"name": "PCASL.nii.gz", "shape": (64, 64, 30, 40), "voxel_mm": (3.0, 3.0, 5.0)}]
    r = data_type_check(files=files)
    assert r.verdict == Verdict.INFO        # routing, excluded from aggregation


def test_schema_degrades_without_sidecar():
    r = schema_check(sidecar=None, detected={"vendor": "GE"})
    assert r.verdict == Verdict.WARN


def test_schema_valid_sidecar_pass():
    r = schema_check(sidecar={"ArterialSpinLabelingType": "PCASL",
                              "MRAcquisitionType": "3D", "PostLabelingDelay": 2.0})
    assert r.verdict == Verdict.PASS


def test_volume_integrity_even_pass_odd_fail():
    assert volume_integrity_check(n_volumes=80).verdict == Verdict.PASS
    assert volume_integrity_check(n_volumes=15).verdict == Verdict.FAIL
    assert volume_integrity_check(structure="pre-subtracted deltaM").verdict == Verdict.NA


def test_swap_normal_pass_and_swapped_fail():
    normal = synthetic_control_label(swapped=False, seed=1)
    swapped = synthetic_control_label(swapped=True, seed=1)
    assert swap_check(asl_4d=normal).verdict == Verdict.PASS
    assert swap_check(asl_4d=swapped).verdict == Verdict.FAIL


def test_swap_na_under_bs():
    series = synthetic_control_label(seed=2)
    assert swap_check(asl_4d=series, background_suppression=True).verdict == Verdict.NA
