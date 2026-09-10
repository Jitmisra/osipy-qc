"""
Module 1, second half - QEI-Net, the deep-learning quality index.

The classical QEI (`1.qei`) is a formula. This is a 3-D convolutional network
trained on rated CBF maps by Beltran Urbano et al., which returns one number in
[0, 1] where higher is better. The two are complementary rather than redundant,
and the saturation guard below is the reason why.

How it is wired
---------------
The model is NOT part of this package and never will be. Its weights are
unpublished, they were shared privately for integration work, and they are not
ours to redistribute. So this check shells out to the author's own inference
script in the author's own environment, and reads one float back from stdout:

    OSIPY_QEI_NET_PYTHON=/path/to/qei_env/bin/python
    OSIPY_QEI_NET_SCRIPT=/path/to/inference_package/src/run_qei.py

With neither set the check returns UNKNOWN and the rest of the report is
unaffected. That keeps this package on numpy + nibabel: torch, torchio and
SimpleITK live in the other environment, not in ours.

Why it reports and does not grade
---------------------------------
No validated cut-off for "acceptable" has been published for this model yet.
The question is open with its author. Until there is one, inventing a line here
would be exactly the mistake the rest of this toolbox exists to avoid, so the
score is reported as INFO and never decides a verdict.

Why a saturation guard
----------------------
The author's preprocessing normalises with `clip(cbf, -100, 100) / 100`. That
is right for a correctly quantified map and destructive for a mis-scaled one.
Measured on a real GE map whose calibration was ~50x too high: 74.2% of the
voxels inside the brain mask pinned to +1.0, leaving two distinct values in the
whole volume, and the network still returned 0.634 on what had become a binary
blob. The classical QEI scored the same map 0.0006 and was right to.

So the check measures the saturated fraction first and refuses to report a
score when the map is mostly pinned. A number computed from a saturated volume
is not a quality measurement, and passing it through would launder a broken
scan into a middling score.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np

from ..core.config import QCConfig
from ..core.registry import register_qc_check
from ..core.result import CheckResult, Verdict

# The author's preprocessing clips here before dividing. Mirrored, not guessed:
# see `preprocess.py`, `np.clip(cbf, -100, 100) / 100.0`.
_CLIP = 100.0

# Above this share of the brain pinned to the clip bound, the normalised volume
# carries too little structure for the score to mean anything. Uncalibrated: a
# judgement about when a measurement stops being a measurement, not a quality
# threshold, and it never decides a verdict either way.
_SATURATION_LIMIT = 0.30

_TIMEOUT_S = 900


def _configured(cfg: QCConfig) -> tuple[str | None, str | None]:
    """The interpreter and script to shell out to, or (None, None).

    Read from the environment rather than from config so that a machine-specific
    path to somebody else's unpublished model never ends up in a committed file.
    """
    py = getattr(cfg, "qei_net_python", None) or os.environ.get("OSIPY_QEI_NET_PYTHON")
    script = getattr(cfg, "qei_net_script", None) or os.environ.get("OSIPY_QEI_NET_SCRIPT")
    return (py or None), (script or None)


def _model_fingerprint(script: str) -> str:
    """A short, stable id for the weights that produced a score.

    A QEI-Net number is only interpretable next to the model that made it, and
    the model is expected to change as its paper progresses. This hashes the
    checkpoint bytes rather than reading a version string, so it cannot drift
    from what actually ran.
    """
    weights = pathlib.Path(script).resolve().parent.parent / "weights"
    if not weights.is_dir():
        return "unknown"
    h = hashlib.sha256()
    for p in sorted(weights.rglob("*.pth")):
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
        with p.open("rb") as fh:          # head and tail: full hash of 70 MB is wasteful
            h.update(fh.read(65536))
            fh.seek(-65536, os.SEEK_END)
            h.update(fh.read())
    return h.hexdigest()[:12]


def _saturated_fraction(cbf: np.ndarray, brain: np.ndarray | None) -> float:
    """Share of brain voxels the author's clip would pin to a bound."""
    arr = np.asarray(cbf, dtype=float)
    vals = arr[np.asarray(brain, dtype=bool)] if brain is not None else arr[arr != 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(np.abs(vals) >= _CLIP))


def _brain_from_tissue(gm, wm, csf) -> np.ndarray | None:
    """A brain mask from whatever tissue maps were supplied."""
    parts = [np.asarray(x, dtype=float) for x in (gm, wm, csf) if x is not None]
    if not parts:
        return None
    total = parts[0]
    for p in parts[1:]:
        if p.shape != total.shape:
            return None
        total = total + p
    return total > 0.5


@register_qc_check("1.1.qei_net", stream="B", required=False)
def qei_net_check(cbf=None, gm=None, wm=None, csf=None, cbf_path=None, affine=None,
                  cfg: QCConfig = QCConfig(), **_) -> CheckResult:
    """Score the CBF map with QEI-Net, when the model is available locally.

    Reports the number. Never grades on it, because no validated cut-off for
    this model has been published yet.
    """
    py, script = _configured(cfg)
    if not py or not script:
        # N/A rather than UNKNOWN, and the distinction is deliberate. UNKNOWN
        # means the DATA was missing something, and it dents the coverage figure
        # so a reader knows the report is partial. A model the operator chose not
        # to install says nothing about the data. Reporting it as UNKNOWN would
        # mark every report on every ordinary install permanently incomplete,
        # which would drain the meaning out of the one number carrying "some of
        # this scan could not be looked at". A configured model that then fails
        # IS an UNKNOWN, and the branches below return exactly that.
        return CheckResult(
            "1.1.qei_net", Verdict.NA,
            reason="QEI-Net not configured - set OSIPY_QEI_NET_PYTHON and "
                   "OSIPY_QEI_NET_SCRIPT to score with it (the model is not part "
                   "of this package)")
    if not os.path.exists(py) or not os.path.exists(script):
        missing = py if not os.path.exists(py) else script
        return CheckResult(
            "1.1.qei_net", Verdict.UNKNOWN,
            reason=f"QEI-Net is configured but {os.path.basename(missing)} is not "
                   "at the path given")
    if cbf is None and cbf_path is None:
        return CheckResult("1.1.qei_net", Verdict.UNKNOWN, reason="needs a CBF map")

    brain = _brain_from_tissue(gm, wm, csf)

    # Refuse before spending a minute of GPU time on a volume the normalisation
    # is about to flatten.
    if cbf is not None:
        sat = _saturated_fraction(cbf, brain)
        if np.isfinite(sat) and sat > _SATURATION_LIMIT:
            return CheckResult(
                "1.1.qei_net", Verdict.UNKNOWN,
                metric={"saturated_fraction": round(sat, 4),
                        "clip_bound": _CLIP,
                        "saturation_limit": _SATURATION_LIMIT},
                reason=f"{sat:.1%} of the brain is at or beyond the {_CLIP:.0f} "
                       "clip bound, so QEI-Net's normalisation would flatten this "
                       "map before scoring it - check the calibration first "
                       "(3.1.cbf_level)")

    with tempfile.TemporaryDirectory(prefix="osipy-qeinet-") as tmp:
        tmpdir = pathlib.Path(tmp)
        import nibabel as nib

        if cbf_path and os.path.exists(cbf_path):
            cbf_file = cbf_path                     # the original, header intact
        else:
            if affine is None:
                return CheckResult(
                    "1.1.qei_net", Verdict.UNKNOWN,
                    reason="needs either the CBF file path or its affine - the "
                           "model resamples to a fixed RAS grid and cannot do that "
                           "from a bare array")
            cbf_file = str(tmpdir / "cbf.nii.gz")
            nib.save(nib.Nifti1Image(np.asarray(cbf, dtype=np.float32), affine), cbf_file)

        cmd = [py, script, "--single", cbf_file]
        if brain is not None and affine is not None:
            mask_file = str(tmpdir / "brainmask.nii.gz")
            nib.save(nib.Nifti1Image(brain.astype(np.uint8), affine), mask_file)
            cmd += ["--mask", mask_file]
        else:
            # Only legal when the input already sits on the model's own grid;
            # the script says so itself and fails clearly when it does not.
            cmd += ["--derive_mask_from_cbf"]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return CheckResult("1.1.qei_net", Verdict.UNKNOWN,
                               reason=f"QEI-Net did not finish within {_TIMEOUT_S}s")
        except OSError as exc:
            return CheckResult("1.1.qei_net", Verdict.UNKNOWN,
                               reason=f"could not run QEI-Net: {exc}")

        if proc.returncode != 0:
            why = (proc.stderr or proc.stdout or "").strip().splitlines()
            return CheckResult(
                "1.1.qei_net", Verdict.UNKNOWN,
                reason="QEI-Net could not score this map: "
                       + (why[-1][:160] if why else f"exit {proc.returncode}"))
        try:
            score = float((proc.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return CheckResult("1.1.qei_net", Verdict.UNKNOWN,
                               reason="QEI-Net returned no score on stdout")

    mask_note = "tissue-derived brain mask" if brain is not None else "mask derived from CBF"
    return CheckResult(
        "1.1.qei_net", Verdict.INFO,
        metric={"qei_net": round(score, 4),
                "model": _model_fingerprint(script),
                "mask_source": mask_note,
                "saturated_fraction": (round(_saturated_fraction(cbf, brain), 4)
                                       if cbf is not None else None)},
        reason=f"QEI-Net {score:.3f} [model {_model_fingerprint(script)}] - reported, "
               "not graded: no validated cut-off has been published for this model")
