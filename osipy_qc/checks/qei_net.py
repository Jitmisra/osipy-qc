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

Why an emptiness guard as well
------------------------------
The mirror image of the same problem. The network answers whatever it is asked,
including when it is asked about nothing, and the answer looks like every other
answer. Measured against this model:

    an all-zero volume                    -> 0.109
    a flat constant volume (every voxel 5) -> 0.242
    a 1%-sparse volume                     -> 0.019

Not zero, not an error, and not even ordered by how much signal is present. Two
of this project's own oxford_asl outputs came out 98% and 90% empty inside the
grey and white matter and scored 0.259 and 0.104 - numbers a reader would take
for a poor-but-real quality estimate rather than for the absence of one.

So the check also measures how much of the TISSUE ROI carries data, and refuses
below half. Two details of that sentence are load-bearing, and both were got
wrong first:

  * the ROI is GM|WM at `cfg.tissue_thresh`, the same denominator 4.2.coverage
    uses, because the refusal tells the reader to go and look at that check. A
    (gm+wm+csf) mask answered a different question - on a good real map whose
    CBF had been tissue-masked it read 55% while 4.2.coverage read 99.6% and
    PASSed, one point from refusing a sound scan for not perfusing CSF.
  * "carries data" is a magnitude floor, not `!= 0`. A pipeline padding with
    1e-9 instead of an exact zero defeated the first version outright: both
    near-empty maps passed and scored exactly as before.

Emptiness is NOT badness: a genuinely terrible map still has signal everywhere
(the synthetic "garbage" case covers 100% of its ROI) and is still scored. This
guard fires only when there is nothing to look at.
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

# Below this share of the TISSUE ROI carrying data, there is not enough map left
# to score. Uncalibrated, like the saturation limit: a judgement about when a
# measurement stops being a measurement, and it never decides a verdict either.
#
# Set from the gap in the measured data rather than picked round. Measured with
# `_covered_fraction` as it ships, on GM|WM at the default tissue threshold:
#
#     a good real GE map                        0.997
#     the same map tissue-masked by a pipeline  0.997
#     synthetic clean / borderline / garbage    1.000
#     ------------------------------------------------
#     oxford_asl output that came out empty     0.104
#     another that came out emptier             0.020
#     an all-zero or uniformly-1e-9 volume      0.000
#
# Half sits in the middle of that gap, with room on both sides. An earlier
# version of this constant was justified against a (gm+wm+csf) mask instead, and
# those numbers ran 0.853 down to 0.551 for the same good map - a single point
# of margin, because CSF carries no perfusion and was being counted as missing.
_MIN_COVERAGE = 0.50

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


#: A voxel below this magnitude is not a measurement in any unit a CBF map is
#: written in - mL/100g/min, %M0, or an arbitrary scale normalised near 1. It
#: exists because a purely relative floor cannot see a map that is uniformly
#: tiny: scale it by its own 99th percentile and a volume of 1e-9 looks exactly
#: like a volume of 45.
_ABSOLUTE_FLOOR = 1e-6


def _data_floor(vals: np.ndarray) -> float:
    """Below this magnitude a voxel is padding, not perfusion.

    Relative to the map's own robust scale, because the units are not declared
    for brain and a fixed physical floor would be wrong for a map in %M0 or in
    arbitrary units. Floored absolutely, because a relative test alone is blind
    to a uniformly tiny volume.
    """
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0
    scale = float(np.percentile(np.abs(finite), 99))
    return max(_ABSOLUTE_FLOOR, 1e-4 * scale)


def _covered_fraction(cbf, gm, wm, cfg: QCConfig) -> float:
    """Share of the TISSUE ROI that carries an actual measurement.

    The ROI is GM|WM at `cfg.tissue_thresh` - deliberately the same denominator
    `4.2.coverage` uses, because this guard's refusal tells the reader to go and
    look at that check. It first measured (gm+wm+csf) > 0.5 instead, and the two
    answered different questions: on a good real GE map whose CBF had been
    tissue-masked by its pipeline, this said 55% while 4.2.coverage said 99.6%
    and PASSed. 98% of the voxels it counted as missing were CSF, where
    perfusion is legitimately absent - so the guard was one point from refusing
    a perfectly good map for not measuring blood flow in cerebrospinal fluid,
    and would have sent the reader to a check reporting that nothing was wrong.

    NaN when there is no tissue ROI to measure against, and the caller then skips
    the guard: 4.2.coverage is silent in that case too, so there would be nothing
    to point at.
    """
    from ..utils.masks import clean_nonfinite, threshold_prob

    parts = [x for x in (gm, wm) if x is not None]
    if not parts:
        return float("nan")
    arr = clean_nonfinite(np.asarray(cbf, dtype=float))
    roi = np.zeros(arr.shape, dtype=bool)
    for prob in parts:
        prob = np.asarray(prob, dtype=float)
        if prob.shape != arr.shape:
            return float("nan")
        roi |= threshold_prob(prob, getattr(cfg, "tissue_thresh", 0.7))
    if not roi.any():
        return float("nan")
    vals = arr[roi]
    # `!= 0` was not enough. A pipeline that pads with 1e-9 rather than an exact
    # zero defeated it outright: the two near-empty maps this guard was written
    # for passed it and scored exactly as before.
    return float(np.mean(np.abs(vals) > _data_floor(vals)))


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

        # ...and refuse the opposite failure too: a map with nothing in it.
        covered = _covered_fraction(cbf, gm, wm, cfg)
        if np.isfinite(covered) and covered < _MIN_COVERAGE:
            return CheckResult(
                "1.1.qei_net", Verdict.UNKNOWN,
                metric={"covered_fraction": round(covered, 4),
                        "coverage_limit": _MIN_COVERAGE},
                reason=f"only {covered:.1%} of the grey and white matter carries "
                       "data, so there is not enough map here for QEI-Net to score - "
                       "check the quantification and the coverage first (4.2.coverage)")

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
