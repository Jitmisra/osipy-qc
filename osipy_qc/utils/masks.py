"""
Mask helpers: turn probability maps into boolean tissue masks, and a cheap
brain-mask fallback when none is supplied. Pure NumPy.
"""

from __future__ import annotations

import numpy as np


def threshold_prob(prob: np.ndarray, thresh: float = 0.7) -> np.ndarray:
    """Binarise a probability map with a STRICT greater-than, matching the live
    ASLPrep compute_qei (`probseg > thresh`)."""
    return np.asarray(prob, dtype=float) > thresh


def clean_nonfinite(volume: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace NaN/Inf with `fill` so no downstream statistic is poisoned."""
    vol = np.asarray(volume, dtype=float).copy()
    vol[~np.isfinite(vol)] = fill
    return vol


def covered_tissue_mask(cbf: np.ndarray, prob: np.ndarray, thresh: float = 0.7) -> np.ndarray:
    """Tissue mask restricted to voxels the CBF map actually covers.

    Why this exists
    ---------------
    Tissue masks are segmented on the T1 and then resampled into ASL space. The
    ASL field of view is usually smaller than the T1's — the cerebellum, in
    particular, is often not covered. Those voxels stay inside the tissue ROI but
    hold exactly 0 in the CBF map, so a plain `cbf[gm > 0.7].mean()` silently
    averages structural zeros into the perfusion mean: mean GM CBF is dragged
    down, and the GM/WM ratio is corrupted with it.

    This is not hypothetical. ASLPrep's own `average_cbf_by_tissue()`
    (aslprep/utils/confounds.py) does NOT guard against it. ExploreASL is the
    tool that does, via a coverage-aware mask.

    `coverage_fraction()` reports how much of the ROI was lost this way, so a
    badly-cropped FOV surfaces as its own finding rather than as a quietly wrong
    number.
    """
    tissue = threshold_prob(prob, thresh)
    covered = clean_nonfinite(cbf) != 0.0
    return tissue & covered


def coverage_fraction(cbf: np.ndarray, prob: np.ndarray, thresh: float = 0.7) -> float:
    """Share of the tissue ROI that the CBF map actually covers (0..1).

    1.0 = every tissue voxel has data. 0.8 = a fifth of the ROI sits outside the
    ASL FOV (or is otherwise zero) and would have been averaged in as zeros.
    Returns 0.0 for an empty ROI.
    """
    tissue = threshold_prob(prob, thresh)
    n_tissue = int(np.count_nonzero(tissue))
    if n_tissue == 0:
        return 0.0
    n_covered = int(np.count_nonzero(tissue & (clean_nonfinite(cbf) != 0.0)))
    return n_covered / n_tissue


def brain_mask_fallback(cbf: np.ndarray, percentile: float = 50.0) -> np.ndarray:
    """A crude brain mask when none is provided: voxels whose magnitude exceeds
    a percentile of the non-zero magnitudes. Good enough for QC-grade stats;
    flagged as 'rough' by the caller."""
    mag = np.abs(np.asarray(cbf, dtype=float))
    nz = mag[mag > 0]
    if nz.size == 0:
        return np.zeros(mag.shape, dtype=bool)
    return mag > np.percentile(nz, percentile)


def check_prob_range(prob, name: str = "tissue map") -> tuple[bool, float]:
    """Is this actually a probability map, or a 0-255 one wearing the name?

    SPM, ANTs and DICOM-converted segmentations are commonly written 0-255 or
    0-100. Every mask in this package thresholds with `prob > 0.7`, which on a
    0-255 map selects every voxel holding more than 0.3% tissue — so grey
    matter, white matter and CSF all become the whole brain and the score is
    quietly wrong rather than obviously broken.

    Returns (looks_like_probability, observed_max).
    """
    arr = np.asarray(prob, dtype=float)
    finite = arr[np.isfinite(arr)]
    pmax = float(finite.max()) if finite.size else 0.0
    return pmax <= 1.001, pmax
