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


def brain_mask_fallback(cbf: np.ndarray, percentile: float = 50.0) -> np.ndarray:
    """A crude brain mask when none is provided: voxels whose magnitude exceeds
    a percentile of the non-zero magnitudes. Good enough for QC-grade stats;
    flagged as 'rough' by the caller."""
    mag = np.abs(np.asarray(cbf, dtype=float))
    nz = mag[mag > 0]
    if nz.size == 0:
        return np.zeros(mag.shape, dtype=bool)
    return mag > np.percentile(nz, percentile)
