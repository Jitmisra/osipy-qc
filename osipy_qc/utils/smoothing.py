"""
Pure-NumPy Gaussian smoothing at a target FWHM in millimetres. NO scipy.

The QEI needs a 5 mm FWHM-smoothed CBF map (its constants were fitted on smoothed
data). A Gaussian is separable, so we convolve a 1-D Gaussian kernel along each
axis in turn — far cheaper than a 3-D kernel and exactly equivalent.

    FWHM = 2*sqrt(2*ln2) * sigma   ->   sigma = FWHM / 2.3548
sigma is then converted from mm to voxels per axis using the voxel size.
"""

from __future__ import annotations

import numpy as np

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ~0.42466


def gaussian_kernel_1d(sigma_vox: float, truncate: float = 4.0) -> np.ndarray:
    """A normalised 1-D Gaussian kernel with radius ceil(truncate*sigma)."""
    if sigma_vox <= 0:
        return np.array([1.0])
    radius = int(np.ceil(truncate * sigma_vox))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x ** 2) / (2.0 * sigma_vox ** 2))
    k /= k.sum()
    return k


def smooth_fwhm(volume: np.ndarray, fwhm_mm: float = 5.0,
                voxel_mm: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Smooth a 3-D volume with a Gaussian of the given FWHM (mm).

    Convolution is `mode="same"`, so the array shape is preserved and total
    signal is conserved in the interior (boundary voxels lose a little mass,
    as with any finite-support smoother).
    """
    vol = np.asarray(volume, dtype=float)
    if vol.ndim != 3:
        raise ValueError(f"smooth_fwhm expects a 3-D volume, got shape {vol.shape}")
    sigma_mm = fwhm_mm * _FWHM_TO_SIGMA
    out = vol
    for axis in range(3):
        n = out.shape[axis]
        sigma_vox = sigma_mm / float(voxel_mm[axis])
        if sigma_vox <= 0:
            continue
        # cap the radius so the kernel never exceeds the axis length, otherwise
        # np.convolve(mode="same") would return the kernel length and reshape the volume.
        radius = min(int(np.ceil(4.0 * sigma_vox)), max((n - 1) // 2, 0))
        if radius < 1:
            continue
        x = np.arange(-radius, radius + 1, dtype=float)
        k = np.exp(-(x ** 2) / (2.0 * sigma_vox ** 2))
        k /= k.sum()
        out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis, out)
    return out
