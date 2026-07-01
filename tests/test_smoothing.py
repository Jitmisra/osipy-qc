"""Known-answer tests for the pure-NumPy Gaussian smoother."""

import numpy as np

from osipy_qc.utils.smoothing import gaussian_kernel_1d, smooth_fwhm


def test_kernel_normalised_and_symmetric():
    k = gaussian_kernel_1d(2.0)
    assert abs(k.sum() - 1.0) < 1e-12          # normalised
    assert np.allclose(k, k[::-1])             # symmetric
    assert k.argmax() == len(k) // 2           # peak in the centre


def test_fwhm_to_sigma_peak_relation():
    # For a unit impulse far from the boundary, the smoothed peak equals the
    # product of the three central kernel weights (separable Gaussian).
    vol = np.zeros((21, 21, 21))
    vol[10, 10, 10] = 1.0
    out = smooth_fwhm(vol, fwhm_mm=5.0, voxel_mm=(2.0, 2.0, 2.0))
    sigma_vox = (5.0 / 2.3548200450309493) / 2.0
    k0 = gaussian_kernel_1d(sigma_vox)[len(gaussian_kernel_1d(sigma_vox)) // 2]
    assert abs(out[10, 10, 10] - k0 ** 3) < 1e-6


def test_smoothing_conserves_signal_interior():
    # An impulse well inside the volume: total signal is conserved.
    vol = np.zeros((25, 25, 25))
    vol[12, 12, 12] = 100.0
    out = smooth_fwhm(vol, fwhm_mm=5.0, voxel_mm=(3.0, 3.0, 3.0))
    assert abs(out.sum() - 100.0) < 1e-6


def test_smoothing_reduces_peak_and_spreads():
    vol = np.zeros((25, 25, 25))
    vol[12, 12, 12] = 100.0
    out = smooth_fwhm(vol, fwhm_mm=6.0, voxel_mm=(3.0, 3.0, 3.0))
    assert out[12, 12, 12] < 100.0             # peak reduced
    assert out[11, 12, 12] > 0.0               # spread to neighbours
