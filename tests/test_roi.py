"""Known-answer tests for the ROI/mask toolkit shared by kidney and placenta.

Every expected value here is computed by hand in the test, not copied from a
previous run of the code.
"""

import numpy as np
import pytest

from osipy_qc.core import Verdict
from osipy_qc.utils.roi import (as_mask, as_sides, asymmetry_index, box_mean,
                                component_sizes, connected_components, cov,
                                largest_component_fraction, local_ssim,
                                roi_fraction, roi_stats, roi_values,
                                touches_fov_edge, worst)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def test_roi_stats_hand_computed():
    v = np.arange(27, dtype=float).reshape(3, 3, 3)
    m = np.zeros((3, 3, 3), bool)
    m[0, 0, :] = True                        # voxels 0, 1, 2
    s = roi_stats(v, m)
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(1.0)
    assert s["median"] == pytest.approx(1.0)
    assert s["std"] == pytest.approx(np.sqrt(2 / 3))     # population SD (ddof=0)
    assert (s["min"], s["max"]) == (0.0, 2.0)


def test_roi_stats_ignores_nonfinite_and_reports_the_count_used():
    v = np.arange(27, dtype=float).reshape(3, 3, 3)
    v[0, 0, 1] = np.nan
    m = np.zeros((3, 3, 3), bool)
    m[0, 0, :] = True
    s = roi_stats(v, m)
    assert s["n"] == 2                        # not 3 - the caller must see this
    assert s["mean"] == pytest.approx(1.0)    # (0 + 2) / 2


def test_empty_roi_is_nan_not_zero():
    """'No voxels' and 'zero perfusion' are different findings, and only one of
    them is a pass."""
    v = np.ones((4, 4, 4))
    s = roi_stats(v, np.zeros((4, 4, 4), bool))
    assert s["n"] == 0
    assert np.isnan(s["mean"])
    assert np.isnan(roi_fraction(v, np.zeros((4, 4, 4), bool), below=0))


def test_roi_stats_rejects_a_mask_on_the_wrong_grid():
    with pytest.raises(ValueError, match="resample"):
        roi_stats(np.ones((4, 4, 4)), np.ones((8, 8, 8), bool))


def test_cov_of_a_nonpositive_mean_is_nan():
    """A CoV about a negative mean would read as 'very heterogeneous' when the
    real finding is 'the sign is wrong'."""
    v = -np.ones((4, 4, 4))
    assert np.isnan(cov(v, np.ones((4, 4, 4), bool)))


def test_roi_fraction_counts_both_tails():
    v = np.array([-1.0, 0.5, 2.0, 900.0]).reshape(4, 1, 1)
    m = np.ones((4, 1, 1), bool)
    assert roi_fraction(v, m, below=0) == pytest.approx(0.25)
    assert roi_fraction(v, m, above=500) == pytest.approx(0.25)
    assert roi_fraction(v, m, below=0, above=500) == pytest.approx(0.5)


def test_as_mask_thresholds_a_probability_map_strictly():
    assert as_mask(np.array([0.4, 0.5, 0.6])).tolist() == [False, False, True]
    b = np.array([True, False])
    assert as_mask(b) is b                    # a boolean mask passes through


def test_roi_values_returns_only_finite_in_roi():
    v = np.array([1.0, np.nan, 3.0]).reshape(3, 1, 1)
    assert roi_values(v, np.ones((3, 1, 1), bool)).tolist() == [1.0, 3.0]


# --------------------------------------------------------------------------- #
# sides
# --------------------------------------------------------------------------- #
def test_as_sides_never_duplicates_one_side_across_both():
    """Reporting one kidney's number twice would fabricate an agreement between
    the two that was never measured."""
    single = np.ones((2, 2, 2))
    out = as_sides(single)
    assert list(out) == ["single"]
    assert as_sides({"left": single, "right": None}) == {"left": single}
    assert as_sides(None) == {}


def test_worst_picks_the_most_severe():
    assert worst([Verdict.PASS, Verdict.WARN, Verdict.FAIL]) is Verdict.FAIL
    assert worst([Verdict.PASS, Verdict.WARN]) is Verdict.WARN
    assert worst([Verdict.NA, Verdict.UNKNOWN]) is Verdict.UNKNOWN
    assert worst([]) is Verdict.UNKNOWN


def test_asymmetry_index_is_symmetric_and_scale_free():
    assert asymmetry_index(100, 120) == pytest.approx(20 / 110 * 100)
    assert asymmetry_index(120, 100) == asymmetry_index(100, 120)
    # scale-free: doubling both sides leaves it unchanged
    assert asymmetry_index(200, 240) == pytest.approx(asymmetry_index(100, 120))
    assert np.isnan(asymmetry_index(0, 0))
    assert np.isnan(asymmetry_index(np.nan, 1))


# --------------------------------------------------------------------------- #
# mask geometry
# --------------------------------------------------------------------------- #
def test_connected_components_is_strictly_6_connected():
    """The first implementation rolled the progressively-updated array, so a
    value moved along axis 0 and then axis 1 within one sweep - silently making
    the labelling 26-connected. Two voxels touching only at a corner were then
    reported as one object."""
    corner = np.zeros((5, 5, 5), bool)
    corner[1, 1, 1] = corner[2, 2, 2] = True
    assert component_sizes(corner) == [1, 1]

    edge = np.zeros((5, 5, 5), bool)          # touching along two axes
    edge[1, 1, 1] = edge[1, 2, 2] = True
    assert component_sizes(edge) == [1, 1]

    face = np.zeros((5, 5, 5), bool)          # sharing a face - connected
    face[1, 1, 1] = face[1, 1, 2] = True
    assert component_sizes(face) == [2]


def test_connected_components_does_not_wrap_around():
    """np.roll wraps; opposite faces of the volume must not merge."""
    w = np.zeros((6, 6, 6), bool)
    w[0, 2, 2] = w[-1, 2, 2] = True
    assert component_sizes(w) == [1, 1]


def test_connected_components_propagates_along_an_L():
    """More sweeps than one are needed for a long thin object."""
    L = np.zeros((12, 12, 12), bool)
    L[2, 2:10, 2] = True                       # 8 voxels
    L[2, 9, 2:10] = True                       # 8 more, sharing one corner voxel
    assert component_sizes(L) == [15]


def test_largest_component_fraction_and_empty_mask():
    m = np.zeros((10, 10, 10), bool)
    m[1:4, 1:4, 1:4] = True                    # 27
    m[7:9, 7:9, 7:9] = True                    # 8
    assert component_sizes(m) == [27, 8]
    assert largest_component_fraction(m) == pytest.approx(27 / 35)
    assert np.isnan(largest_component_fraction(np.zeros((4, 4, 4), bool)))


def test_touches_fov_edge():
    inside = np.zeros((6, 6, 6), bool)
    inside[2:4, 2:4, 2:4] = True
    assert not touches_fov_edge(inside)
    for idx in ((0, 3, 3), (-1, 3, 3), (3, 0, 3), (3, -1, 3), (3, 3, 0), (3, 3, -1)):
        m = inside.copy()
        m[idx] = True
        assert touches_fov_edge(m), f"face {idx} not detected"


# --------------------------------------------------------------------------- #
# box_mean / local_ssim (P6.3's local structural test, scipy-free)
# --------------------------------------------------------------------------- #
def test_box_mean_edges_and_degenerate_radii():
    a = np.arange(27, dtype=float).reshape(3, 3, 3)
    # a box wider than the volume is the global mean everywhere
    assert np.allclose(box_mean(a, 5), a.mean())
    # radius 0 is the identity
    assert np.allclose(box_mean(a, 0), a)
    # edge voxels average over the part of the box that EXISTS, not over zeros
    b = np.zeros((5, 5, 5))
    b[0, 0, 0] = 8.0
    # the corner's 3x3x3 box has 2*2*2 = 8 real voxels, so the mean is 1.0
    assert box_mean(b, 1)[0, 0, 0] == pytest.approx(1.0)


def test_local_ssim_is_one_for_identical_volumes_and_drops_with_noise():
    x = np.random.default_rng(0).normal(10, 2, (8, 8, 8))
    assert np.nanmedian(local_ssim(x, x)) == pytest.approx(1.0, abs=1e-6)
    y = x + np.random.default_rng(1).normal(0, 4, (8, 8, 8))
    assert np.nanmedian(local_ssim(x, y)) < 0.8


def test_local_ssim_sees_a_local_change_that_global_correlation_misses():
    """The reason the design asks for BOTH: a correlation over the whole ROI
    stays high while one region is badly deformed."""
    from osipy_qc.utils.mathops import pearson
    rng = np.random.default_rng(2)
    a = rng.normal(10, 3, (20, 20, 20))
    b = a.copy()
    b[2:5, 2:5, 2:5] = 20.0                      # one locally destroyed block
    # 27 voxels in 8000: the global correlation is essentially untouched...
    assert pearson(a.ravel(), b.ravel()) > 0.95
    # ...while the local measure drops far below the 0.6 line the design uses
    assert np.nanmin(local_ssim(a, b)) < 0.5
