"""Known-answer tests for the pure-NumPy math helpers."""

import numpy as np

from osipy_qc.utils.mathops import (
    dice,
    fraction,
    geometric_mean,
    jaccard,
    pearson,
    pooled_within_tissue_variance,
    skewness,
)


def test_pearson_perfect_positive():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson(a, 2 * a + 5) == 1.0          # affine -> r = 1


def test_pearson_perfect_negative():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson(a, -a) == -1.0


def test_pearson_constant_is_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert pearson(a, np.array([5.0, 5.0, 5.0])) == 0.0


def test_geometric_mean_known():
    # (1*2*4)^(1/3) = 8^(1/3) = 2
    assert abs(geometric_mean([1.0, 2.0, 4.0]) - 2.0) < 1e-9


def test_geometric_mean_collapses_on_zero():
    assert geometric_mean([0.0, 1.0, 1.0]) == 0.0


def test_pooled_variance_two_groups():
    # ASLPrep-faithful: V = sum (n_k-1)*var_pop_k / sum (n_k-1), var_pop ddof=0.
    # var_pop([1,2,3]) = 2/3; (n-1)=2 each; V = (2*2/3 + 2*2/3) / (2+2) = 2/3.
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert abs(pooled_within_tissue_variance([a, b]) - 2.0 / 3.0) < 1e-9


def test_skewness_symmetric_is_zero():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert abs(skewness(x)) < 1e-9


def test_skewness_left_tail_negative():
    x = np.array([-10.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # long tail toward negatives
    assert skewness(x) < 0


def test_dice_and_jaccard_known():
    a = np.zeros((4, 4, 4), dtype=bool)
    b = np.zeros((4, 4, 4), dtype=bool)
    a[:2] = True            # 32 voxels
    b[1:3] = True           # 32 voxels, overlap on slice index 1 -> 16 voxels
    # Dice = 2*16/(32+32) = 0.5 ; Jaccard = 16/(32+32-16) = 16/48 = 1/3
    assert abs(dice(a, b) - 0.5) < 1e-9
    assert abs(jaccard(a, b) - 1.0 / 3.0) < 1e-9
    # relationship J = D/(2-D)
    d = dice(a, b)
    assert abs(jaccard(a, b) - d / (2 - d)) < 1e-9


def test_fraction_safe():
    assert fraction(3, 12) == 0.25
    assert fraction(1, 0) == 0.0
