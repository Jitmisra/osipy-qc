"""
Pure-NumPy math helpers shared across checks. NO scipy (osipy portability rule).

Each function is small and independently testable, with the formula in the
docstring so it can be checked against the source papers by hand.
"""

from __future__ import annotations

import numpy as np


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two flattened arrays.

        r = sum((a-ma)(b-mb)) / sqrt(sum((a-ma)^2) * sum((b-mb)^2))

    Returns 0.0 if either array is constant (zero variance).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return 0.0
    return float(np.sum(a * b) / denom)


def geometric_mean(values) -> float:
    """Geometric mean = (prod v_i)^(1/n). Used to combine the QEI components so
    one near-zero component collapses the whole score.

    Defined only for non-negative inputs (the QEI factors are always in [0,1]).
    A zero OR negative component collapses the result to 0.0 — we never take a
    root of a possibly-negative product (which would leak NaN to callers)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0
    if np.any(v <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(v))))


def pooled_within_tissue_variance(arrays: list[np.ndarray]) -> float:
    """Dispersion numerator/denominator, byte-faithful to ASLPrep `dispersion_index`:

        V = sum_k (n_k - 1) * var_pop(tissue_k) / (sum_k n_k - K)

    where var_pop is the POPULATION variance (np.var, ddof=0) — ASLPrep applies
    the (n_k - 1) weighting *outside* np.var, not as a Bessel correction inside it.
    (Verified against reference_docs/aslprep/SOURCE_confounds_QEI.py lines 350-353.)

    This is numerically IDENTICAL to ASLPrep when all three tissues are non-empty
    (its denominator n_total-3 == sum(n_k-1)). It deliberately DEGRADES more
    gracefully than ASLPrep on the degenerate empty-tissue case: an empty tissue is
    skipped here, whereas ASLPrep would compute np.var([])=NaN and raise.
    """
    num = 0.0
    den = 0.0
    for vals in arrays:
        vals = np.asarray(vals, dtype=float).ravel()
        n = vals.size
        if n < 1:
            continue
        num += (n - 1) * np.var(vals)   # ddof=0, matches ASLPrep exactly
        den += (n - 1)
    return float(num / den) if den > 0 else 0.0


def skewness(values: np.ndarray) -> float:
    """Population skewness = third standardized moment:

        g1 = mean((x - mu)^3) / sigma^3

    Negative -> left-tailed (long tail toward low/negative values), which for a
    CBF distribution hints at noise / failed labeling.
    """
    x = np.asarray(values, dtype=float).ravel()
    if x.size < 2:
        return 0.0
    mu = x.mean()
    sigma = x.std()  # population std (ddof=0)
    if sigma == 0:
        return 0.0
    return float(np.mean((x - mu) ** 3) / sigma ** 3)


def fraction(condition_count: int, total: int) -> float:
    """Safe fraction; returns 0.0 when total is 0."""
    return float(condition_count) / total if total else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two boolean masks: 2|A∩B| / (|A| + |B|)."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.count_nonzero(a & b))
    size = int(np.count_nonzero(a) + np.count_nonzero(b))
    return (2.0 * inter / size) if size else 0.0


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard overlap of two boolean masks: |A∩B| / |A∪B|.
    (Monotonically related to Dice: J = D / (2 - D).)"""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return (inter / union) if union else 0.0
