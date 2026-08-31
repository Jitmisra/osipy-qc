"""
ROI statistics for organs that are graded inside a supplied mask.

The brain checks lean on tissue PROBABILITY maps (GM/WM/CSF) and can fall back
to a percentile brain mask when none is given. Kidney and placenta cannot: both
designs make the mask a declared input with recorded provenance, because there
is no equivalent of "just run BET on it" for either organ, and a mask this code
invented would be a mask nobody could check.

So everything here is mask-first and NaN-safe:

* masks arrive as boolean arrays OR as probability maps (a manual segmentation
  is often 0/1 floats, an automatic one is often continuous) - `as_mask`
  normalises both without pretending a probability is a certainty;
* every statistic ignores non-finite voxels rather than propagating them, since
  real perfusion maps carry NaN outside the organ and a single NaN would
  otherwise make an entire kidney unmeasurable;
* a statistic over an empty ROI returns NaN and a count of 0, never 0.0 - a mean
  of "no voxels" is not zero perfusion, and the difference decides UNKNOWN vs a
  verdict.

Sides
-----
The renal consensus (Nery 2020 R10.1, 100% agreement) requires reporting left
and right kidneys SEPARATELY, so per-side inputs are a first-class shape here
rather than something each check unpacks by hand.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.result import Verdict

SIDES: tuple[str, str] = ("left", "right")


def as_mask(mask: Any, thresh: float = 0.5) -> np.ndarray:
    """Normalise a mask input to a boolean array.

    Accepts a boolean array, a 0/1 integer array, or a probability map. A
    probability map is thresholded at `thresh`; the default 0.5 is the ordinary
    "more likely inside than outside" line and is recorded as an IMPLEMENTATION
    choice, not a published one.
    """
    arr = np.asarray(mask)
    # A mask written as (X, Y, Z, 1) is a 3-D mask. NIfTI permits the shape and
    # segmentation tools emit it; rejecting it made five kidney checks report
    # "check error: operands could not be broadcast" about an ordinary file.
    while arr.ndim == 4 and arr.shape[3] == 1:
        arr = arr[..., 0]
    if arr.dtype == bool:
        return arr
    arr = arr.astype(float)
    finite = np.isfinite(arr)
    # An all-or-nothing map (only 0s and 1s) is already a mask; thresholding it
    # at 0.5 gives the same answer, so this branch exists only for clarity.
    return finite & (arr > thresh)


def roi_stats(volume: Any, mask: Any = None, thresh: float = 0.5) -> dict[str, float]:
    """NaN-safe descriptive statistics of `volume` inside `mask`.

    Returns mean/median/std/min/max/n. `n` is the number of FINITE voxels that
    were actually used, which is what a caller needs to decide whether a number
    is trustworthy - a mask covering 30000 voxels of which 4 are finite should
    not yield a confident mean.
    """
    vol = np.asarray(volume, dtype=float)
    if mask is None:
        sel = np.isfinite(vol)
    else:
        m = as_mask(mask, thresh)
        if m.shape != vol.shape:
            raise ValueError(
                f"mask shape {m.shape} != volume shape {vol.shape} - resample the "
                "mask into the image grid before grading")
        sel = m & np.isfinite(vol)
    vals = vol[sel]
    n = int(vals.size)
    if n == 0:
        nan = float("nan")
        return {"mean": nan, "median": nan, "std": nan, "min": nan, "max": nan, "n": 0}
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        # population std (ddof=0), matching the rest of the package
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "n": n,
    }


def roi_values(volume: Any, mask: Any = None, thresh: float = 0.5) -> np.ndarray:
    """The finite voxel values of `volume` inside `mask`, as a 1-D array."""
    vol = np.asarray(volume, dtype=float)
    sel = np.isfinite(vol) if mask is None else (as_mask(mask, thresh) & np.isfinite(vol))
    return vol[sel]


def roi_fraction(volume: Any, mask: Any = None, *, below: float | None = None,
                 above: float | None = None, thresh: float = 0.5) -> float:
    """Fraction of finite in-ROI voxels that are `< below` or `> above`.

    Returns NaN for an empty ROI rather than 0.0: "no voxels were implausible"
    and "there were no voxels" are different findings, and only one of them is
    a pass.
    """
    vals = roi_values(volume, mask, thresh)
    if vals.size == 0:
        return float("nan")
    bad = np.zeros(vals.shape, dtype=bool)
    if below is not None:
        bad |= vals < below
    if above is not None:
        bad |= vals > above
    return float(np.count_nonzero(bad) / vals.size)


def cov(volume: Any, mask: Any = None, thresh: float = 0.5) -> float:
    """Coefficient of variation (std/mean) inside the ROI.

    NaN when the ROI is empty or its mean is <= 0: a CoV about a non-positive
    mean is not a dispersion measure, it is a sign artefact, and returning a
    large number there would read as "very heterogeneous" instead of "invalid".
    """
    s = roi_stats(volume, mask, thresh)
    if s["n"] == 0 or not np.isfinite(s["mean"]) or s["mean"] <= 0:
        return float("nan")
    return float(s["std"] / s["mean"])


def as_sides(value: Any) -> dict[str, Any]:
    """Normalise a per-side input to {"left": ..., "right": ...}.

    Accepts a dict (any subset of the two sides), or a single array meaning "one
    unlabelled side". A single array is deliberately NOT duplicated across both
    sides: the consensus asks for the two kidneys separately, and silently
    reporting one kidney's number twice would fabricate an agreement between
    them that was never measured.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if v is not None}
    return {"single": value}


def worst(verdicts) -> Verdict:
    """The most severe verdict in an iterable, by FAIL > WARN > PASS.

    Used where a check grades several ROIs (two kidneys) and must return one
    verdict: the design says take the WORSE side and name it in the reason, so a
    failing right kidney is never averaged away by a healthy left one.
    Absence verdicts (UNKNOWN/N/A/INFO) are returned only if nothing was graded.
    """
    vs = list(verdicts)
    for level in (Verdict.FAIL, Verdict.WARN, Verdict.PASS):
        if level in vs:
            return level
    for level in (Verdict.UNKNOWN, Verdict.INFO, Verdict.NA):
        if level in vs:
            return level
    return Verdict.UNKNOWN


def asymmetry_index(a: float, b: float) -> float:
    """Symmetric percentage difference between two sides: |a-b| / mean(a,b) * 100.

    Symmetric by construction so that swapping left and right cannot change the
    answer, and normalised by the mean so it is scale-free - which matters when
    the absolute level is itself uncalibrated.
    """
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    denom = (abs(a) + abs(b)) / 2.0
    if denom == 0:
        return float("nan")
    return float(abs(a - b) / denom * 100.0)


# --------------------------------------------------------------------------- #
# Mask geometry
# --------------------------------------------------------------------------- #
def connected_components(mask: Any) -> np.ndarray:
    """Label 6-connected components of a 3-D boolean mask. Pure NumPy.

    scipy.ndimage.label would be one line, but scipy is not a dependency (GPU
    portability - the package uses only NumPy so `xp = get_array_module()` stays
    possible). A Python flood-fill would also work and be O(n), but it is a
    per-voxel interpreted loop; this instead seeds every voxel with a unique id
    and propagates the maximum through the 6-neighbourhood until nothing
    changes. Each sweep is vectorised, and the number of sweeps is the diameter
    of the largest component, not the number of voxels.

    Returns an int array: 0 outside the mask, and a positive id per component
    (ids are arbitrary and not contiguous).
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"connected_components expects a 3-D mask, got {m.shape}")
    lab = np.zeros(m.shape, dtype=np.int64)
    n = int(m.sum())
    if n == 0:
        return lab
    lab[m] = np.arange(1, n + 1, dtype=np.int64)

    while True:
        # Every neighbour contribution is taken from `lab` (this sweep's fixed
        # starting state), never from the partially-updated result. Rolling the
        # progressively-updated array instead lets a value move along axis 0 and
        # then again along axis 1 within ONE sweep, which propagates diagonally -
        # the labelling silently became 26-connected, and two voxels touching
        # only at a corner were reported as one object.
        nxt = lab
        for axis in range(3):
            for shift in (1, -1):
                rolled = np.roll(lab, shift, axis=axis)
                # np.roll wraps around; a face voxel must not be neighboured by
                # the far face, which would merge two genuinely separate objects
                idx: list[Any] = [slice(None)] * 3
                idx[axis] = 0 if shift == 1 else -1
                rolled = rolled.copy()
                rolled[tuple(idx)] = 0
                nxt = np.maximum(nxt, rolled)
        nxt = nxt * m
        if np.array_equal(nxt, lab):
            return nxt
        lab = nxt


def component_sizes(mask: Any) -> list[int]:
    """Voxel counts of each connected component, largest first."""
    lab = connected_components(mask)
    ids = lab[lab > 0]
    if ids.size == 0:
        return []
    _, counts = np.unique(ids, return_counts=True)
    return sorted((int(c) for c in counts), reverse=True)


def largest_component_fraction(mask: Any) -> float:
    """Share of mask voxels in its single largest connected component.

    1.0 means one clean object. Below 1.0 means the mask has islands - stray
    voxels, a second organ caught in the same label, or a segmentation that
    broke apart. NaN for an empty mask, since "the largest piece of nothing" is
    not 0% or 100%.
    """
    sizes = component_sizes(mask)
    if not sizes:
        return float("nan")
    return float(sizes[0] / sum(sizes))


def touches_fov_edge(mask: Any) -> bool:
    """True if any mask voxel lies on one of the six faces of the volume.

    An organ touching the edge of the field of view is probably cut off by it,
    and a mean computed over a clipped organ is a mean over whatever happened to
    fit in the box.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3 or not m.any():
        return False
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
                or m[:, :, 0].any() or m[:, :, -1].any())


def box_mean(volume: Any, radius: int) -> np.ndarray:
    """Local mean over a (2*radius+1)^3 box, in pure NumPy.

    Implemented with cumulative sums, so the cost does not grow with the kernel
    size. Edge voxels average over the part of the box that exists rather than
    over zero-padding - the count is accumulated the same way as the sum, so no
    voxel is silently pulled toward zero at the border.
    """
    a = np.asarray(volume, dtype=float)
    if a.ndim != 3:
        raise ValueError(f"box_mean expects a 3-D volume, got {a.shape}")
    if radius < 1:
        return a.copy()

    def _running(x):
        out = x
        for axis in range(3):
            c = np.cumsum(out, axis=axis)
            # pad a leading zero so a window starting at 0 is representable
            zeros = np.zeros_like(np.take(c, [0], axis=axis))
            c = np.concatenate([zeros, c], axis=axis)
            n = out.shape[axis]
            hi = np.minimum(np.arange(n) + radius + 1, n)
            lo = np.maximum(np.arange(n) - radius, 0)
            out = np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)
        return out

    total = _running(a)
    count = _running(np.ones_like(a))
    return total / count


def local_ssim(a: Any, b: Any, radius: int = 3) -> np.ndarray:
    """Local structural similarity map between two volumes, pure NumPy.

    Uses a box kernel via `box_mean`, matching the design's "local means and
    variances from cumulative sums, no scipy". The stabilising constants follow
    the usual C1 = (0.01 L)^2, C2 = (0.03 L)^2 with L the dynamic range of the
    two volumes together, so the map is scale-free and works on perfusion values
    in any units.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"local_ssim needs matching shapes, got {x.shape} and {y.shape}")
    finite = np.isfinite(x) & np.isfinite(y)
    xs = np.where(finite, x, 0.0)
    ys = np.where(finite, y, 0.0)
    both = np.concatenate([x[finite].ravel(), y[finite].ravel()]) if finite.any() else np.array([0.0])
    rng = float(both.max() - both.min()) or 1.0
    c1, c2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2

    mx, my = box_mean(xs, radius), box_mean(ys, radius)
    vx = np.maximum(box_mean(xs * xs, radius) - mx * mx, 0.0)
    vy = np.maximum(box_mean(ys * ys, radius) - my * my, 0.0)
    cxy = box_mean(xs * ys, radius) - mx * my
    ssim = (((2 * mx * my + c1) * (2 * cxy + c2))
            / ((mx * mx + my * my + c1) * (vx + vy + c2)))
    return np.where(finite, ssim, np.nan)
