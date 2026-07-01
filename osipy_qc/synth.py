"""
Synthetic ASL data with KNOWN quality, for known-answer tests and demos.

The whole point: we control the inputs, so we know roughly what the QEI / CoV /
verdicts *should* be. A "clean" case correlates strongly with the tissue
template and has almost no negatives -> high QEI; a "garbage" case is mostly
noise with many negatives -> low QEI; "borderline" sits in between.

No scipy. Tissue maps are smooth blobs; CBF = scale * (2.5*GM + 1*WM) + noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils.smoothing import smooth_fwhm


@dataclass
class SyntheticCase:
    cbf: np.ndarray          # 3-D CBF map (mL/100g/min-ish)
    gm: np.ndarray           # GM probability [0,1]
    wm: np.ndarray           # WM probability [0,1]
    csf: np.ndarray          # CSF probability [0,1]
    brain: np.ndarray        # boolean brain mask
    voxel_mm: tuple[float, float, float]
    quality: str


def _spherical_brain(shape, rng):
    """Build GM/WM/CSF probability fields inside a spherical brain.

    Three concentric, near-disjoint regions (WM core, GM cortical shell, CSF rim)
    with LIGHT smoothing for partial-volume realism — kept light so each tissue's
    interior stays confidently above the 0.7 mask threshold.
    """
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = (np.array(shape) - 1) / 2.0
    r = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    radius = 0.45 * min(shape)
    brain = r <= radius

    wm_core = r < 0.50 * radius
    gm_shell = (r >= 0.50 * radius) & (r < 0.85 * radius)
    csf_rim = (r >= 0.85 * radius) & (r <= radius)

    gm = smooth_fwhm(gm_shell.astype(float), 1.5) * brain
    wm = smooth_fwhm(wm_core.astype(float), 1.5) * brain
    csf = smooth_fwhm(csf_rim.astype(float), 1.5) * brain

    # normalise where the three probabilities sum above 1 (keeps confident cores high)
    total = gm + wm + csf
    scale = np.clip(total, 1.0, None)
    gm, wm, csf = gm / scale, wm / scale, csf / scale
    return gm, wm, csf, brain


def synthetic_case(shape=(36, 36, 28), quality: str = "clean", seed: int = 0,
                   voxel_mm=(3.0, 3.0, 3.0)) -> SyntheticCase:
    """Generate one synthetic case. `quality` in {clean, borderline, garbage}."""
    rng = np.random.default_rng(seed)
    gm, wm, csf, brain = _spherical_brain(shape, rng)

    spcbf = 2.5 * gm + 1.0 * wm          # the QEI structural template
    gm_target = 60.0                     # aim GM CBF ~60 mL/100g/min
    scale = gm_target / 2.5              # so 2.5*1.0 -> 60
    structured = scale * spcbf

    # A SMOOTH distractor field (low-frequency, uncorrelated with tissue). Unlike
    # high-frequency noise, 5 mm smoothing can't remove it — so it genuinely lowers
    # the structural correlation. Scaled to ~structured's spread within the brain.
    distractor = smooth_fwhm(rng.standard_normal(shape), 8.0, voxel_mm)
    d = distractor[brain]
    distractor = (distractor - d.mean()) / (d.std() + 1e-6) * float(structured[brain].std())

    # cbf = w*structure + k*distractor + high-freq noise + negative bias.
    #   w  -> how much real tissue structure (the main quality knob)
    #   k  -> how much competing smooth artifact (drags rho down)
    if quality == "clean":
        w, k, sigma, neg_bias = 1.0, 0.0, 2.0, 0.0
    elif quality == "borderline":
        w, k, sigma, neg_bias = 1.0, 2.2, 5.0, 0.0
    elif quality == "garbage":
        w, k, sigma, neg_bias = 0.0, 0.0, 30.0, -12.0
    else:
        raise ValueError(f"unknown quality {quality!r}")

    cbf = (w * structured + k * distractor + rng.normal(neg_bias, sigma, shape)) * brain
    return SyntheticCase(cbf=cbf, gm=gm, wm=wm, csf=csf, brain=brain,
                         voxel_mm=voxel_mm, quality=quality)


def synthetic_control_label(shape=(16, 16, 10), n_pairs=8, contrast=0.02,
                            swapped: bool = False, seed: int = 0) -> np.ndarray:
    """A 4-D control/label series. Control volumes are slightly brighter than
    label (by `contrast`), unless `swapped`. Even index = control by convention."""
    rng = np.random.default_rng(seed)
    base = 450.0 + rng.normal(0, 1.0, shape)
    vols = []
    for i in range(2 * n_pairs):
        is_even = (i % 2 == 0)
        is_control = is_even ^ swapped
        level = base * (1.0 + contrast) if is_control else base
        vols.append(level + rng.normal(0, 0.5, shape))
    return np.stack(vols, axis=-1)
