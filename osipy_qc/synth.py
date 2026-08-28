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


# --------------------------------------------------------------------------- #
# Non-brain phantoms: kidney and placenta
# --------------------------------------------------------------------------- #
# These are DELIBERATELY geometric, not anatomical. A bean-shaped kidney drawn
# by hand would look more convincing and test exactly the same code paths, while
# inviting the reader to mistake it for a validated phantom. What the checks
# actually need is: two separately-masked organs, an inner/outer contrast, a
# perfusion level in real units, and a way to make each of those wrong on
# purpose. Ellipsoids give all four and stay honest about what they are.
#
# No real renal or placental ASL data was available while these were written
# (see the dataset survey), so nothing here is calibrated against real images -
# it exists so every branch has a known answer, not to model physiology.


@dataclass
class SyntheticKidneyCase:
    rbf: np.ndarray                      # perfusion map, mL/100g/min
    delta_m: np.ndarray                  # mean subtraction image (same grid as m0)
    m0: np.ndarray                       # calibration image
    kidney_masks: dict                   # {"left": bool array, "right": bool array}
    cortex_masks: dict
    medulla_masks: dict
    voxel_mm: tuple[float, float, float]
    quality: str
    truth: dict                          # the values it was built from


@dataclass
class SyntheticPlacentaCase:
    perfusion: np.ndarray                # perfusion map (units declared by caller)
    m0: np.ndarray
    placenta_mask: np.ndarray
    voxel_mm: tuple[float, float, float]
    quality: str
    truth: dict


def _ellipsoid(shape, centre, radii):
    """Boolean ellipsoid mask: ((x-c)/r)^2 summed over axes <= 1."""
    zz, yy, xx = np.indices(shape).astype(float)
    cz, cy, cx = centre
    rz, ry, rx = radii
    return (((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0


def synthetic_kidney_case(shape=(32, 40, 40), quality: str = "clean", seed: int = 0,
                          voxel_mm=(3.0, 3.0, 3.0),
                          cortex_rbf: float = 300.0,
                          medulla_rbf: float = 120.0) -> SyntheticKidneyCase:
    """Two kidneys, each an outer cortical shell around a medullary core.

    Defaults sit mid-range in the published literature spread (cortex 139-427
    mL/100g/min, odudu2018) with a cortico-medullary ratio of 2.5, inside the
    1.5-4.0 band the CMR check treats as anatomically ordinary. They are the
    middle of a range, NOT a reference value - the design grades a 50-500 sanity
    bound precisely because no reference interval exists.

    `quality`:
      clean      - both kidneys well inside every bound
      borderline - left kidney's cortical value drifts toward the sanity edge
      garbage    - no cortico-medullary contrast, heavy negatives (a failed
                   subtraction), which should trip the negative-fraction and
                   CMR-integrity checks rather than the level check
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    cz, cy = (nz - 1) / 2.0, (ny - 1) / 2.0
    # two organs, left and right of centre, well separated so masks never touch
    offsets = {"left": nx * 0.28, "right": nx * 0.72}
    radii_outer = (nz * 0.30, ny * 0.28, nx * 0.11)
    radii_inner = (nz * 0.17, ny * 0.16, nx * 0.06)

    kidney, cortex, medulla = {}, {}, {}
    for side, cx in offsets.items():
        whole = _ellipsoid(shape, (cz, cy, cx), radii_outer)
        core = _ellipsoid(shape, (cz, cy, cx), radii_inner)
        kidney[side] = whole
        medulla[side] = core
        cortex[side] = whole & ~core          # the shell between the two

    if quality == "clean":
        levels = {"left": (cortex_rbf, medulla_rbf), "right": (cortex_rbf * 0.97, medulla_rbf)}
        sigma, neg_bias = 8.0, 0.0
    elif quality == "borderline":
        # left kidney low enough to sit near the sanity edge, and the two sides
        # far enough apart to exercise the left-vs-right consistency check
        levels = {"left": (70.0, 55.0), "right": (cortex_rbf, medulla_rbf)}
        sigma, neg_bias = 20.0, 0.0
    elif quality == "garbage":
        # no contrast at all and a negative offset: a subtraction that failed
        levels = {"left": (20.0, 20.0), "right": (20.0, 20.0)}
        sigma, neg_bias = 60.0, -40.0
    else:
        raise ValueError(f"unknown quality {quality!r}")

    rbf = np.zeros(shape, dtype=float)
    for side in offsets:
        c_level, m_level = levels[side]
        rbf[cortex[side]] = c_level
        rbf[medulla[side]] = m_level
    inside = kidney["left"] | kidney["right"]
    rbf = rbf + rng.normal(neg_bias, sigma, shape) * inside
    rbf *= inside                                  # background is exactly zero

    # M0: a flat proton-density-like image, brighter inside the organs. The PWS
    # check divides by this, so it must be comfortably positive in every ROI.
    m0 = 500.0 + rng.normal(0, 5.0, shape)
    m0 = m0 + 300.0 * inside

    # delta_m is built FROM the M0 so the perfusion-weighted signal lands where
    # the literature puts it (cortex ~2.95-3.09% of M0, garciaruiz2025) instead
    # of wherever an arbitrary scaling of the RBF map happened to fall. The
    # constant is a phantom convenience, not a quantification model: rbf is
    # divided by its own clean-case cortical level so that a clean cortex gives
    # exactly `pws_cortex_frac` of M0, and the garbage case inherits its broken
    # sign through the same path.
    pws_cortex_frac = 0.03
    delta_m = m0 * pws_cortex_frac * (rbf / cortex_rbf) * inside

    return SyntheticKidneyCase(
        rbf=rbf, delta_m=delta_m, m0=m0, kidney_masks=kidney, cortex_masks=cortex,
        medulla_masks=medulla, voxel_mm=voxel_mm, quality=quality,
        truth={"cortex_rbf": {s: levels[s][0] for s in offsets},
               "medulla_rbf": {s: levels[s][1] for s in offsets},
               "cmr": {s: (levels[s][0] / levels[s][1] if levels[s][1] else float("nan"))
                       for s in offsets},
               "pws_cortex_frac": pws_cortex_frac,
               "noise_sd": sigma, "neg_bias": neg_bias})


def synthetic_placenta_case(shape=(24, 40, 40), quality: str = "clean", seed: int = 0,
                            voxel_mm=(3.0, 3.0, 3.0),
                            perfusion: float = 180.0) -> SyntheticPlacentaCase:
    """A placental slab: one curved ROI, no left/right split, no internal
    tissue classes.

    The default 180 mL/100g/min sits inside the published whole-placenta spread
    (Zun's VSASL 176 +/- 91 and the 249-336 range across gestation), and the
    check that grades it treats that spread as context rather than a band.

    `quality`:
      clean      - homogeneous, no negatives
      borderline - strongly heterogeneous, which the design reports as INFO
                   rather than grading (placental heterogeneity is physiological)
      garbage    - heavy negatives and near-zero mean, a failed subtraction
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    # a thick curved slab: an ellipsoid shell segment, which is closer to a
    # placenta lying against the uterine wall than a box would be
    outer = _ellipsoid(shape, ((nz - 1) / 2.0, ny * 0.62, (nx - 1) / 2.0),
                       (nz * 0.42, ny * 0.46, nx * 0.42))
    inner = _ellipsoid(shape, ((nz - 1) / 2.0, ny * 0.78, (nx - 1) / 2.0),
                       (nz * 0.34, ny * 0.40, nx * 0.34))
    mask = outer & ~inner

    if quality == "clean":
        level, sigma, neg_bias, gradient = perfusion, 12.0, 0.0, 0.0
    elif quality == "borderline":
        # a strong through-slab gradient: real placentas are heterogeneous, and
        # the design explicitly refuses to grade that as a defect
        level, sigma, neg_bias, gradient = perfusion, 25.0, 0.0, 120.0
    elif quality == "garbage":
        level, sigma, neg_bias, gradient = 15.0, 70.0, -45.0, 0.0
    else:
        raise ValueError(f"unknown quality {quality!r}")

    _, yy, _ = np.indices(shape).astype(float)
    ramp = gradient * (yy / max(ny - 1, 1) - 0.5)
    perf = (level + ramp + rng.normal(neg_bias, sigma, shape)) * mask
    m0 = 400.0 + rng.normal(0, 5.0, shape) + 250.0 * mask

    return SyntheticPlacentaCase(
        perfusion=perf, m0=m0, placenta_mask=mask, voxel_mm=voxel_mm,
        quality=quality,
        truth={"level": level, "noise_sd": sigma, "neg_bias": neg_bias,
               "gradient": gradient, "n_voxels": int(mask.sum())})
