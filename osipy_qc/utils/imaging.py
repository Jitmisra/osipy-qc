"""
Turning arrays into pictures, with no plotting dependency.

The package is numpy + nibabel only — no scipy, no nilearn, and (deliberately)
no matplotlib. A visual report still needs images, so this module encodes them
from first principles using nothing but the standard library:

    PNG  <- zlib + struct  (about 20 lines; PNG is a simple container)
    SVG  <- f-strings      (text, so it stays sharp at any zoom and costs nothing)

That keeps `pip install osipy-qc` dependency-free, which matters for a QC tool
meant to run anywhere, and it keeps the report honest: every pixel here is drawn
from the same arrays the checks graded.
"""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np


# --------------------------------------------------------------------------- #
# PNG
# --------------------------------------------------------------------------- #
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 array as PNG bytes.

    Each scanline is prefixed with filter byte 0 ("None"), which is all a
    truecolour 8-bit PNG needs.
    """
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) uint8, got {arr.shape}")
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", zlib.compress(raw, 9))
            + _png_chunk(b"IEND", b""))


def png_data_uri(rgb: np.ndarray) -> str:
    """PNG as a base64 data: URI, so the report is a single self-contained file."""
    return "data:image/png;base64," + base64.b64encode(encode_png(rgb)).decode("ascii")


# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #
def _lerp(stops: list[tuple[float, tuple[int, int, int]]], t: np.ndarray) -> np.ndarray:
    """Piecewise-linear colour ramp over t in [0, 1] -> (..., 3) uint8."""
    out = np.zeros(t.shape + (3,), dtype=float)
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        m = (t >= p0) & (t <= p1)
        if not np.any(m):
            continue
        f = ((t[m] - p0) / (p1 - p0))[:, None]
        out[m] = np.array(c0, dtype=float) * (1 - f) + np.array(c1, dtype=float) * f
    return np.clip(out, 0, 255).astype(np.uint8)


# The perfusion colormap. This is an "inferno"-style perceptually-uniform ramp
# (black -> deep violet -> magenta -> red -> orange -> yellow), the modern
# standard for scientific/medical imaging: it reads cleanly at every intensity
# and does not muddy into a single flat red the way a naive hot ramp does.
_HOT = [
    (0.00, (0, 0, 4)),      (0.13, (31, 12, 72)),   (0.25, (85, 15, 109)),
    (0.38, (136, 34, 106)), (0.50, (186, 54, 85)),  (0.63, (227, 89, 51)),
    (0.75, (249, 140, 10)), (0.88, (249, 201, 50)), (1.00, (252, 255, 164)),
]
# A cool cyan for negative values: physically impossible CBF must be *visible*
# against the warm ramp, not lost in the dark low end. It is deliberately NOT the
# histogram's negative blue (#3B6FA8): that one has to carry on white paper, this
# one on a near-black image, and a single colour cannot do both. So captions must
# name the two separately rather than calling both "blue".
_NEG = (60, 200, 230)

#: The unit a CBF map is quantified in. One spelling in one place, so a figure
#: and the caption that repeats its numbers cannot disagree about the units.
CBF_UNITS = "mL/100 g/min"


#: Where the colour ramp starts for the lowest MEASURED value. The ramp's own
#: zero end is (0, 0, 4) - visually black - and voxels outside the organ are also
#: black, so the lowest real tissue and "no data here" rendered identically. On a
#: kidney that made the medulla, which is genuine tissue at a legitimately low
#: perfusion, look like a hole punched through the organ. Starting at 0.15 gives
#: the lowest measured value a visible dark violet (40, 12, 78) while leaving
#: exact zeros black, so "nothing" and "the least of something" look different.
_RAMP_FLOOR = 0.15


def colorise(slice2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map a 2-D slice to RGB.

    Three cases are deliberately distinguishable: exact zero (outside the imaged
    organ) stays black, negative voxels are painted cyan so the report shows them
    rather than hiding them at the bottom of the ramp, and everything else runs
    the ramp from `_RAMP_FLOOR` upward.
    """
    a = np.asarray(slice2d, dtype=float)
    a = np.where(np.isfinite(a), a, 0.0)
    span = (vmax - vmin) or 1.0
    t = np.clip((a - vmin) / span, 0.0, 1.0)
    t = _RAMP_FLOOR + (1.0 - _RAMP_FLOOR) * t
    rgb = _lerp(_HOT, t)
    rgb[a == 0] = 0            # no data: black, and distinct from the ramp floor
    rgb[a < 0] = _NEG
    return rgb


def mosaic_window(volume: np.ndarray) -> tuple[float, float]:
    """The (vmin, vmax) `slice_mosaic` would use for this volume.

    Exposed so a caller can label the colour scale. Without it the mosaic is a
    pretty picture with no units, and because the window is per-scan the colours
    are not comparable between scans - a fact the reader has to be told.
    """
    vol = np.asarray(volume, dtype=float)
    if vol.ndim == 4:
        vol = vol.mean(axis=3)
    vol = np.where(np.isfinite(vol), vol, 0.0)
    nz = vol[vol != 0]
    if not nz.size:
        return 0.0, 1.0
    lo, hi = float(np.percentile(nz, 2)), float(np.percentile(nz, 98))
    return (lo, hi if hi > lo else lo + 1.0)


def ramp_stops() -> list[tuple[float, str]]:
    """The perfusion colour ramp as (offset, #rrggbb), for drawing a colourbar.

    Rendered through the SAME floor `colorise` applies, so the bar shows the
    colours the image actually uses. Returning the raw ramp would draw a bar
    whose left end is black while no voxel in the picture is ever that colour -
    a legend that misstates its own image, which is the failure this function
    exists to prevent.
    """
    # position along the BAR (0 = vmin, 1 = vmax), coloured by what colorise
    # would paint a voxel at that value - i.e. through the same floor
    pos = np.linspace(0.0, 1.0, 9)
    rgb = _lerp(_HOT, (_RAMP_FLOOR + (1.0 - _RAMP_FLOOR) * pos).reshape(1, -1))[0]
    return [(float(p), "#%02x%02x%02x" % tuple(int(c) for c in rgb[i]))
            for i, p in enumerate(pos)]


def negative_colour() -> str:
    """The colour `colorise` paints impossible (negative) CBF, as #rrggbb.

    Exposed so a caption can *read* the colour it is about to name instead of
    guessing it: the mosaic caption used to say "blue" while the code painted
    cyan, and the ramp's own low end is a dark violet a reader will happily
    mistake for blue.
    """
    return "#%02x%02x%02x" % _NEG


def format_level(v: float) -> str:
    """A CBF number as short as it can be written without lying.

    Shared by a colourbar tick and any prose that repeats the same number, so the
    two can never round it differently.
    """
    return f"{v:.0f}" if abs(v) >= 10 else f"{v:.3g}"


def slice_axis_of(shape, voxel_mm=None) -> int:
    """Which array axis to slice along for a mosaic.

    Prefers the thickest voxel dimension, which IS the slice-encoding direction
    in any anisotropic acquisition. Falls back to the axis with the fewest
    samples, which is the same axis in almost all real data and is what an
    isotropic volume can support.

    The mosaic used to hard-code axis 2. That is right for a brain stored the
    usual way and wrong the moment it is not: on a two-kidney volume whose organs
    are separated along axis 2, it sliced BETWEEN the kidneys - half the panels
    came out empty and no panel ever showed both kidneys at once, which is the
    one thing a per-kidney report needs to show.
    """
    dims = list(shape[:3])
    if voxel_mm is not None and len(voxel_mm) >= 3:
        vox = [float(v) for v in voxel_mm[:3]]
        others = sorted(vox)
        if others[-1] >= 1.5 * others[-2]:          # genuinely anisotropic
            return int(np.argmax(vox))
    return int(np.argmin(dims))


def slice_mosaic(volume: np.ndarray, n: int = 12, cols: int = 6,
                 vmin: float | None = None, vmax: float | None = None,
                 slice_axis: int | None = None, voxel_mm=None) -> np.ndarray:
    """A grid of evenly-spaced slices through a 3-D volume -> RGB.

    Windowed to the 2nd..98th percentile of the non-zero values by default, so a
    handful of extreme voxels cannot wash the whole image out.
    """
    vol = np.asarray(volume, dtype=float)
    if vol.ndim == 4:
        vol = vol.mean(axis=3)
    if vol.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got {vol.shape}")
    vol = np.where(np.isfinite(vol), vol, 0.0)

    finite_nz = vol[vol != 0]
    if vmin is None:
        vmin = float(np.percentile(finite_nz, 2)) if finite_nz.size else 0.0
    if vmax is None:
        vmax = float(np.percentile(finite_nz, 98)) if finite_nz.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    axis = slice_axis_of(vol.shape, voxel_mm) if slice_axis is None else int(slice_axis)
    nz = vol.shape[axis]
    # Space the slices across the part of the volume that HAS data. Spanning the
    # full extent spends panels on the empty margins above and below the organ -
    # on the kidney phantom a third of the mosaic came out blank, which reads as
    # missing data rather than as empty air.
    occupied = [k for k in range(nz) if np.take(vol, k, axis=axis).any()]
    lo, hi = (occupied[0], occupied[-1]) if occupied else (0, nz - 1)
    idx = np.linspace(lo, hi, min(n, hi - lo + 1)).round().astype(int)
    tiles = [colorise(np.rot90(np.take(vol, k, axis=axis)), vmin, vmax) for k in idx]

    th, tw, _ = tiles[0].shape
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
    return canvas


# --------------------------------------------------------------------------- #
# SVG
# --------------------------------------------------------------------------- #
def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def colorbar_svg(vmin: float, vmax: float, width: int = 280, height: int = 46,
                 label: str = CBF_UNITS, n: int = 96) -> str:
    """The colour scale for a mosaic: the ramp, numeric ticks and the unit.

    A mosaic without this is a pretty picture — the reviewer cannot tell 60 from
    600, which is the difference between a healthy scan and a broken M0.

    The swatches are produced by `colorise` itself rather than from a second copy
    of the ramp, so the bar cannot drift from the image it explains: a window
    whose low end is negative comes back showing the negative colour there,
    exactly as those voxels are painted.

    Colours are literal rather than CSS variables because this SVG is also served
    on its own as image/svg+xml, where a variable resolves to nothing. That also
    means the bar adds no colour token for a print block to re-declare.
    """
    lo, hi = float(vmin), float(vmax)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return '<svg width="10" height="10"></svg>'

    x0 = 6.0
    pw = width - 2 * x0
    span = hi - lo

    def sx(v: float) -> float:
        return x0 + (v - lo) / span * pw

    mids = lo + (np.arange(n) + 0.5) / n * span
    swatches = colorise(mids[None, :], lo, hi)[0]

    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'xmlns="http://www.w3.org/2000/svg" role="img">',
             f'<title>Colour scale, {format_level(lo)} to {format_level(hi)} '
             f'{_esc(label)}</title>']

    # `n` swatches rather than a gradient, because a gradient would need its own
    # copy of the ramp. 96 is where the steps stop being visible: a stepped bar
    # implies the image is quantised, and it is not.
    bw = pw / n
    for i, px in enumerate(swatches):
        fill = "#%02x%02x%02x" % tuple(int(c) for c in px)
        # +0.6 overlap: abutting rects at fractional x leave hairline seams
        parts.append(f'<rect x="{x0 + i * bw:.1f}" y="4" width="{bw + 0.6:.1f}" height="13" '
                     f'fill="{fill}"/>')
    # An outline, because the ramp's top end is a pale yellow that otherwise
    # bleeds into the page and hides where the scale stops.
    parts.append(f'<rect x="{x0:.1f}" y="4" width="{pw:.1f}" height="13" fill="none" '
                 'stroke="#8a7d6d"/>')

    # Ends first, then the physical zero — where impossible values stop matters
    # more than the middle of a scale — then the midpoint. A tick is dropped if
    # its label would collide with one already placed, judged from the label's own
    # width (monospace at font-size 10 is about 6 px per character).
    def half_label(v: float) -> float:
        return 3.0 * len(format_level(v))

    ticks: list[float] = []
    for v in [lo, hi] + ([0.0] if lo < 0 < hi else []) + [lo + span / 2]:
        if all(abs(sx(v) - sx(k)) > half_label(v) + half_label(k) + 4 for k in ticks):
            ticks.append(v)

    fs = 'font-family="ui-monospace,Menlo,monospace" font-size="10" fill="#6B5D4F"'
    for v in ticks:
        anchor = "start" if v == lo else "end" if v == hi else "middle"
        tx = x0 if v == lo else (x0 + pw if v == hi else sx(v))
        parts.append(f'<line x1="{sx(v):.1f}" y1="17" x2="{sx(v):.1f}" y2="21" stroke="#6B5D4F"/>')
        parts.append(f'<text x="{tx:.1f}" y="31" text-anchor="{anchor}" {fs}>'
                     f'{format_level(v)}</text>')
    parts.append(f'<text x="{width / 2:.0f}" y="43" text-anchor="middle" {fs}>{_esc(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def histogram_svg(values: np.ndarray, bins: int = 40, width: int = 520,
                  height: int = 200, label: str = f"CBF ({CBF_UNITS})",
                  bands: list[tuple[float, float, str]] | None = None) -> str:
    """A hand-drawn SVG histogram — the plot Maria asked to see.

    `bands` shades expected ranges behind the bars, e.g. the population's GM CBF
    band, so "is this distribution where it should be?" is answerable at a glance.
    Bars left of zero are drawn blue: negative CBF is impossible, and the eye
    should catch it immediately.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return '<svg width="10" height="10"></svg>'

    counts, edges = np.histogram(v, bins=bins)
    cmax = int(counts.max()) or 1
    x0, y0 = 44, 8                       # plot origin (leave room for axes)
    pw, ph = width - x0 - 10, height - y0 - 26
    lo, hi = float(edges[0]), float(edges[-1])
    span = (hi - lo) or 1.0

    def sx(x: float) -> float:
        return x0 + (x - lo) / span * pw

    parts = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'xmlns="http://www.w3.org/2000/svg" role="img">',
             f'<title>{_esc(label)} distribution</title>']

    # Expected-range shading, behind everything.
    #
    # If the band falls entirely outside the plotted range the shading would be
    # clipped to nothing, leaving a caption that promises an expected range the
    # reader cannot see - on a badly scaled map that reads as a healthy
    # distribution. Draw an explicit off-scale marker instead of omitting it.
    for blo, bhi, colour in (bands or []):
        bx0, bx1 = sx(max(blo, lo)), sx(min(bhi, hi))
        if bx1 > bx0:
            parts.append(f'<rect x="{bx0:.1f}" y="{y0}" width="{bx1-bx0:.1f}" '
                         f'height="{ph}" fill="{colour}" opacity="0.5"/>')
        else:
            left = bhi <= lo
            ex = x0 + 2 if left else x0 + pw - 2
            parts.append(
                f'<line x1="{ex:.1f}" y1="{y0}" x2="{ex:.1f}" y2="{y0+ph}" '
                f'stroke="#0B8A6B" stroke-width="2" stroke-dasharray="3 3"/>'
                f'<text x="{(x0 + 8) if left else (x0 + pw - 8):.1f}" y="{y0+12}" '
                f'text-anchor="{"start" if left else "end"}" font-size="9" fill="#0B8A6B" '
                f'font-family="monospace">expected {blo:g}–{bhi:g} '
                f'{"◂" if left else "▸"} off scale</text>')

    # bars
    bw = pw / len(counts)
    for i, c in enumerate(counts):
        if c == 0:
            continue
        bh = c / cmax * ph
        bx = x0 + i * bw
        mid = (edges[i] + edges[i + 1]) / 2
        fill = "#3B6FA8" if mid < 0 else "#C2571A"   # blue = impossible negative CBF
        parts.append(f'<rect x="{bx:.1f}" y="{y0+ph-bh:.1f}" width="{max(bw-1,0.6):.1f}" '
                     f'height="{bh:.1f}" fill="{fill}"/>')

    # zero line, if the data spans it
    if lo < 0 < hi:
        zx = sx(0.0)
        parts.append(f'<line x1="{zx:.1f}" y1="{y0}" x2="{zx:.1f}" y2="{y0+ph}" '
                     'stroke="#3B6FA8" stroke-width="1" stroke-dasharray="3 2"/>')

    # axes
    parts.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#8a7d6d"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}" stroke="#8a7d6d"/>')
    fs = 'font-family="ui-monospace,Menlo,monospace" font-size="9" fill="#6B5D4F"'
    for frac in (0.0, 0.5, 1.0):
        xv = lo + frac * span
        parts.append(f'<text x="{sx(xv):.1f}" y="{y0+ph+13:.0f}" text-anchor="middle" {fs}>'
                     f'{xv:.0f}</text>')
    parts.append(f'<text x="4" y="{y0+9}" {fs}>{cmax}</text>')
    parts.append(f'<text x="4" y="{y0+ph}" {fs}>0</text>')
    parts.append(f'<text x="{x0+pw/2:.0f}" y="{height-2}" text-anchor="middle" {fs}>'
                 f'{_esc(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)
