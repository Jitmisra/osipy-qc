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


# A perfusion-style warm ramp (black -> red -> orange -> yellow -> white).
_HOT = [(0.0, (0, 0, 0)), (0.35, (150, 25, 10)), (0.6, (230, 110, 20)),
        (0.85, (250, 210, 60)), (1.0, (255, 255, 255))]
# Blue for negative values: physically impossible CBF should be *visible*, not
# silently clipped to black.
_NEG = (60, 120, 220)


def colorise(slice2d: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map a 2-D slice to RGB. Negative voxels are painted blue so the report
    shows them rather than hiding them at the bottom of the ramp."""
    a = np.asarray(slice2d, dtype=float)
    a = np.where(np.isfinite(a), a, 0.0)
    span = (vmax - vmin) or 1.0
    t = np.clip((a - vmin) / span, 0.0, 1.0)
    rgb = _lerp(_HOT, t)
    rgb[a < 0] = _NEG
    return rgb


def slice_mosaic(volume: np.ndarray, n: int = 12, cols: int = 6,
                 vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """A grid of evenly-spaced axial slices through a 3-D volume -> RGB.

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

    nz = vol.shape[2]
    idx = np.linspace(0, nz - 1, min(n, nz)).round().astype(int)
    tiles = [colorise(np.rot90(vol[:, :, k]), vmin, vmax) for k in idx]

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


def histogram_svg(values: np.ndarray, bins: int = 40, width: int = 520,
                  height: int = 200, label: str = "CBF (mL/100g/min)",
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

    # expected-range shading, behind everything
    for blo, bhi, colour in (bands or []):
        bx0, bx1 = sx(max(blo, lo)), sx(min(bhi, hi))
        if bx1 > bx0:
            parts.append(f'<rect x="{bx0:.1f}" y="{y0}" width="{bx1-bx0:.1f}" '
                         f'height="{ph}" fill="{colour}" opacity="0.5"/>')

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
