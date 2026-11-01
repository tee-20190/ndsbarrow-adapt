"""DENSE-TERRAIN-1 four-channel input (FREEZE.md Section 3.2): native-grid residuals, support, 2x bilinear render.

Native 1 m grid, m = valid source cell; separable Gaussian G_s truncated at 3 s (scipy gaussian_filter, truncate 3,
mode constant, float64):
    w_s = G_s * m,  r_s = z - G_s * (z m) / w_s     (s = 2 m and 32 m)
    support = m & (w_2 >= 0.99) & (w_32 >= 0.99)
Render to the 0.5 m view with adapter.source_view's geometry (GDAL bilinear, 2x: output pixel p samples source index
p / 2 - 0.25; separable weights 0.75 / 0.25). v = 1 iff all four stencil cells have support; residual channels are
asinh(r_2 / 0.1), asinh(r_32 / 1) where v = 1 and 0 elsewhere. The legacy channel is never touched here.

Pure numpy/scipy; no I/O except the optional raster helpers. Tested by test_dt1.py (render parity with WarpedVRT).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

SIGMAS = (2.0, 32.0)
TRUNCATE = 3.0
SUPPORT_MIN = 0.99
SCALE = {2.0: 0.1, 32.0: 1.0}          # asinh(r / scale), metres
HALO_M = 97                            # 96 m filter support + 1 source cell (FREEZE Section 3.2)


def native_products(z: np.ndarray) -> dict[str, np.ndarray]:
    """z: 1 m elevations, NaN = invalid. Returns float32 r2, r32 (0 where support is 0) and uint8 support."""
    m = np.isfinite(z)
    mf = m.astype(np.float64)
    zf = np.where(m, z, 0.0).astype(np.float64)
    zf -= zf[m].mean() if m.any() else 0.0          # numerical offset only: r is invariant to a constant
    zc = np.where(m, zf, 0.0)
    sup = m.copy()
    out = {}
    for s in SIGMAS:
        w = gaussian_filter(mf, s, truncate=TRUNCATE, mode="constant", cval=0.0)
        num = gaussian_filter(zc, s, truncate=TRUNCATE, mode="constant", cval=0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = zc - num / w
        sup &= w >= SUPPORT_MIN
        out[s] = r
    for s in SIGMAS:
        out[s] = np.where(sup, out[s], 0.0).astype(np.float32)
    return {"r2": out[2.0], "r32": out[32.0], "support": sup.astype(np.uint8)}


def _up_axis(a: np.ndarray, axis: int) -> np.ndarray:
    """2x bilinear along one axis: out[2k] = .25 a[k-1] + .75 a[k], out[2k+1] = .75 a[k] + .25 a[k+1] (edge cells
    replicated; edges never carry support in this experiment)."""
    a = np.moveaxis(a, axis, 0)
    prev = np.concatenate([a[:1], a[:-1]], 0)
    nxt = np.concatenate([a[1:], a[-1:]], 0)
    out = np.empty((2 * a.shape[0],) + a.shape[1:], dtype=np.result_type(a.dtype, np.float32))
    out[0::2] = 0.25 * prev + 0.75 * a
    out[1::2] = 0.75 * a + 0.25 * nxt
    return np.moveaxis(out, 0, axis)


def render2x(a: np.ndarray) -> np.ndarray:
    """float32 2x bilinear render (rows, then columns), adapter.source_view geometry."""
    return _up_axis(_up_axis(a.astype(np.float32), 0), 1).astype(np.float32)


def _all_axis(a: np.ndarray, axis: int) -> np.ndarray:
    a = np.moveaxis(a, axis, 0)
    prev = np.concatenate([np.zeros_like(a[:1]), a[:-1]], 0)
    nxt = np.concatenate([a[1:], np.zeros_like(a[-1:])], 0)
    out = np.empty((2 * a.shape[0],) + a.shape[1:], dtype=bool)
    out[0::2] = prev & a
    out[1::2] = a & nxt
    return np.moveaxis(out, 0, axis)


def render_support(sup: np.ndarray) -> np.ndarray:
    """v at the 0.5 m view: all four bilinear stencil cells have support (out-of-array cells count as unsupported)."""
    return _all_axis(_all_axis(sup.astype(bool), 0), 1)


def view_channels(prod: dict[str, np.ndarray]) -> np.ndarray:
    """(3, 2H, 2W) float32: asinh(r2 / .1), asinh(r32 / 1), v, rendered from native products."""
    v = render_support(prod["support"])
    a2 = np.arcsinh(render2x(prod["r2"]) / np.float32(SCALE[2.0]))
    a32 = np.arcsinh(render2x(prod["r32"]) / np.float32(SCALE[32.0]))
    return np.stack([np.where(v, a2, 0), np.where(v, a32, 0), v.astype(np.float32)]).astype(np.float32)


def window_channels(halo_z: np.ndarray, halo_px: int = HALO_M) -> np.ndarray:
    """Training: halo raw window (1 m, window + halo_px cells each side) -> the window's (3, 2h, 2w) view channels on
    exactly the 0.5 m grid adapter.source_view gives the bare window. Products are computed on the halo array, cropped to
    the window + 1 border cell, rendered, then the 2 border output pixels per side are dropped."""
    prod = native_products(halo_z)
    b = halo_px - 1
    sl = (slice(b, prod["r2"].shape[0] - b), slice(b, prod["r2"].shape[1] - b))
    ch = view_channels({k: v[sl] for k, v in prod.items()})
    return np.ascontiguousarray(ch[:, 2:-2, 2:-2])


def e_value(r2: np.ndarray, r32: np.ndarray) -> np.ndarray:
    """SSL stratification variable e = sqrt((r2 / .1)^2 + (r32 / 1)^2) (FREEZE Section 5)."""
    return np.sqrt((r2 / 0.1) ** 2 + (r32 / 1.0) ** 2)
