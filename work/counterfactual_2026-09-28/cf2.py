"""Experiment 1 v2 core: composite implants, samplers, placement and the pinned preprocessing on arrays.

PROTOCOL.md sections 3, 5 and 8.1. Pro's v1 core (cf_v1/counterfactual.py) is imported unchanged; this module adds
  - the `ditch` component (negative ring) and composite implants whose positive target is an explicit outer outline;
  - the mound-family sampler (DESIGN_SIZES.json training table) and v1's enclosure sampler;
  - placement over the feasible rectangle of the central 256 m crop, with forbidden / non-finite rejection;
  - relief_from_array(): r2_train.relief_window's body on an in-memory copy of a raw window (no file written).
All geometry is in metres on native 1 m grids; x = column axis, y = row axis, offsets from the top-left corner.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from scipy.ndimage import binary_dilation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "cf_v1"))
sys.path.insert(0, str(ROOT / "work/campaign_2026-09-22"))
import counterfactual as cf1  # noqa: E402  (Pro's v1 core, unchanged)
import r2_train as rt  # noqa: E402

adapter = rt.adapter
DESIGN = HERE / "DESIGN_SIZES.json"
MARGIN_M = 12.0
NATIVE = 423  # every R2 raw window is 423 x 423 native 1 m cells
CROP_LO, CROP_HI = 84, 339  # native cells fully inside rt.crop(img, 0, 0) = 0.5 m cells [167, 679)
CENTRE_M = NATIVE / 2  # 211.5 m: the crop centre in native metres
MIN_MAD_M = 0.005
S_MIN, S_MAX = 0.1, 0.8
DITCH_RATIO = 0.65  # v1's ditch-to-bank ratio (render(): ring - 0.65 * outer)
SLACK_M = 1.0


# ---------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Component:
    kind: str  # 'mound' | 'ditch' | 'bank_ditch' | 'ditch_bank'
    a_m: float
    b_m: float
    width_m: float
    relief_m: float
    exponent: float = 2.0
    gap_rad: float = 0.0
    gap_angle_rad: float = 0.0


@dataclass(frozen=True)
class Implant:
    x_m: float
    y_m: float
    angle_rad: float
    profile: str  # 'bump' (training) | 'cosine' (diagnostic)
    components: tuple[Component, ...]
    target_a_m: float
    target_b_m: float
    target_exponent: float = 2.0

    def at(self, x_m: float, y_m: float) -> "Implant":
        return Implant(x_m, y_m, self.angle_rad, self.profile, self.components, self.target_a_m, self.target_b_m,
                       self.target_exponent)

    def scaled(self, relief_factor: float) -> "Implant":
        comps = tuple(Component(c.kind, c.a_m, c.b_m, c.width_m, c.relief_m * relief_factor, c.exponent, c.gap_rad,
                                c.gap_angle_rad) for c in self.components)
        return Implant(self.x_m, self.y_m, self.angle_rad, self.profile, comps, self.target_a_m, self.target_b_m,
                       self.target_exponent)


def implant_to_dict(imp: Implant) -> dict:
    d = asdict(imp)
    d["components"] = [asdict(c) for c in imp.components]
    return d


def implant_from_dict(d: dict) -> Implant:
    comps = tuple(Component(**c) for c in d["components"])
    return Implant(d["x_m"], d["y_m"], d["angle_rad"], d["profile"], comps, d["target_a_m"], d["target_b_m"],
                   d["target_exponent"])


def _rho(shape: tuple[int, int], x_m: float, y_m: float, a: float, b: float, angle: float,
         exponent: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cf1.render's superellipse radius on a 1 m grid (same op sequence)."""
    yy, xx = np.indices(shape, dtype=np.float64)
    xx = (xx + .5) - x_m
    yy = (yy + .5) - y_m
    c, s = math.cos(angle), math.sin(angle)
    u, v = c * xx + s * yy, -s * xx + c * yy
    return ((np.abs(u) / a) ** exponent + (np.abs(v) / b) ** exponent) ** (1 / exponent), u, v


# Half of the full width at half maximum of cf1._bump on [-1, 1]: cosine 0.5 exactly; bump solves
# exp(-4.5 t^2)(1 - t^2)^2 = 1/2 (t = 0.32376). A ditch of mapped width W gets support W / (2 t_half) so that its
# full width at half maximum equals W under both profiles.
HALF_MAX_T = {"cosine": 0.5, "bump": 0.32376}


def _ditch(shape: tuple[int, int], imp: Implant, comp: Component) -> np.ndarray:
    """Negative ring centred on rho = 1 of (a, b) along cf1's radial surrogate distance; FWHM = width_m."""
    if not (comp.width_m > 0 and comp.width_m <= min(comp.a_m, comp.b_m) / 3 and comp.relief_m >= 0):
        raise ValueError("invalid ditch component")
    rho, _, _ = _rho(shape, imp.x_m, imp.y_m, comp.a_m, comp.b_m, imp.angle_rad, comp.exponent)
    d = (rho - 1.0) * min(comp.a_m, comp.b_m)
    support = comp.width_m / (2 * HALF_MAX_T[imp.profile])
    return -cf1._bump(d / support, imp.profile)


def _component_dz(shape: tuple[int, int], imp: Implant, comp: Component) -> np.ndarray:
    """Unit-relief field of one component (float64, relief applied by the caller)."""
    if comp.kind == "ditch":
        return _ditch(shape, imp, comp)
    spec = cf1.Earthwork(imp.x_m, imp.y_m, comp.a_m, comp.b_m, comp.width_m, 1.0, imp.angle_rad, comp.kind,
                         comp.exponent, comp.gap_rad, comp.gap_angle_rad, imp.profile)
    dz, _, _ = cf1.render(shape, 1.0, spec, margin_m=0.0)
    return dz.astype(np.float64)


def render_implant(shape: tuple[int, int], imp: Implant, margin_m: float = MARGIN_M
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (delta z float32, positive target, influence) on a native 1 m grid.

    Target = filled outer outline (target_a, target_b, target_exponent). Support = non-zero relief of any component
    or the target; influence = support dilated by margin_m (cf1's 3 x 3 structure, same Chebyshev growth)."""
    if not imp.components:
        raise ValueError("an implant needs at least one component")
    dz = np.zeros(shape, dtype=np.float64)
    support = np.zeros(shape, dtype=bool)
    for comp in imp.components:
        unit = _component_dz(shape, imp, comp)
        support |= unit != 0
        dz += unit * comp.relief_m
    rho, _, _ = _rho(shape, imp.x_m, imp.y_m, imp.target_a_m, imp.target_b_m, imp.angle_rad, imp.target_exponent)
    target = rho <= 1.0
    if not target.any():
        raise ValueError("empty target at this resolution")
    support |= target
    n = int(math.ceil(margin_m))
    ys, xs = np.nonzero(support)
    if ys.min() < n or xs.min() < n or ys.max() >= shape[0] - n or xs.max() >= shape[1] - n:
        raise ValueError("implant lacks a complete preprocessing context margin")
    influence = binary_dilation(support, structure=np.ones((3, 3), bool), iterations=n) if n else support.copy()
    return dz.astype(np.float32), target, influence


def in_crop(influence: np.ndarray) -> bool:
    ys, xs = np.nonzero(influence)
    return bool(ys.min() >= CROP_LO and ys.max() < CROP_HI and xs.min() >= CROP_LO and xs.max() < CROP_HI)


def check(imp: Implant, blocked: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render and require the influence inside the crop and clear of blocked (forbidden or non-finite) cells."""
    dz, target, influence = render_implant(blocked.shape, imp)
    if not in_crop(influence):
        raise ValueError("implant influence leaves the central crop")
    if np.any(blocked & influence):
        raise ValueError("implant touches forbidden or non-finite cells")
    return dz, target, influence


def apply(dtm_nan: np.ndarray, imp: Implant, blocked: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Augmented copy of a NaN-nodata native DTM (never modifies the input)."""
    if not np.array_equal(blocked | ~np.isfinite(dtm_nan), blocked):
        raise ValueError("blocked mask must cover every non-finite cell")
    dz, target, influence = check(imp, blocked)
    out = dtm_nan.copy()
    out += dz
    return out, target, influence


def support_extent(imp: Implant) -> tuple[float, float, float, float]:
    """Metric extent (x_min, x_max, y_min, y_max) of the relief support + target around the implant centre."""
    reach = max(imp.target_a_m, imp.target_b_m)
    for c in imp.components:
        grow = (5 * c.width_m if c.kind in ("bank_ditch", "ditch_bank") else
                c.width_m / (2 * HALF_MAX_T[imp.profile]) if c.kind == "ditch" else 0.)
        reach = max(reach, math.hypot(c.a_m, c.b_m) * (1 + grow / min(c.a_m, c.b_m)))
    half = int(math.ceil(reach)) + 4
    size = 2 * half + 1
    probe = imp.at(half + .5, half + .5)
    dz, target, _ = render_implant((size, size), probe, margin_m=0.0)
    ys, xs = np.nonzero((dz != 0) | target)
    return (xs.min() + .5 - probe.x_m, xs.max() + .5 - probe.x_m, ys.min() + .5 - probe.y_m, ys.max() + .5 - probe.y_m)


def feasible_rectangle(imp: Implant) -> tuple[float, float, float, float] | None:
    """Centre range (x_lo, x_hi, y_lo, y_hi) keeping support + margin inside crop cells [CROP_LO, CROP_HI)."""
    x0, x1, y0, y1 = support_extent(imp)
    lo, hi = CROP_LO + .5 + MARGIN_M + SLACK_M, CROP_HI - .5 - MARGIN_M - SLACK_M
    x_lo, x_hi, y_lo, y_hi = lo - x0, hi - x1, lo - y0, hi - y1
    return (x_lo, x_hi, y_lo, y_hi) if x_lo <= x_hi and y_lo <= y_hi else None


# ---------------------------------------------------------------- samplers (PROTOCOL.md 5.1-5.3)
def design_table() -> list[dict]:
    return json.loads(DESIGN.read_text(encoding="utf-8"))["training_table"]


def mound_components(row: dict, s: float) -> tuple[Component, ...]:
    a, b = row["a_out_m"], row["b_out_m"]
    if row["subtype"] == "mound":
        return (Component("mound", a, b, min(1.0, b / 3), s),)
    w = row["ditch_width_m"]
    ditch = Component("ditch", a - w / 2, b - w / 2, w, s if row["subtype"] == "ring_ditch" else DITCH_RATIO * s)
    if row["subtype"] == "ring_ditch":
        return (ditch,)
    if row["subtype"] != "mound_ditch":
        raise ValueError(f"unknown subtype {row['subtype']}")
    return (Component("mound", a - w, b - w, min(1.0, (b - w) / 3), s), ditch)


def draw_geometry(rng: np.random.Generator, table: list[dict]) -> tuple[Implant, dict]:
    """One training geometry at a placeholder centre; draw order is fixed by this code."""
    family = int(rng.integers(3))
    angle = float(rng.uniform(0, 2 * math.pi))
    s = float(math.exp(rng.uniform(math.log(S_MIN), math.log(S_MAX))))
    if family == 0:
        row = table[int(rng.integers(len(table)))]
        comps = mound_components(row, s)
        imp = Implant(CENTRE_M, CENTRE_M, angle, "bump", comps, row["a_out_m"], row["b_out_m"], 2.0)
        return imp, {"family": "mound", "subtype": row["subtype"], "strength_m": s}
    exponent = 2.0 if family == 1 else 4.0
    a = float(rng.uniform(15, 60))
    b = float(rng.uniform(max(12, .6 * a), a))
    w = float(rng.uniform(1, min(4, b / 8)))
    kind = "bank_ditch" if rng.random() < .5 else "ditch_bank"
    gap, gap_angle = (0.0, 0.0) if rng.random() < .5 else (float(rng.uniform(.1, .6)), float(rng.uniform(0, 2 * math.pi)))
    comps = (Component(kind, a, b, w, s, exponent, gap, gap_angle),)
    imp = Implant(CENTRE_M, CENTRE_M, angle, "bump", comps, a, b, exponent)
    return imp, {"family": "ellipse" if family == 1 else "rounded_rectangle", "subtype": kind, "strength_m": s}


def place(rng: np.random.Generator, imp: Implant, blocked: np.ndarray, attempts: int,
          rect: tuple[float, float, float, float]) -> tuple[Implant | None, int, np.ndarray | None]:
    """Uniform centre over the feasible rectangle; up to `attempts` draws against blocked cells.
    Returns (placed implant or None, attempts used, delta z of the placed implant)."""
    for n in range(1, attempts + 1):
        cand = imp.at(float(rng.uniform(rect[0], rect[1])), float(rng.uniform(rect[2], rect[3])))
        try:
            dz, _, _ = check(cand, blocked)
        except ValueError:
            continue
        return cand, n, dz
    return None, attempts, None


# ---------------------------------------------------------------- diagnostic shapes (PROTOCOL.md 8.1)
DIAG_STRENGTHS = (0.15, 0.30, 0.60)
GATE_SHAPES = ("mound", "ellipse", "rounded_rectangle")
SECONDARY_SHAPES = ("mound_ditch", "ring_ditch")


def diagnostic_implant(shape: str, s: float, angle: float) -> Implant:
    c = CENTRE_M
    if shape == "mound":
        return Implant(c, c, angle, "cosine", (Component("mound", 11.8, 9.6, 1.0, s),), 11.8, 9.6, 2.0)
    if shape == "ellipse":
        return Implant(c, c, angle, "cosine", (Component("ditch_bank", 35., 25., 2., s, 2.0),), 35., 25., 2.0)
    if shape == "rounded_rectangle":
        return Implant(c, c, angle, "cosine", (Component("ditch_bank", 35., 25., 2., s, 4.0, .35, 0.),), 35., 25., 4.0)
    row = {"subtype": shape, "a_out_m": 17.2, "b_out_m": 14.4, "ditch_width_m": 2.1}
    if shape in SECONDARY_SHAPES:
        return Implant(c, c, angle, "cosine", mound_components(row, s), 17.2, 14.4, 2.0)
    raise ValueError(f"unknown diagnostic shape {shape}")


# ---------------------------------------------------------------- rasters and preprocessing
def read_native(path: Path) -> tuple[np.ndarray, dict]:
    """Native DTM with nodata as NaN (the -3.4e38 sentinel is finite, so it must be masked, not tested)."""
    with rasterio.open(path) as ds:
        if ds.crs.to_epsg() != 27700 or ds.res != (1.0, 1.0) or ds.shape != (NATIVE, NATIVE):
            raise ValueError(f"{path.name}: unexpected grid {ds.crs} {ds.res} {ds.shape}")
        t = ds.transform
        if t.b != 0 or t.d != 0 or t.c != round(t.c) or t.f != round(t.f):
            raise ValueError(f"{path.name}: rotated or non-integer-metre grid")
        arr = ds.read(1, masked=True).filled(np.nan).astype("float32")
        return arr, dict(ds.profile)


def up2(mask: np.ndarray) -> np.ndarray:
    """Nearest resample of a native 1 m mask onto the 0.5 m source_view grid (each cell -> 2 x 2)."""
    return np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)


def down2_any(label: np.ndarray) -> np.ndarray:
    h, w = label.shape
    return label.reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3))


def blocked_native(dtm_nan: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Forbidden (any positive / ignore label cell) or non-finite native cells."""
    if label.shape != (2 * dtm_nan.shape[0], 2 * dtm_nan.shape[1]):
        raise ValueError("label grid is not the 0.5 m view of the DTM")
    return down2_any(label != 0) | ~np.isfinite(dtm_nan)


def relief_from_array(dtm_nan: np.ndarray, profile: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """r2_train.relief_window on an in-memory raster: (model input, finite mask, UNCLIPPED SLRM), all 0.5 m."""
    a = dtm_nan.astype("float32").copy()
    nod = profile.get("nodata")
    if nod is not None:  # restore the sentinel; a window without a nodata value keeps its own (NaN) cells
        a[~np.isfinite(a)] = nod
    with MemoryFile() as mem:
        with mem.open(**profile) as ds:
            ds.write(a, 1)
        with mem.open() as original:
            src = adapter.source_view(original)
            try:
                v = src.read(1, masked=True).filled(np.nan).astype("float32")
            finally:
                if src is not original:
                    src.close()
    r = rt.RADIUS
    data = np.full((v.shape[0] + 2 * r, v.shape[1] + 2 * r), np.nan, dtype="float32")
    data[r:-r, r:-r] = v
    slrm = adapter.rvt.vis.slrm(data, radius_cell=r)
    unclipped = np.array(slrm[r:-r, r:-r], dtype="float32", copy=True)  # normalize_image clips in place
    norm = adapter.normalize_image("slrm", slrm, -.5, .5, "value")[r:-r, r:-r]
    return np.nan_to_num(norm, nan=0.).astype("float32"), np.isfinite(norm), unclipped


def crop_centre(arr: np.ndarray) -> np.ndarray:
    return rt.crop(arr, 0, 0)


def mad(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    return float(np.median(np.abs(v - np.median(v)))) if v.size else float("nan")


def roughness(unclipped_slrm: np.ndarray) -> float:
    """MAD of unclipped SLRM over the central 256 m (0.5 m cells), finite cells only."""
    return mad(crop_centre(unclipped_slrm))


def contrast(unclipped_clean: np.ndarray, unclipped_aug: np.ndarray, influence_native: np.ndarray,
             background_mad: float) -> float:
    delta = np.abs(unclipped_aug - unclipped_clean)[up2(influence_native)]
    delta = delta[np.isfinite(delta)]
    return float(delta.max() / background_mad) if delta.size else float("nan")
