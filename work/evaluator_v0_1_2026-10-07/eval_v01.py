"""Evaluator v0.1: benchmark v0 with role-based record treatment (SPEC.md; GPT-6 Pro Round 4, R4-6).

Barrow endpoint = benchmark v0 (work/stage1_d1_v0_2026-10-07/score_v0.py, itself a copy of detector_week1/bench_v0.py):
one-to-one maximum-cardinality matching, 15 m footprint reach (frozen region_score tree), K = floor(b x scored km2)
counting every eligible proposal, RAW (every in-core proposal, rank_key order) and NMS60 (frozen ranked_pool).

v0.1 changes only how register records act:
- target (eligible individual barrow): the entity;
- container (archaeologically defined cemetery) and legal (protection polygon): metadata only;
- other (any other record): no effect; a proposal on it consumes a slot;
- void: an explicit, frozen geometry from a void table (never from the register, never from proximity to targets),
  removed from the scored area, from proposal eligibility (before NMS and K) and from reference eligibility
  (a target whose 15 m reach zone has no area left in the scored area is removed from N).
No rule removes proposals by proximity to targets. With no voids v0.1 is v0 (reading (i), the historical
implementation), number for number.

Complex endpoint (member-supported cemetery recovery): complexes are container records; membership by register
site-number link, else by footprint-centroid containment in a valid (>= 200 m2) polygon, smallest polygon first, ties to
the lower record id; legal polygons never define membership. A complex is available when it has >= 1 eligible
member. Proposal-complex edges exist when the proposal reaches any eligible member; one-to-one maximum matching of the
same top-K proposals to available complexes.
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "work/holdout_region_2026-10/step2"))
import region_score as rs  # noqa: E402  frozen, read-only (b3337741...)

cur = rs.cur
VERSION = "v0.1"
B = (0.5, 1.0, 2.0)
K_EPS = 1e-9  # bench_v0.py / score_v0.py tolerance
DECODERS = ("RAW", "NMS60")
CONTAINER_TYPES = ("Grabhügelfeld",)
LEGAL_TYPES = ("Grabungsschutzgebiet",)
TARGET_CLASS = "barrow"
MIN_EXTENT_M2 = 200.0
SITE_KEY = re.compile(r"Gmkg\.\s*(.+?)\s+FStNr\.\s*(\d+)")
VOID_REASONS = ("NO_TERRAIN_DATA", "OUTSIDE_LICENSED_COVERAGE", "OTHER_DOCUMENTED")
VOID_FORBIDDEN_SOURCES = ("target", "entity", "reference", "barrow", "proposal", "detector", "score", "label")


# ---------------------------------------------------------------- roles and voids
def record_role(props: dict, record_id: int, target_ids: set[int]) -> str:
    """Role of one register record. Targets come from the frozen manifest's eligible entities."""
    if record_id in target_ids:
        return "target"
    kind = props.get("objekttyp")
    if kind in CONTAINER_TYPES:
        return "container"
    if kind in LEGAL_TYPES:
        return "legal"
    return "other"


@dataclass(frozen=True)
class Void:
    void_id: str
    reason_code: str
    reason: str
    derived_from: str
    geometry: object


def validate_voids(table: dict) -> list[Void]:
    """A void table: {"voids": [{void_id, reason_code, reason, derived_from, geometry (GeoJSON)}]}. Refuses unknown
    codes, empty reasons and any void derived from targets, labels or detector output."""
    out, seen = [], set()
    for v in table.get("voids", []):
        if v["reason_code"] not in VOID_REASONS:
            raise ValueError(f"void {v.get('void_id')}: unknown reason code {v['reason_code']!r}")
        if not str(v.get("reason", "")).strip() or not str(v.get("derived_from", "")).strip():
            raise ValueError(f"void {v.get('void_id')}: reason and derived_from are required")
        src = str(v["derived_from"]).lower()
        if any(tok in src for tok in VOID_FORBIDDEN_SOURCES):
            raise ValueError(f"void {v['void_id']}: derived from targets, labels or detector output ({src!r})")
        if v["void_id"] in seen:
            raise ValueError(f"duplicate void id {v['void_id']}")
        seen.add(v["void_id"])
        g = shapely.make_valid(shape(v["geometry"]))
        if g.is_empty or g.area <= 0:
            raise ValueError(f"void {v['void_id']}: empty geometry")
        out.append(Void(v["void_id"], v["reason_code"], v["reason"], v["derived_from"], g))
    return out


# ---------------------------------------------------------------- setup
@dataclass
class Setup:
    s: dict
    area_km2: float
    n_eligible: int
    reachable: np.ndarray
    void: object | None
    diag: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.reachable.sum())


def setup(manifest: dict, voids: list[Void] | None = None) -> Setup:
    s = rs.setup(manifest)
    cores = unary_union([box(*b) for t in manifest["tiles"] for b in t["core"]])
    void = unary_union([v.geometry for v in voids]) if voids else None
    scored = cores.difference(void) if void is not None else cores
    parts = {q["id"]: q for q in manifest["parts"]}
    if void is None:
        reach_ok = np.ones(len(s["eligible"]), dtype=bool)
    else:
        reach_ok = np.array([shape(parts[p]["geometry"]).buffer(rs.REACH_M).intersection(scored).area > 0
                             for p in s["eligible"]], dtype=bool)
    diag = {"voids": len(voids or []), "void_km2_in_cores": 0.0 if void is None else cores.intersection(void).area / 1e6,
            "unreachable_targets": int((~reach_ok).sum())}
    return Setup(s, scored.area / 1e6, len(s["eligible"]), reach_ok, void, diag)


def eligible_objects(objs: dict, st: Setup) -> tuple[dict, int]:
    """Drop proposals inside a void (covers: boundary counts as inside). Applied before NMS, ranking and K."""
    if st.void is None:
        return objs, 0
    shapely.prepare(st.void)
    out, dropped = {}, 0
    for tid, rows in objs.items():
        keep = [o for o in rows if not st.void.covers(Point(o["easting"], o["northing"]))]
        dropped += len(rows) - len(keep)
        out[tid] = keep
    return out, dropped


# ---------------------------------------------------------------- pools, edges, matching
def raw_pool(objs: dict, s: dict) -> list[dict]:
    rows = []
    for t in s["wins"]:
        rows += [dict(o, window=t["id"]) for o in objs[t["id"]] if rs.in_core(o, t)]
    return sorted(rows, key=cur.rank_key)


def pools(objs: dict, st: Setup) -> dict[str, list[dict]]:
    o, _ = eligible_objects(objs, st)
    return {"RAW": raw_pool(o, st.s), "NMS60": rs.ranked_pool(o, st.s)}


def edges(rows: list[dict], st: Setup) -> list[np.ndarray]:
    """Reachable target indices within 15 m reach of each proposal."""
    pts = shapely.points([r["easting"] for r in rows], [r["northing"] for r in rows]) if rows else []
    out = []
    for p in pts:
        e = np.sort(np.asarray(st.s["tree"].query(p, predicate="intersects"), dtype=np.int64))
        out.append(e[st.reachable[e]] if len(e) else e)
    return out


def max_matching(rows_edges: list[np.ndarray], n_cols: int) -> int:
    r = [i for i, e in enumerate(rows_edges) for _ in e]
    c = [int(v) for e in rows_edges for v in e]
    if not r:
        return 0
    g = csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows_edges), n_cols))
    return int((maximum_bipartite_matching(g, perm_type="column") >= 0).sum())


def budget(b: float, area_km2: float) -> int:
    return math.floor(b * area_km2 + K_EPS)


# ---------------------------------------------------------------- barrow endpoint
def score_barrow(objs: dict, st: Setup) -> dict:
    out = {}
    for dec, pool in pools(objs, st).items():
        e = edges(pool, st)
        c_full = max_matching(e, st.n_eligible)
        rec = {"supply": len(pool), "C": c_full, "G_pool": st.n - c_full}
        for b in B:
            k = budget(b, st.area_km2)
            m = max_matching(e[:k], st.n_eligible)
            rec[str(b)] = {"K": k, "matches": m, "recall": m / st.n, "ceiling_c": min(1.0, k / st.n),
                           "per_100_slots": 100.0 * m / k, "slots_filled": min(k, len(pool)),
                           "G_ranking": min(k, c_full) - m}
        rec["primary"] = float(np.mean([rec[str(b)]["recall"] for b in B]))
        out[dec] = rec
    return out


# ---------------------------------------------------------------- complex endpoint
def site_key(props: dict) -> tuple | None:
    m = SITE_KEY.search(str(props.get("siteName") or ""))
    return (m.group(1), int(m.group(2))) if m else None


@dataclass
class Complexes:
    member_of: np.ndarray          # per eligible target index: complex index or -1
    available: list[int]           # complex indices with >= 1 eligible member
    diag: dict


def _containers(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        if (r.get("properties") or {}).get("objekttyp") not in CONTAINER_TYPES or not r.get("geometry"):
            continue
        g = shapely.make_valid(shape(r["geometry"]))
        out.append({"id": int(r["id"]), "geom": g, "area": g.area, "valid": g.area >= MIN_EXTENT_M2,
                    "key": site_key(r.get("properties") or {})})
    return sorted(out, key=lambda c: c["id"])


def build_complexes(records: list[dict], manifest: dict, st: Setup) -> Complexes:
    """records: register snapshot + edge-supplement records (dicts with id, geometry, properties)."""
    cx = _containers(records)
    props = {int(r["id"]): r.get("properties") or {} for r in records}
    parts = {q["id"]: q for q in manifest["parts"]}
    key_owner: dict = {}
    for i, c in enumerate(cx):
        if c["key"] is not None:
            key_owner.setdefault(c["key"], []).append(i)
    member_of = np.full(len(st.s["eligible"]), -1, dtype=np.int64)
    via = {"site_number_link": 0, "containment": 0, "containment_overlaps_resolved": 0, "unassigned": 0}
    for j, pid in enumerate(st.s["eligible"]):
        if not st.reachable[j]:
            continue
        rid = int(parts[pid]["record_id"])
        owners = key_owner.get(site_key(props.get(rid, {})), []) if site_key(props.get(rid, {})) else []
        if len(owners) == 1:
            member_of[j] = owners[0]
            via["site_number_link"] += 1
            continue
        cen = shape(parts[pid]["geometry"]).centroid
        hits = [i for i, c in enumerate(cx) if c["valid"] and c["geom"].covers(cen)]
        if hits:
            member_of[j] = min(hits, key=lambda i: (cx[i]["area"], cx[i]["id"]))
            via["containment"] += 1
            via["containment_overlaps_resolved"] += int(len(hits) > 1)
        else:
            via["unassigned"] += 1
    avail = sorted({int(i) for i in member_of if i >= 0})
    sizes = np.bincount(member_of[member_of >= 0], minlength=len(cx)) if len(cx) else np.zeros(0)
    diag = {"container_records": len(cx), "valid_geometry": sum(c["valid"] for c in cx),
            "no_extent_geometry": sum(not c["valid"] for c in cx),
            "available_complexes": len(avail),
            "available_without_extent": sum(not cx[i]["valid"] for i in avail),
            "unavailable_complexes": len(cx) - len(avail), "eligible_targets": int(st.reachable.sum()),
            "membership": via, "members_per_available_complex": {
                "min": int(sizes[avail].min()) if avail else 0, "median": float(np.median(sizes[avail])) if avail else 0,
                "max": int(sizes[avail].max()) if avail else 0}}
    return Complexes(member_of, avail, diag)


def score_complex(objs: dict, st: Setup, cxs: Complexes) -> dict:
    col = {c: i for i, c in enumerate(cxs.available)}
    out = {}
    for dec, pool in pools(objs, st).items():
        ce = [np.unique([col[int(cxs.member_of[v])] for v in e if cxs.member_of[v] >= 0]).astype(np.int64)
              for e in edges(pool, st)]
        rec = {"supply": len(pool), "available": len(col), "C": max_matching(ce, len(col))}
        for b in B:
            k = budget(b, st.area_km2)
            m = max_matching(ce[:k], len(col))
            rec[str(b)] = {"K": k, "complexes_recovered": m, "recall": m / len(col) if col else None}
        out[dec] = rec
    return out


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))
