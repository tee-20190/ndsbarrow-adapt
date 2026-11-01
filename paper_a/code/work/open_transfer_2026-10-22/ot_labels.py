"""OPEN-TRANSFER-1 labels: RCE grafheuvels (CC0) -> primary eligible barrow records, province, role.

No terrain and no detector output is read here. Run in WSL (shapely):
  <venv>/bin/python -I work/open_transfer_2026-10-22/ot_labels.py census|build
"""
import collections, hashlib, json, math, sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import shape, Point
from shapely.strtree import STRtree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIV = ROOT / "outputs/open_transfer_2026-10-22_PRIVATE"
SRC = PRIV / "src/grafheuvels.geojson"
PROV = ROOT / "outputs/benchmark_plan_2026-10-05/rev4_raw/nl_prov.json"
PROV_SHA = "30c573162429848c453f8e7ef5f58392b0936c679a6bc5774aa72976fff98188"
CORE_M = 5000          # RD New 5 km core grid, origin (0, 0)

# Role by province (fixed in SPEC section 3 before any terrain or detector output)
EVAL = ("Drenthe", "Overijssel", "Utrecht", "Noord-Holland", "Friesland", "Groningen", "Flevoland")
TEST_SEALED = ("Gelderland", "Noord-Brabant", "Limburg")  # + two reserved provinces with no primary record [names redacted; see paper_a/REDACTIONS.md]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dedup(rows):
    """NL-SCOPE-1 rule (Codex census.py, unchanged): drop a record whose centroid is within 2 m of, or whose polygon
    has IoU >= 0.5 with, an earlier-id kept record."""
    rows = sorted(rows, key=lambda r: r["id"])
    if not rows: return [], 0
    gs = [r["geom"] for r in rows]; tree = STRtree(gs); ctree = STRtree([g.centroid for g in gs]); keep = set()
    for i, g in enumerate(gs):
        cand = set(map(int, tree.query(g))) | set(map(int, ctree.query(g.centroid, predicate="dwithin", distance=2)))
        if not any(g.centroid.distance(gs[j].centroid) <= 2 or g.intersection(gs[j]).area / g.union(gs[j]).area >= .5
                   for j in cand & keep): keep.add(i)
    return [rows[i] for i in sorted(keep)], len(rows) - len(keep)


def groups(xy, d=60.0):
    if len(xy) == 0: return np.zeros(0, int)
    pairs = list(cKDTree(xy).query_pairs(d))
    if not pairs: return np.arange(len(xy))
    a, b = np.array(pairs).T
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(xy), len(xy)))
    return connected_components(g, directed=False)[1]


def load():
    raw = json.loads(SRC.read_text(encoding="utf-8"))
    assert raw["crs"]["properties"]["name"].endswith("28992")
    assert len(raw["features"]) == raw["numberMatched"] == raw["numberReturned"] == 2898, "snapshot count changed"
    assert sha(PROV) == PROV_SHA
    prov = json.loads(PROV.read_text(encoding="utf-8"))
    bounds = [(p["properties"]["statnaam"], shape(p["geometry"])) for p in prov["features"]]
    rows, invalid = [], 0
    for ft in raw["features"]:
        g = shape(ft["geometry"]) if ft.get("geometry") else None
        if g is None or not g.is_valid or g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon") or g.area <= 0:
            invalid += 1; continue
        pv = [n for n, b in bounds if b.covers(g.centroid)]
        rows.append(dict(id=ft["id"], geom=g, p=ft["properties"], province=pv[0] if len(pv) == 1 else None))
    return rows, invalid, bounds


def primary(rows):
    """NL-SCOPE-1 fixed primary cohort (unchanged): bron AMK, inwinning empty/AMK, empty opmerking, area >= 16 pi m2."""
    return [r for r in rows if r["p"].get("bron") == "AMK"
            and str(r["p"].get("inwinning") or "").strip() in ("", "AMK")
            and not str(r["p"].get("opmerking") or "").strip()
            and r["geom"].area >= math.pi * 16]


def role(province):
    if province in EVAL: return "EVAL"
    if province in TEST_SEALED: return "TEST_SEALED"
    return "UNASSIGNED"


def records():
    rows, invalid, bounds = load()
    prim, ndup = dedup(primary(rows))
    out = []
    for r in prim:
        c = r["geom"].centroid
        out.append(dict(id=r["id"], e=round(c.x, 2), n=round(c.y, 2), area_m2=round(r["geom"].area, 1),
                        province=r["province"], role=role(r["province"]),
                        core=[math.floor(c.x / CORE_M) * CORE_M, math.floor(c.y / CORE_M) * CORE_M]))
    xy = np.array([[o["e"], o["n"]] for o in out])
    lab = groups(xy)
    for o, g in zip(out, lab): o["group60"] = int(g)
    return out, dict(invalid=invalid, primary_before_dedup=len(primary(rows)), duplicates=ndup,
                     valid_all=len(rows)), bounds


def census():
    out, meta, _ = records()
    by = collections.defaultdict(list)
    for o in out: by[o["province"]].append(o)
    print(json.dumps(meta))
    for pv, rr in sorted(by.items(), key=lambda kv: -len(kv[1])):
        cores = collections.Counter(tuple(o["core"]) for o in rr)
        print(f"{str(pv):15s} {role(pv):12s} rec {len(rr):4d} groups {len({o['group60'] for o in rr}):4d} "
              f"cores {len(cores):3d} cores>=3 {sum(v >= 3 for v in cores.values()):3d} "
              f"top {sorted(cores.values(), reverse=True)[:6]}")



KAART = PRIV / "src/kaartbladindex.json"
HALO_M = 256.0
REGION_CODE = {"Drenthe": "DR", "Overijssel": "OV", "Utrecht": "UT", "Noord-Holland": "NH", "Friesland": "FR",
               "Groningen": "GR", "Flevoland": "FL"}


def build():
    """Detector-blind region specs. Core = 5 km RD square holding >= 1 EVAL primary record whose core rectangle does
    not intersect any TEST_SEALED province (halo context may); region = the province holding most of the core's EVAL
    records (tie -> name order); candidate sites = EVAL records in the region's cores."""
    from shapely.geometry import box
    out, meta, bounds = records()
    sealed = [b for n, b in bounds if n in TEST_SEALED]
    kb = json.loads(KAART.read_text(encoding="utf-8"))["features"]
    tiles = [(f["properties"]["name"], shape(f["geometry"]).bounds) for f in kb]
    cores = collections.defaultdict(list)
    for o in out:
        if o["role"] == "EVAL": cores[tuple(o["core"])].append(o)
    dropped, regions = [], collections.defaultdict(list)
    for k, rr in sorted(cores.items()):
        rect = box(k[0], k[1], k[0] + CORE_M, k[1] + CORE_M)
        if any(b.intersects(rect) for b in sealed):
            dropped.append(dict(core=list(k), records=len(rr))); continue
        pv = sorted(collections.Counter(o["province"] for o in rr).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        regions[REGION_CODE[pv]].append((k, rr))
    summary = dict(meta, eval_records_total=sum(o["role"] == "EVAL" for o in out),
                   test_sealed_records=sum(o["role"] == "TEST_SEALED" for o in out),
                   unassigned_records=sum(o["role"] == "UNASSIGNED" for o in out),
                   dropped_cores_touching_sealed=len(dropped), dropped_records=sum(d["records"] for d in dropped),
                   source_sha256=sha(SRC), kaartbladindex_sha256=sha(KAART), provinces_sha256=PROV_SHA, regions={})
    all_tiles = set()
    for R, cl in sorted(regions.items()):
        clusters, cands = [], []
        for k, rr in cl:
            ext = (k[0] - HALO_M, k[1] - HALO_M, k[0] + CORE_M + HALO_M, k[1] + CORE_M + HALO_M)
            ts = [dict(key=n, bounds=[float(v) for v in b]) for n, b in tiles
                  if min(b[2], ext[2]) > max(b[0], ext[0]) and min(b[3], ext[3]) > max(b[1], ext[1])]
            cid = f"{R}-E{k[0] // 1000:03d}N{k[1] // 1000:03d}"
            clusters.append(dict(id=cid, core=[float(k[0]), float(k[1]), float(k[0] + CORE_M), float(k[1] + CORE_M)],
                                 tiles=ts))
            cands += [dict(id=o["id"], cls="barrow", e=o["e"], n=o["n"], area_m2=o["area_m2"], group60=o["group60"],
                           province=o["province"], possible=False) for o in rr]
        d = PRIV / "regions" / R
        d.mkdir(parents=True, exist_ok=True)
        (d / "REGION.json").write_text(json.dumps(dict(region=R, phase="AHN4", halo_m=HALO_M, clusters=clusters),
                                                  indent=1), encoding="utf-8", newline="\n")
        (d / "CANDIDATE_SITES.json").write_text(json.dumps(cands, indent=1), encoding="utf-8", newline="\n")
        summary["regions"][R] = dict(cores=len(clusters), candidate_records=len(cands),
                                     groups60=len({c["group60"] for c in cands}),
                                     tiles=len({t["key"] for c in clusters for t in c["tiles"]}),
                                     region_json_sha256=sha(d / "REGION.json"),
                                     candidates_sha256=sha(d / "CANDIDATE_SITES.json"))
        all_tiles.update(t["key"] for c in clusters for t in c["tiles"])
    summary["tiles_total"] = len(all_tiles)
    (HERE / "REGIONS_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    {"census": census, "build": build}[sys.argv[1]]()
