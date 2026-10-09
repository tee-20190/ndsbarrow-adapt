"""FRESH-1 Step 2 (Landkreis Verden): shared paths, constants, refusals and small helpers (no side effects at import).

Design: FRESH1_PREPIXEL.md sections 6-7 (locked, sha256 bd77a299...) and STEP0_DECISIONS.md (district NoData,
older delivery wins). Reused byte-identically: HR3 infer_irl.py (frozen B0 inference path through HR1 infer_dev /
tta_run / adaf_configs), HR3 vis_irl.paste (first valid value wins), HR1 region_score.py and hier_uncertainty.py
(scorer), FUSION-1 fuse_eval.py (INTERLEAVE). The only new data code is lgln_reader.py.
Private files (coordinates, tile names) live under outputs/fresh_region_2026-10-03/step2/; code and coordinate-free
records under work/fresh_region_2026-10-03/step2/.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent                     # work/fresh_region_2026-10-03/step2
FRESH = HERE.parent                                         # work/fresh_region_2026-10-03
ROOT = FRESH.parents[1]
PRIV = ROOT / "outputs/fresh_region_2026-10-03"
STEP0_OUT = PRIV / "step0"
OUT = PRIV / "step2"                                        # private Step 2 files: never committed
HR1_CHAIN = ROOT / "work/holdout_region_2026-10/chain"
HR1_STEP2 = ROOT / "work/holdout_region_2026-10/step2"
HR3_CHAIN = ROOT / "work/hr3_ireland_2026-10-02/step2/chain"
HR3_VIS = ROOT / "work/hr3_ireland_2026-10-02/visibility"
FUSION = ROOT / "work/fusion_2026-10-03"
JOINT = ROOT / "work/joint_campaign_2026-10-03"
JOINT_OUT = ROOT / "outputs/joint_campaign_2026-10-03"

# ---------------------------------------------------------------- locked inputs
PACKET = FRESH / "FRESH1_PREPIXEL.md"
PACKET_LOCK = FRESH / "FRESH1_PREPIXEL_LOCK.json"
PACKET_SHA256 = "bd77a29920ae232a227e7d2845caeb18a52f417b740a8d41dc991e7f2a4cc3de"
STEP0_HASHES = FRESH / "STEP0_HASHES.json"
STEP0_DECISIONS = FRESH / "STEP0_DECISIONS.md"
DISTRICT_GML = PRIV / "labels/landkreis_Verden_20261003T084547Z.gml"
GROUP_MANIFEST = STEP0_OUT / "GROUP_MANIFEST_PRIVATE.json"
INDIVIDUAL_MANIFEST = STEP0_OUT / "INDIVIDUAL_MANIFEST_PRIVATE.json"
NODATA_MEMBERS = STEP0_OUT / "NODATA_MEMBERS_PRIVATE.json"
BUILD_RECORD = STEP0_OUT / "BUILD_RECORD_PRIVATE.json"
TILE_INDEX = STEP0_OUT / "TILE_INDEX_PRIVATE.json"
TILE_LEDGER = STEP0_OUT / "TILE_LEDGER_PRIVATE.jsonl"
VALIDITY = STEP0_OUT / "VALIDITY_MASKS_PRIVATE.npz"
REGISTER = PRIV / "labels/denkmale_Verden_20261003T084547Z.geojson"

# ---------------------------------------------------------------- Step 2 build outputs (private)
INFER_MANIFEST = OUT / "INFER_MANIFEST_PRIVATE.json"        # tiles + windows, no parts (the VM never sees labels)
ENCLOSURE_MANIFEST = OUT / "ENCLOSURE_MANIFEST_PRIVATE.json"  # descriptive windows around the 15 enclosure records
ENCLOSURE_RECORDS = OUT / "ENCLOSURE_RECORDS_PRIVATE.json"    # their register geometries (local only, scoring)
STAGING_PLAN = OUT / "STAGING_PLAN_PRIVATE.json"
INSIDE_MASKS = OUT / "INSIDE_MASKS_PRIVATE.npz"
SMOKE_PLAN = OUT / "SMOKE_PLAN_PRIVATE.json"
SMOKE_EN_MANIFEST = OUT / "SMOKE_EN_MANIFEST_PRIVATE.json"   # EQ-1 English block: the smoke's inference check
DRYRUN_PLAN = OUT / "DRYRUN_PLAN_PRIVATE.json"
DRYRUN_DIR = OUT / "dryrun"                                 # quarantine of the CPU dry run (synthetic manifest)
PLAN_REPORT = HERE / "PLAN_REPORT.json"                     # coordinate-free
TAU_DIR = HERE / "tau"                                      # coordinate-free tau records
FREEZE_DIR = HERE / "freeze"

# ---------------------------------------------------------------- region, grid and source (packet sections 1, 6)
REGION = "Lower Saxony, Landkreis Verden (FRESH-1)"
EPSG = 25832
CRS = "EPSG:25832"
LGLN_HOST = "dgm1.s3.eu-de.cloud-object-storage.appdomain.cloud"
TILE_NAME = re.compile(r"^dgm1_32_(\d{3})_(\d{4})_1_ni_(\d{4})\.tif$")
TILE_M = TILE_PX = 1000
SRC_NODATA = -9999.0
HALO_M = 512                         # window = core (positive: the group's full core; background: the 1 km cell) + 512 m
DISTRICT_MARGIN_M = 1500.0           # every window lies within the district envelope + this margin (refused otherwise)

# ---------------------------------------------------------------- models (packet section 6)
CONT_REL = "outputs/aim_train_audit_2026-09-24/training/CONT/epoch41.pt"
CONT_SHA256 = "91091fc6d3b7640bd300c8f7820626ea4ccc5ea3ee8f9f89425000f958bdc383"
AO_SHA256 = "a2232cc15d5849d75c79c35af9b8b72cbd512766b880851c9c72493a36c79dbb"
ADAF_CODE = "a44acdd"
REFERENCE_MODELS = ("ADAF_AO", "CONT")          # the tau references (tau_fresh.py; unchanged by amendment A1)
MODEL_NAME = re.compile(r"^[A-Za-z0-9_]{1,40}$")

# Amendment A1 (STEP2_AMENDMENT_A1.md, before the freeze): ADAF's public barrow HRNet is a mandatory core reference.
# Spec = JOINT-1 fit_infer.py REFS["ADAF_barrow"] (file, md5 from the Zenodo record API, sha256) = JOINT-1 FREEZE.json
# binding.refs_sha256; the verified local copy below (downloaded 2026-09-23 with the record md5; adaf_configs
# qualification PASS) travels in the bundle and is pinned like the other weights. No tau is computed for it.
BARROW = "ADAF_barrow"
BARROW_FILE = "barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar"
BARROW_REL = f"work/campaign_2026-09-22/adaf_weights/{BARROW_FILE}"
BARROW_SHA256 = "51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216"
BARROW_MD5 = "8c2cf34634c1bc45874a324a92d2096b"
BARROW_BYTES = 861_530_273
BARROW_SOURCE = {
    "record": "Zenodo record 15848663 (ADAF ML model weights v1.0.0; concept DOI 10.5281/zenodo.15848662)",
    "url": f"https://zenodo.org/api/records/15848663/files/{BARROW_FILE}/content",
    "weights_licence": "CC-BY-SA-4.0", "code_licence": "Apache-2.0 (github.com/EarthObservation/adaf, a44acdd)",
    "attribution": "ADAF barrow HRNet, Čož, Corns, Curran, Kocev, Kokalj 2026, J. Archaeol. Sci. Rep. 71:105733; "
                   "weights CC BY-SA 4.0 (Zenodo 10.5281/zenodo.15848662), code Apache-2.0",
    "joint1_spec": "work/joint_campaign_2026-10-03/chain/fit_infer.py REFS['ADAF_barrow']",
    "local_copy": "work/campaign_2026-09-22/adaf_weights/download_class_checkpoints.sh (2026-09-23, md5 checked); "
                  "work/campaign_2026-09-22/adaf_qualification/barrow/qualification.json PASS"}
CORE_REFERENCES = ("ADAF_AO", BARROW, "CONT")    # A1: every core pass runs these three plus S*
A1_RULE = ("SUPERIOR if DEPLOY - ADAF_AO and DEPLOY - ADAF_barrow both have lo95 > 0; INFERIOR if either hi95 < 0; "
           "else NOT_DISTINGUISHED (group unit; hierarchical_ci, one pair each, the same 2,000 draws and seed)")

# ---------------------------------------------------------------- operating point and scoring (packet section 6)
RATE_COMPONENT = 0.5                 # tau of a DEPLOY component: pooled DEV background rate first exceeds 0.5 / km2
RATE_SINGLE = 1.0                    # tau of a single system
DEV_POOL_KM2 = 489.6                 # England holdout + Wales DEV + Ireland DEV (JOINT-1's frozen pools)
BOOT_SEED, BOOT_DRAWS = 20261003, 2000
ENCLOSURE_DENSITY = 2.0              # enclosures: hit or miss per system within 2 background proposals per km2
ENCLOSURE_TYPES = ("Ringwall", "Burg", "Schanze", "Motte (Turmhügelburg)", "Wasserburg", "Befestigter Gutshof",
                   "Umwallung/Gehege [landw./forstw.]")      # packet section 2 item 7 (fachbereich Archäologie)
ENCLOSURE_COUNT = 15
REACH_M = 15.0                       # region_score reach
V_HAT_PROCEED, V_HAT_STOP = 0.60, 0.40

# ---------------------------------------------------------------- resources (packet section 7)
CU_CEILING = 60.0
LOCAL_PEAK_GB = 3.0
LOCAL_RESERVE_GB = 23.0              # coordinator: C: free >= 23 GB at all times (JOINT-1 needs 22.5 GB to launch)
VM_RESERVE_GB = 20.0

# Refused sources and terrain (the project's sealed TEST and reserved sets all lie outside Lower Saxony; this chain
# reads only Step 0's LGLN DGM1 records of Landkreis Verden, so every other host or tile scheme is refused in code).
REFUSED_HOSTS = ("gsi.geodata.gov.ie", "environment.data.gov.uk", "datamap.gov.wales", "geoservices.ign.fr",
                 "data.geopf.fr", "lidar.amazon")
REFUSED_PATH_TOKENS = ("/TEST_", "_TEST/", "test_block", "reserved", "VALIDATION", "holdout_region_2026-10/test")


class RefusedTerrain(RuntimeError):
    """An input lies outside Landkreis Verden's Step 0 terrain (TEST, reserved or another region): nothing is read."""


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def digest(obj: Any) -> str:
    return sha256_bytes(canonical(obj))


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_bytes(obj: Any, indent: int | None = 1) -> bytes:
    return (json.dumps(obj, indent=indent, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path | str, obj: Any, indent: int | None = 1) -> Path:
    """LF-only UTF-8 with a trailing newline, atomic replace (never CRLF on Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(json_bytes(obj, indent))
    tmp.replace(path)
    return path


def create_only(path: Path | str, obj: Any, indent: int | None = 1) -> Path:
    if Path(path).exists():
        raise SystemExit(f"{Path(path).name} exists; create-only")
    return write_json(path, obj, indent)


def write_same(path: Path | str, obj: Any, indent: int | None = 1) -> Path:
    """Write, or accept an existing file only when its bytes are identical (deterministic rebuilds; no silent change
    of a file that may already be pinned)."""
    path, data = Path(path), json_bytes(obj, indent)
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"{path.name} exists with different content; remove it deliberately to rebuild")
        return path
    return write_json(path, obj, indent)


def rel(path: Path | str) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def boxes_overlap(a: Iterable[float], b: Iterable[float]) -> bool:
    """Half-open boxes share interior (touching edges do not)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return max(ax0, bx0) < min(ax1, bx1) and max(ay0, by0) < min(ay1, by1)


def in_box(x: float, y: float, b: Iterable[float]) -> bool:
    x0, y0, x1, y1 = b
    return x0 <= x < x1 and y0 <= y < y1


def window_bounds(core: Iterable[float], halo: int = HALO_M) -> list[int]:
    """Integer 1 m window: the core rounded outward to whole metres, plus the halo on every side."""
    x0, y0, x1, y1 = core
    return [math.floor(x0) - halo, math.floor(y0) - halo, math.ceil(x1) + halo, math.ceil(y1) + halo]


def tile_name_box(name: str) -> list[int]:
    """The 1 km box an LGLN DGM1 tile name declares (dgm1_32_<E km>_<N km>_1_ni_<year>.tif)."""
    m = TILE_NAME.match(name)
    if not m:
        raise RefusedTerrain(f"{name}: not an LGLN DGM1 tile name")
    e, n = int(m.group(1)), int(m.group(2))
    return [e * TILE_M, n * TILE_M, (e + 1) * TILE_M, (n + 1) * TILE_M]


def check_url(name: str, url: str) -> None:
    """LGLN host over https, the tile name as the last path component; refused hosts and sealed paths never pass."""
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.scheme != "https" or u.netloc != LGLN_HOST or u.path.rsplit("/", 1)[-1] != name:
        raise RefusedTerrain(f"{name}: url is off the LGLN host or names another tile")
    if any(h in url for h in REFUSED_HOSTS) or any(t.lower() in url.lower() for t in REFUSED_PATH_TOKENS):
        raise RefusedTerrain(f"{name}: url names a refused source")


def check_region(manifest: dict) -> None:
    """Every Step 2 manifest is a Landkreis Verden EPSG:25832 manifest (TEST and reserved terrain refused)."""
    if manifest.get("crs") != CRS or "Verden" not in str(manifest.get("region", "")):
        raise RefusedTerrain(f"manifest region {manifest.get('region')!r} / crs {manifest.get('crs')!r} is not "
                             f"Landkreis Verden in {CRS}")
    if str(manifest.get("role", "EVAL")).upper() in ("TEST", "RESERVED", "VALIDATION"):
        raise RefusedTerrain(f"manifest role {manifest.get('role')!r} is sealed")


def check_path(path: Path | str) -> None:
    s = Path(path).as_posix()
    if any(t.lower() in s.lower() for t in REFUSED_PATH_TOKENS):
        raise RefusedTerrain(f"{s}: path names sealed terrain")


def model_spec_name(spec: str) -> str:
    name = spec.split("=", 1)[0]
    if not MODEL_NAME.match(name):
        raise SystemExit(f"model name {name!r}: letters, digits and _ only")
    return name


def a1_violations(freeze: dict) -> list[str]:
    """Amendment A1: a freeze is usable only if ADAF_AO, ADAF_barrow and CONT are core models and ADAF_barrow carries
    the pinned weights (the driver, the session verification and the scorer all refuse otherwise)."""
    bad = [f"A1: {n} is not a core model of this freeze" for n in CORE_REFERENCES if n not in freeze.get("core", [])]
    m = freeze.get("models", {}).get(BARROW, {})
    if m.get("sha256") != BARROW_SHA256 or m.get("role") != "reference":
        bad.append(f"A1: {BARROW} is not a reference with the pinned weights ({BARROW_SHA256[:12]}...)")
    return bad
