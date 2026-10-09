"""ADAPT-NDS-1 (PACKET.md, fixed 2026-10-05 00:45Z): shared paths, constants, refusals and small helpers.

No side effects at import. Private files (coordinates, register rows, masks, manifests, tile names) live under
outputs/adapt_nds_2026-10-05/; code and coordinate-free records under work/adapt_nds_2026-10-05/.
Reused read-only (imported with python -B, never copied unless named): FRESH-1 prepixel_stats.py (label rules 1-7,
groups, cores, background order), FRESH-1 step0.py (validity/NODATA/manifest functions), FRESH-1 step2 lgln_reader.py
and plan files (DEV path, unchanged), HR1 masks.py (FIT mask grid and label hash), HR1 welsh_train.py and the frozen
AMZ-ADAPT recipe modules (adapt_train, plans, r2_train, paired_train), JOINT-1 joint_train.py (J's epoch structure and
Irish draws), HR3 infer_irl.py (inference), HR1 region_score.py / hier_uncertainty.py (scorer, byte-identical) and
FUSION-1 fuse_eval.py (INTERLEAVE).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent                     # work/adapt_nds_2026-10-05
ROOT = HERE.parents[1]
PRIV = ROOT / "outputs/adapt_nds_2026-10-05"                 # private: never committed
REGISTER = PRIV / "register"                                # register snapshots, boundaries (CC BY-SA / CC BY)
CONFIRM_DIR = PRIV / "confirm"                              # CONFIRM labels and plan (never shipped to a VM)
FIT_DIR = PRIV / "fit"                                      # FIT window table (labels travel to the VM for training)
PACKET = HERE / "PACKET.md"
CONFIRM_SELECTION = HERE / "CONFIRM_SELECTION.json"          # create-only, coordinate-free

FRESH = ROOT / "work/fresh_region_2026-10-03"
FRESH_STEP2 = FRESH / "step2"
FRESH_PRIV = ROOT / "outputs/fresh_region_2026-10-03"
HR1_CHAIN = ROOT / "work/holdout_region_2026-10/chain"
HR1_STEP2 = ROOT / "work/holdout_region_2026-10/step2"
HR3_CHAIN = ROOT / "work/hr3_ireland_2026-10-02/step2/chain"
HR3_VIS = ROOT / "work/hr3_ireland_2026-10-02/visibility"
FUSION = ROOT / "work/fusion_2026-10-03"
J1_CHAIN = ROOT / "work/joint_campaign_2026-10-03/chain"
J1_OUT = ROOT / "outputs/joint_campaign_2026-10-03"

# ---------------------------------------------------------------- sources (packet "Eligible data", FRESH-1 section 7)
API = ("https://geodatendienste.denkmalatlas.niedersachsen.de/xtraserver-webapi/denkmale/v1/collections/"
       "protectedsite/items")
WFS = "https://opendata.lgln.niedersachsen.de/doorman/noauth/verwaltungsgrenzen_wfs"
INDEX = ("https://services-eu1.arcgis.com/4v3xxN52w88W065F/arcgis/rest/services/lgln_opengeodata_dgm1/FeatureServer/"
         "0/query")
CRS_URI = "http://www.opengis.net/def/crs/EPSG/0/25832"
EPSG = 25832
CRS = "EPSG:25832"
LGLN_HOST = "dgm1.s3.eu-de.cloud-object-storage.appdomain.cloud"
TILE_NAME = re.compile(r"^dgm1_32_(\d{3})_(\d{4})_1_ni_(\d{4})\.tif$")
TILE_M = TILE_PX = 1000
SRC_NODATA = -9999.0
USER_AGENT = "adapt-nds-1/1.0"

ATTRIBUTION = {
    "terrain": "LGLN (2026), DGM1, CC BY 4.0; modified (resampled to 0.5 m, SLRM)",
    "register": "Niedersächsisches Landesamt für Denkmalpflege, Denkmalatlas Niedersachsen, CC BY-SA 4.0, modified",
    "boundaries": "LGLN (2026), Verwaltungsgrenzen, CC BY 4.0",
}

# ---------------------------------------------------------------- districts (packet)
FIT_DISTRICTS = ("Uelzen", "Harburg", "Heidekreis")
DEV_DISTRICT = "Verden"
NOT_CONFIRM = FIT_DISTRICTS + (DEV_DISTRICT,)
PACKET_FIT_GRABHUEGEL = {"Uelzen": 2518, "Harburg": 1927, "Heidekreis": 1799}   # register counts quoted by the packet
GROUP_FLOOR = 200                    # FRESH-1's floor of scorable groups (after NODATA)
ADJACENCY_TOL_M = 1.0                # "share a boundary": polygons within 1 m (touching, incl. corner contact)
FIT_HALO_M = 2000.0                  # FIT windows lie >= 2 km from Verden and from CONFIRM
EDGE_M = 1000.0                      # edge supplement: other-district records within 1 km (FRESH-1 step0)

# ---------------------------------------------------------------- FIT window plan (HR1/HR3 FIT format)
FIT_WINDOW_M = 423                   # HR1/HR3 FIT windows: 423 m (846 x 846 px at 0.5 m)
FIT_SEP_M = 251.0                    # background windows: >= 251 m from any register record
FIT_IGNORE_M = 30.0                  # other register records: IGNORE + 30 m (FRESH-1 rule 6)
FIT_FRAME_M = 100.0                  # window boxes inside the FIT district union shrunk by 100 m (FRESH-1 frame)
SEED_FIT_POS, SEED_FIT_BG = 20261005, 20261006   # FIT sampling (fixed in this build, before any FIT pixel)

# ---------------------------------------------------------------- training (packet "Arms")
SEEDS = (2026092802, 2026092803, 2026092804)
ARMS = ("A", "C")
EPOCHS = (4, 5)                      # J's epoch sequence continued: two J epochs, 2 x 2,558 = 5,116 updates
N_UPDATES = 5116
LS_MOD, LS_RESIDUE = 3, 0            # A: pair p of epoch e is Lower Saxony iff (p - e) % 3 == 0 (one third of pairs)
LS_DRAW_OFFSET = 30                  # Lower Saxony draw streams: training seed + 30 (Welsh/Irish use seed + 20)
LR_MAX, LR_MIN = 1e-4, 1e-6          # J's schedule (plans.lr_at: cosine 1e-4 -> 1e-6, no warm-up), compressed

# ---------------------------------------------------------------- weights (DETECTOR_REGISTRY.json checkpoints)
J_START = {                          # JOINT-1 J final weights: the start of A and C at each seed (registry hashes)
    2026092802: ("outputs/joint_campaign_2026-10-03/weights/J_2026092802/final.pt",
                 "aa2f3f62d7573a1346c374797c24615a3cab18f911ec6c78e868bab3f9b97046", 319970119),
    2026092803: ("outputs/joint_campaign_2026-10-03/weights/J_2026092803/final.pt",
                 "3e1626583ff074e47451c2fc635b4f78e76099f00f431ccde854af998986a152", 319970119),
    2026092804: ("outputs/joint_campaign_2026-10-03/weights/J_2026092804/final.pt",
                 "58c84f3c959955809608f7fb5316a4406872d130719b9d3c0518704a6d4e063e", 319970119)}
REFERENCES = {                       # zero-shot references (registry ids ADAF-AO, ADAF-barrow, CONT-e41)
    "ADAF_AO": ("work/adaf_head_to_head_2026-09-19/weights/"
                "AO_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
                "a2232cc15d5849d75c79c35af9b8b72cbd512766b880851c9c72493a36c79dbb"),
    "ADAF_barrow": ("work/campaign_2026-09-22/adaf_weights/"
                    "barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
                    "51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216"),
    "CONT": ("outputs/aim_train_audit_2026-09-24/training/CONT/epoch41.pt",
             "91091fc6d3b7640bd300c8f7820626ea4ccc5ea3ee8f9f89425000f958bdc383")}


def run_name(arm: str, seed: int) -> str:
    """Run / model names of the trained arms (distinct from JOINT-1's C_<seed> and J_<seed>)."""
    if arm not in ARMS or seed not in SEEDS:
        raise SystemExit(f"arm {arm!r} / seed {seed} outside ADAPT-NDS-1")
    return f"{arm}_NDS_{seed}"


def j_name(seed: int) -> str:
    return f"J_{seed}"


# FRESH-1 session 1 (verified 2026-10-04T21:05:59Z): the zero-shot DEV references, bound by hash
FRESH1_S1 = "outputs/fresh_region_2026-10-03/step2/chain_runs/run_s1_20261004T174340Z"
FRESH1_S1_VERIFIED_SHA256 = None     # filled from the file at freeze time (freeze_adapt.py pins it)


# ---------------------------------------------------------------- scoring and outcome rules (packet "Measures")
BOOT_SEED, BOOT_DRAWS = 20261005, 2000
GAIN_MEAN, HARM_MEAN = 0.03, -0.03   # ADAPT_GAIN: mean >= +0.03 and lo95 > 0; ADAPT_HARM: mean <= -0.03; else NULL
BEAT_MEAN = 0.02                     # BEATS_ADAF_BARROW: mean >= +0.02 and lo95 > 0; TRAILS: hi95 < 0
RATES = (0.5, 1.0, 2.0)
REACH_M, NMS_M = 15.0, 60.0

# ---------------------------------------------------------------- chain (driver, VM, freeze)
CHAIN = HERE / "chain"
FREEZE_DIR = HERE / "freeze"
FREEZE_FILE = FREEZE_DIR / "FREEZE.json"
CU_SPEND = CHAIN / "CU_SPEND.jsonl"                         # measured spend per attempt (driver)
CU_LEDGER = CHAIN / "CU_LEDGER.jsonl"                       # colab usage readings (driver)
CU_PROJECTION = HERE / "CU_PROJECTION.json"                 # cu_projection.py (pinned by the freeze)
CHAIN_RUNS = PRIV / "chain_runs"                            # one TOP per session attempt (driver)
WEIGHTS_DIR = PRIV / "weights"                              # verified local copies of A / C final weights
VM_ROOT = "/content/DB/outputs/adapt_nds_2026-10-05/vm"
WEIGHT_BYTES = 319_970_119                                  # a J-family final.pt (registry size)

# ---------------------------------------------------------------- resources
CU_CEILING = 120.0
LOCAL_WEIGHT_GB = 2.0                # local final weights (six = 1.92 GB): raised from 1.5, BUILD_DECISIONS.md 1
LOCAL_STOP_GB = 24.0                 # stop and report if C: free falls below this
WEIGHT_DEFER_GB = 24.0               # weight downloads deferred while C: free is below this
VM_RESERVE_GB = 20.0

# Refused sources and terrain: sealed TEST, reserved terrain, VALIDATION, French and Amazon sets
REFUSED_HOSTS = ("gsi.geodata.gov.ie", "environment.data.gov.uk", "datamap.gov.wales", "geoservices.ign.fr",
                 "data.geopf.fr", "lidar.amazon")
REFUSED_PATH_TOKENS = ("/TEST_", "_TEST/", "test_block", "reserved", "VALIDATION", "holdout_region_2026-10/test",
                       "amazon", "french", "fr_disc", "france")


class RefusedTerrain(RuntimeError):
    """An input lies outside ADAPT-NDS-1's eligible data (sealed, reserved, other country, or CONFIRM in training)."""


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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
    """Write, or accept an existing file only when its bytes are identical (deterministic rebuilds)."""
    path, data = Path(path), json_bytes(obj, indent)
    if path.exists():
        if path.read_bytes() != data:
            raise SystemExit(f"{path.name} exists with different content; remove it deliberately to rebuild")
        return path
    return write_json(path, obj, indent)


def rel(path: Path | str) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def boxes_overlap(a: Iterable[float], b: Iterable[float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return max(ax0, bx0) < min(ax1, bx1) and max(ay0, by0) < min(ay1, by1)


def tile_name_box(name: str) -> list[int]:
    m = TILE_NAME.match(name)
    if not m:
        raise RefusedTerrain(f"{name}: not an LGLN DGM1 tile name")
    e, n = int(m.group(1)), int(m.group(2))
    return [e * TILE_M, n * TILE_M, (e + 1) * TILE_M, (n + 1) * TILE_M]


def check_url(name: str, url: str) -> None:
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.scheme != "https" or u.netloc != LGLN_HOST or u.path.rsplit("/", 1)[-1] != name:
        raise RefusedTerrain(f"{name}: url is off the LGLN host or names another tile")
    if any(h in url for h in REFUSED_HOSTS) or any(t.lower() in url.lower() for t in REFUSED_PATH_TOKENS):
        raise RefusedTerrain(f"{name}: url names a refused source")


def check_path(path: Path | str) -> None:
    s = Path(path).as_posix()
    if any(t.lower() in s.lower() for t in REFUSED_PATH_TOKENS):
        raise RefusedTerrain(f"{s}: path names sealed or ineligible terrain")


def district_tag(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name.replace("ü", "ue").replace("ö", "oe").replace("ä", "ae")
                  .replace("ß", "ss").replace("Ü", "Ue").replace("Ö", "Oe").replace("Ä", "Ae")).strip("_")


def free_gb(path: Path | str = ROOT) -> float:
    import shutil
    return shutil.disk_usage(path).free / 1e9


def check_disk(where: str = "") -> float:
    """The brief's local stop rule: C: free must stay >= 24.0 GB (decimal), else stop and report."""
    g = free_gb()
    if g < LOCAL_STOP_GB:
        raise SystemExit(f"STOP: C: free {g:.2f} GB < {LOCAL_STOP_GB} GB {where}".strip())
    return g
