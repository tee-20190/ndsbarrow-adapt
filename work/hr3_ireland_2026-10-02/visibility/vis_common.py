"""HOLDOUT-REGION-3 (Ireland) visibility audit, PACKET_PREPIXEL.md section 3 Step 1, AMENDMENT_1 item 4 and
STEP1_PROCEDURE.md: shared constants, sample design, estimator, decision table, neutral-folder staging and the
per-session audit configuration. Copied from the HR2 audit (work/hr2_scotland_2026-10-02/visibility/vis_common.py)
with HR3 paths, strata and limits.

Standard library only (review_ops.py, which needs numpy, is loaded on first use), so the same file serves the WSL venv
(points, fetch, NODATA, sample, freeze, render: vis_irl.py) and the Windows Python that drives the reviewer CLI (vis_review.py).
WSL->Windows interop was unavailable on 2026-10-02 (Errno 8), so the reviewer side runs Windows-native.

Judgments are source-blind model judgments (reviewer_kind=model, human_confirmed=false). They form a reported stratum
and never change the eligible set, the denominators or the HR3 decision rule.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent                  # work/hr3_ireland_2026-10-02/visibility
WORK = HERE.parent
ROOT = WORK.parents[1]
PREP_OUT = ROOT / "outputs/hr3_ireland_2026-10-02"
OUT = PREP_OUT / "visibility"
STEP2B = ROOT / "work/holdout_region_2026-10/step2b"
_E = "work/discovery_cont_2026-09-25/review/"
RUNNER_REL = _E + "ops/run_review_v2.sh"
WINDOWS = os.name == "nt"
DEV_BASE = Path("C:/dev") if WINDOWS else Path("/mnt/c/dev")

VERSION = "hr3-visibility-audit-2026-10-02.1"
SEED = 20261002

# ---------------------------------------------------------------- files
FREEZE = HERE / "AUDIT_FREEZE.json"
RENDER_INPUTS = HERE / "RENDER_INPUTS.json"
READ_PLAN = HERE / "READ_PLAN.json"
AUDIT = HERE / "REVIEW_AUDIT.json"
LOCK = HERE / "LOCK.json"
LOCKED_VERDICTS = HERE / "verdicts_locked.jsonl"
RESULT_JSON, RESULT_MD = HERE / "AUDIT_RESULT.json", HERE / "AUDIT_RESULT.md"
HASHES = HERE / "AUDIT_HASHES.json"
TEST_OUTPUT = HERE / "TEST_OUTPUT.txt"
SAMPLE_PRIVATE = OUT / "SAMPLE_PRIVATE.json"
ITEMS_PRIVATE = OUT / "ITEMS_PRIVATE.json"
READ_PLAN_PRIVATE = OUT / "READ_PLAN_PRIVATE.json"
MAPPING = OUT / "SEALED_MAPPING.json"
RENDER_RECEIPT = OUT / "RENDER_RECEIPT.json"
PANELS, RAW = OUT / "panels", OUT / "raw"
SMOKE_PANELS, SMOKE_RAW = OUT / "smoke_panels", OUT / "smoke_raw"
SESSIONS = OUT / "review_sessions"
ATTEMPTS = SESSIONS / "ATTEMPTS.jsonl"
STOP_FILE = SESSIONS / "STOP.json"
SMOKE_OUT = OUT / "smoke"
SMOKE_RECEIPT = SMOKE_OUT / "SMOKE_RECEIPT.json"
VERDICTS = OUT / "verdicts.jsonl"
JOINED_PRIVATE = OUT / "JOINED_PRIVATE.json"
POINTS_PRIVATE = OUT / "POINTS_PRIVATE.json"
RAW_ITEMS = OUT / "raw_items"
FETCH_LEDGER = OUT / "FETCH_LEDGER.jsonl"
FETCH_RECEIPT = OUT / "FETCH_RECEIPT_PRIVATE.json"
NODATA_PRIVATE = OUT / "NODATA_PARTS_PRIVATE.json"
NODATA_RESULT = HERE / "NODATA_RESULT.json"
PROCEDURE = WORK / "STEP1_PROCEDURE.md"
PROCEDURE_LOCK = WORK / "STEP1_PROCEDURE_LOCK.json"
PROCEDURE_A1 = WORK / "STEP1_PROCEDURE_AMENDMENT_1.md"
PROCEDURE_A1_LOCK = WORK / "STEP1_PROCEDURE_AMENDMENT_1_LOCK.json"
PROCEDURE_A2 = WORK / "STEP1_PROCEDURE_AMENDMENT_2.md"
PROCEDURE_A2_LOCK = WORK / "STEP1_PROCEDURE_AMENDMENT_2_LOCK.json"
POINTS_A2_PRIVATE = OUT / "POINTS_A2_PRIVATE.json"
FETCH_RECEIPT_A2 = OUT / "FETCH_RECEIPT_A2_PRIVATE.json"

# ---------------------------------------------------------------- population and sample (AMENDMENT_1 item 4)
STRATA = ("barrow", "enclosure")
N_POP = {"barrow": 9, "enclosure": 151}                 # NODATA_RESULT.json (2026-10-02), before the freeze
N_SAMPLE = {"barrow": min(10, N_POP["barrow"]), "enclosure": 50}
N_DEV = sum(N_POP.values())                             # eligible sampled DEV parts after NODATA
N_PARTS = sum(N_SAMPLE.values())
N_CONTROLS = 20
N_ITEMS = N_PARTS + N_CONTROLS
DEV_FLOOR = 120
HALF_WIN = 171                                          # raw window X +- 171 m, X = floor(cx + 0.5) (HR1 step 2b)

# ---------------------------------------------------------------- question (HR1 step 2b, unchanged)
QUESTION = ("Is a man-made earthwork form visible at the marked location? Answer VISIBLE, UNCERTAIN, NOT_VISIBLE or "
            "NOT_ASSESSABLE; give the form type for VISIBLE or UNCERTAIN, plus confidence and a note of 25 words or "
            "fewer.")
MARKED = "the centre of every view; black ticks on the four edges point to it"
RECORD_FIELDS = ("code", "answer", "form", "confidence", "note")
ANSWERS = ("VISIBLE", "UNCERTAIN", "NOT_VISIBLE", "NOT_ASSESSABLE")
FORM_ANSWERS = ("VISIBLE", "UNCERTAIN")
FORMS = ("round_mound", "ring_bank_or_ring_ditch", "enclosure", "linear_bank_or_ditch", "platform", "hollow_or_pit",
         "other")
DISCRETE_FORMS = ("round_mound", "ring_bank_or_ring_ditch", "enclosure")
CLASS_CONSISTENT = {"barrow": ("round_mound", "ring_bank_or_ring_ditch"),
                    "enclosure": ("enclosure", "ring_bank_or_ring_ditch")}
CONFIDENCE = ("high", "medium", "low")
NOTE_WORDS_SOFT, NOTE_WORDS_HARD = 25, 40
LOW_SPEC_MIN = 4

# ---------------------------------------------------------------- reviewer and session limits
PREFIX = "Q"
SMOKE_CODES = ("S1", "S2", "S3")
NEUTRAL_PREFIX = "review_earthwork_visibility_2026-10-02b"
SMOKE_PREFIX = "review_earthwork_visibility_smoke_2026-10-02b"
MODEL = "claude-opus-5"
MAX_ATTEMPTS = 2
MAX_CONSECUTIVE_FAILURES = 3
SPEND_CAP_USD = 15.0
SESSION_BUDGET_USD = 1.0
SMOKE_BUDGETS_USD = {"CONTROL_B": 1.0, "CONTROL_A": 1.0, "SMOKE_S1": 1.0, "SMOKE_S2": 1.0, "SMOKE_S3": 1.0}
SESSION_TIMEOUT_S = 1200
CODE_IN_TEXT = re.compile(r"\b(Q\d{2}|S\d)\b")
MD_RECORD = re.compile(r"^(?:\*\*|__)?(RECORD|ISSUE):?(?:\*\*|__)?:?\s+(?:\*\*|__)?`?(\{.*\})`?(?:\*\*|__)?$")
FORBIDDEN_CASED = ("CADW", "SAM", "DEV", "FIT", "TEST", "CTRL", "NHLE", "CONT", "ADAF", "TRAIN", "OGL", "BNG", "OSGB",
                   "HER", "RCAHMW", "EPSG", "HES", "NRHE", "CMK", "ITM", "SMR", "NMS", "GSI", "OPW")
FORBIDDEN_WORDS = ("wales", "welsh", "cymru", "cadw", "coflein", "rcahmw", "datamap", "britain", "british", "england",
                   "english", "ireland", "irish", "scotland", "scottish", "canmore", "barrow", "cairn", "hillfort",
                   "henge", "control", "target", "monument", "scheduled", "catalog", "positive", "negative",
                   "holdout", "held-out", "stratum", "strata", "detector", "proposal", "random", "background",
                   "sample", "inclusion", "prevalence", "empty", "heritage", "designat", "archaeolog", "region",
                   "survey", "cropmark", "phase", "sligo", "leitrim", "roscommon", "galway", "mayo",
                   "ringfort", "cashel", "townland", "eire")

# ---------------------------------------------------------------- pinned reused files
REUSED_SHA256 = {
    "work/holdout_region_2026-10/step2b/REVIEWER_PROMPT.md":
        "c75c4f8ba4c0f30016a4a35e3e0b0935e00ece6d5f18d4f7a01d25c6cfb770a3",
    "work/holdout_region_2026-10/step2b/SCHEMA.json":
        "6ff9ddf4efa1c3f6ed222156d383e94235d7d007b36d4669a031b5ca9d1d65c5",
    "work/enclosure_line_2026-09-24/enc_step1.py": "8440c7e2fc8225430ba31ad94328bbc63a005a83175aab3b8ab9309bf2f4097b",
    "work/campaign_2026-09-22/r2_train.py": "11ef821bce4ff00927880005a658002409ac32742aa28272b0720ac849bc5ddd",
    _E + "discovery_review.py": "4742a0a5e9b840e7875ed605ca47bcc2994ecfd27998d64118f5b531256f34ea",
    _E + "ops/review_ops.py": "e9bbe81a296a4d1440a1751a8287b325937c25754b5ec5bd479beb260bf89ebe",
    RUNNER_REL: "0525e81f12b26e7889ebe0a162bc21d0d5786956f159a504e5f31ed6a85b74cb",
    "verify/v3/storage_guard.py": "3e042d497fc502ef516740fda2fa23049e423322993230c30c10da11e3e79d94",
    "verify/v3/archive_gate.py": "91501e583bfecf3413d644ebae4b3ad3ae187f36bea708b774b8091b5221acfe",
}
CODE_FILES = ("visibility/vis_common.py", "visibility/vis_irl.py", "visibility/vis_review.py",
              "visibility/smoke_vis.sh", "visibility/tests/test_vis.py")
DERIVED_FROM = ("work/hr2_scotland_2026-10-02/visibility/vis_common.py", "work/hr2_scotland_2026-10-02/visibility/vis_scot.py",
                "work/hr2_scotland_2026-10-02/visibility/vis_review.py", "work/hr2_scotland_2026-10-02/visibility/smoke_vis.sh")

# ---------------------------------------------------------------- decision table (HR2 PACKET section 7; HR3 reads it as fixed in HR2)
DECISION_TABLE = (
    {"row": 1, "v_hat": ">= 0.60", "implied_delta": ">= about +8", "action": "proceed to the Step 2 GPU freeze"},
    {"row": 2, "v_hat": "0.40 to 0.60", "implied_delta": "about +5 to +8",
     "action": "freeze only if the coordinator accepts P(co-primary gain) between about 0.3 and 0.9; a NULL or "
               "PARTIAL then reads \"visibility-limited, inconclusive\", not evidence against transfer"},
    {"row": 3, "v_hat": "< 0.40", "implied_delta": "< about +5",
     "action": "do not spend GPU on this label set; return with options (another label source, another region, or a "
               "feasibility-only record)"},
)
DECISION_BASIS = ("an effect concentrated on visible parts scales roughly as Delta* = 0.116 x v_enc / 0.875, the Welsh "
                  "enclosure gain over Welsh enclosure visibility (14/16)")
FORECASTS = {"F_vis_ge_060": ("visibility audit v_hat >= 0.60", 0.55),          # PACKET_PREPIXEL section 4
             "F_vis_lt_040": ("visibility audit v_hat < 0.40 (stop)", 0.15)}


def decision_row(v: Fraction) -> dict:
    """Row of the pre-declared table reached by the exact point estimate v (>= 0.60; 0.40 to < 0.60; < 0.40)."""
    v = Fraction(v)
    row = 1 if v >= Fraction(60, 100) else 2 if v >= Fraction(40, 100) else 3
    return dict(DECISION_TABLE[row - 1])


# ---------------------------------------------------------------- small helpers
def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list:
    if not Path(path).exists():
        return []
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(_dump(obj))
    os.replace(tmp, path)


def create_only(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(_dump(obj))
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def gitbash(path: Path) -> str:
    """C:\\a\\b or /mnt/c/a/b -> /c/a/b (the form run_review_v2.sh expects for its neutral folder)."""
    p = Path(path).resolve().as_posix()
    if p.startswith("/mnt/"):
        return "/" + p[len("/mnt/"):]
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    if not m:
        raise ValueError(f"cannot express {p} as a Git Bash path")
    return f"/{m.group(1).lower()}/{m.group(2)}"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_reused() -> dict:
    bad = sorted(r for r, want in REUSED_SHA256.items() if sha(ROOT / r) != want)
    if bad:
        raise SystemExit(f"reused instruments changed since their freeze: {bad}")
    return {r: sha(ROOT / r) for r in REUSED_SHA256}


def code_hashes() -> dict:
    return {f: sha(WORK / f) for f in CODE_FILES}


def codes() -> list[str]:
    return [f"{PREFIX}{i:02d}" for i in range(1, N_ITEMS + 1)]


# ---------------------------------------------------------------- estimator (stratified Horvitz-Thompson)
def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def ht(k: dict, strata=STRATA) -> dict:
    """Stratified Horvitz-Thompson rate over the given strata of the eligible sampled DEV parts from per-stratum sample
    counts k: p = sum_h N_h k_h / n_h / sum_h N_h (exact), SE with finite-population correction
    Var = sum_h (N_h/N)^2 (1 - n_h/N_h) s_h^2 / n_h, s_h^2 = p_h (1 - p_h) n_h / (n_h - 1); census strata add 0;
    95 % normal interval clipped to [0, 1]."""
    n_tot = sum(N_POP[h] for h in strata)
    p = sum((Fraction(N_POP[h] * k[h], N_SAMPLE[h]) for h in strata), Fraction(0)) / n_tot
    var = 0.0
    for h in strata:
        nh, big = N_SAMPLE[h], N_POP[h]
        ph = k[h] / nh
        s2 = ph * (1 - ph) * nh / (nh - 1) if nh > 1 else 0.0
        var += (big / n_tot) ** 2 * (1 - nh / big) * s2 / nh
    se, pf = math.sqrt(var), float(p)
    return {"rate": round(pf, 4), "rate_exact": f"{p.numerator}/{p.denominator}", "se": round(se, 4),
            "ci95": [round(max(0.0, pf - 1.959964 * se), 4), round(min(1.0, pf + 1.959964 * se), 4)],
            "estimated_parts": round(pf * n_tot, 1), "population": n_tot, "_exact": p}


def stratum_rate(k: int, h: str) -> dict:
    nh, big = N_SAMPLE[h], N_POP[h]
    ph = k / nh
    s2 = ph * (1 - ph) * nh / (nh - 1) if nh > 1 else 0.0
    se = math.sqrt((1 - nh / big) * s2 / nh)
    return {"k": k, "n": nh, "N": big, "rate": round(ph, 4), "se_fpc": round(se, 4), "wilson95": wilson(k, nh),
            "inclusion_probability": f"{nh}/{big}"}


# ---------------------------------------------------------------- record schema and audit configuration
def validate(rec: dict) -> list:
    """HR1 step 2b record schema: code, answer, form (VISIBLE/UNCERTAIN only), confidence, note (<= 40 words)."""
    bad = []
    if set(rec) != set(RECORD_FIELDS):
        bad.append(f"fields {sorted(rec)}")
    code = rec.get("code")
    if not isinstance(code, str) or not CODE_IN_TEXT.fullmatch(code):
        bad.append("code")
    answer = rec.get("answer")
    if answer not in ANSWERS:
        bad.append("answer")
    form = rec.get("form", "<missing>")
    if (answer in FORM_ANSWERS and form not in FORMS) or (answer not in FORM_ANSWERS and form is not None):
        bad.append("form")
    if rec.get("confidence") not in CONFIDENCE:
        bad.append("confidence")
    note = rec.get("note")
    if not isinstance(note, str) or not note.strip() or len(note.split()) > NOTE_WORDS_HARD:
        bad.append("note")
    return bad


_RO: dict = {}


def _disabled(name: str):
    def refuse(*_args, **_kwargs):
        raise SystemExit(f"discovery_review.{name} belongs to the CONT/ADAF batch; this audit uses vis_review.py")
    return refuse


def ro():
    """review_ops.py (unchanged, hash-checked) configured for this audit: schema, code pattern, Markdown-tolerant
    record lines (R1 amendment 2); the frozen batch's own writers are disabled in this process."""
    if "m" in _RO:
        return _RO["m"]
    for r in (_E + "discovery_review.py", _E + "ops/review_ops.py"):
        if sha(ROOT / r) != REUSED_SHA256[r]:
            raise SystemExit(f"{r} changed since its freeze")
    m = load_module("review_ops_hr3vis", ROOT / _E / "ops/review_ops.py")
    dr = m.dr
    dr.validate = validate
    for name in ("fetch", "fetch_one", "check_queue", "render", "stage", "smoke", "collect", "lock", "join",
                 "events", "audit_session"):
        setattr(dr, name, _disabled(name))
    frozen_text_events = m._text_events

    def canonical(line: str) -> str:
        if dr.LINE.match(line.strip().strip("`")):
            return line
        mm = MD_RECORD.match(line.strip())
        return f"{mm.group(1)} {mm.group(2)}" if mm else line

    m._text_events = lambda text: frozen_text_events("\n".join(canonical(ln) for ln in text.splitlines()))
    m.CODE_IN_TEXT, m.OPS_VERSION, m.N_CASES = CODE_IN_TEXT, f"{m.OPS_VERSION}+{VERSION}", N_ITEMS
    _RO["m"] = m
    return m


def permitted_for(folder_name: str) -> re.Pattern:
    """Reads permitted in one session: its own folder's MANIFEST, SCHEMA, prompt and panels/<code>.png only."""
    return re.compile(rf"(?:^|/dev/{re.escape(folder_name)}/)"
                      r"(?:panels/(Q\d{2}|S\d)\.png|MANIFEST\.json|SCHEMA\.json|REVIEWER_PROMPT\.md)$", re.I)


def audit_transcript(path: Path, session_codes: list[str], folder_name: str) -> dict:
    m = ro()
    m.dr.PERMITTED = permitted_for(folder_name)
    return m.audit_session(m.events(Path(path)), list(session_codes))


def attempt_ok(a: dict, code: str) -> bool:
    """Accepted only when the unchanged audit accepts the code's record with no problem at all, the run finished
    without error on the pinned model, and the session had Read as its only tool and no MCP server."""
    init = a.get("init") or {}
    return (code in a["accepted"] and not a["problems"] and not a["session_excluded"] and a["final"] is not None
            and not a["final"]["is_error"] and a["models"] == [MODEL] and init.get("tools") == ["Read"]
            and not init.get("mcp_servers"))


def cost_of(path: Path, cap: float) -> float:
    if not Path(path).exists():
        return cap
    finals = [e for e in ro().events(Path(path)) if e["ev"] == "final"]
    c = finals[-1].get("cost_usd") if finals else None
    return float(c) if isinstance(c, (int, float)) else cap


# ---------------------------------------------------------------- neutral folders (one panel per session)
def session_id(code: str, attempt: int) -> str:
    return f"{code}_A{attempt}"


def session_folder(sid: str) -> Path:
    return DEV_BASE / f"{NEUTRAL_PREFIX}_{sid.lower()}"


def smoke_folder(code: str) -> Path:
    return DEV_BASE / f"{SMOKE_PREFIX}_{code.lower()}"


def manifest_text(panel_codes: list[str], view_titles: dict) -> str:
    return json.dumps({"n": len(panel_codes), "question": QUESTION, "answers": list(ANSWERS), "forms": list(FORMS),
                       "views": view_titles, "marked_location": MARKED,
                       "panels": [{"code": c, "file": f"panels/{c}.png"} for c in panel_codes]}, indent=1)


def leak_check(texts: dict) -> None:
    blob = " ".join(texts.values())
    bad = [t for t in FORBIDDEN_CASED if re.search(rf"\b{re.escape(t)}\b", blob)]
    bad += [t for t in FORBIDDEN_WORDS if re.search(rf"\b{re.escape(t)}", blob, re.I)]
    if bad:
        raise SystemExit(f"neutral texts contain forbidden tokens: {bad}")


def listing(folder: Path) -> tuple:
    files = sorted(p.relative_to(folder).as_posix() for p in Path(folder).rglob("*") if p.is_file())
    return files, sha_text("\n".join(f"{f} {sha(Path(folder) / f)}" for f in files))


def stage_folder(folder: Path, panel_png: Path, code: str, view_titles: dict, panel_sha256: str) -> dict:
    """Create-only one-panel review folder: MANIFEST.json, SCHEMA.json, REVIEWER_PROMPT.md (step 2b bytes) and
    panels/<code>.png; one uniform mtime; refuses anything else."""
    if sha(panel_png) != panel_sha256:
        raise SystemExit(f"{code}: panel differs from its render receipt")
    for r in ("work/holdout_region_2026-10/step2b/REVIEWER_PROMPT.md", "work/holdout_region_2026-10/step2b/SCHEMA.json"):
        if sha(ROOT / r) != REUSED_SHA256[r]:
            raise SystemExit(f"{r} changed")
    texts = {"MANIFEST.json": manifest_text([code], view_titles),
             "SCHEMA.json": (STEP2B / "SCHEMA.json").read_bytes().decode("utf-8"),
             "REVIEWER_PROMPT.md": (STEP2B / "REVIEWER_PROMPT.md").read_bytes().decode("utf-8")}
    leak_check(texts)
    folder.mkdir(parents=False, exist_ok=False)
    (folder / "panels").mkdir()
    with (folder / "MANIFEST.json").open("w", encoding="utf-8", newline="\n") as f:
        f.write(texts["MANIFEST.json"])
    shutil.copyfile(STEP2B / "SCHEMA.json", folder / "SCHEMA.json")
    shutil.copyfile(STEP2B / "REVIEWER_PROMPT.md", folder / "REVIEWER_PROMPT.md")
    shutil.copyfile(panel_png, folder / "panels" / f"{code}.png")
    stamp = float(int(time.time()))
    for p in [*folder.rglob("*"), folder / "panels", folder]:
        os.utime(p, (stamp, stamp))
    files, lsha = listing(folder)
    allowed = {"MANIFEST.json", "SCHEMA.json", "REVIEWER_PROMPT.md", f"panels/{code}.png"}
    if set(files) != allowed:
        raise SystemExit(f"{folder}: unexpected content {sorted(set(files) ^ allowed)}")
    return {"folder": folder.name, "files": files, "listing_sha256": lsha, "uniform_mtime": stamp,
            "manifest_sha256": sha(folder / "MANIFEST.json")}
