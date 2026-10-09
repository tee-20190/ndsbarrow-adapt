"""D1-COLLAPSE-1 sequential sessions (PACKET "Sequence"; one VM, one heavy job at a time).

  python -B sessions_d1.py projection                                       # -> CU_PROJECTION.json (pre-freeze)
  python -B sessions_d1.py plan FREEZE INDEX [--previous TOP]               # shell assignments for the driver
  python -B sessions_d1.py verify TOP FREEZE INDEX WEIGHTS_DIR [--previous TOP]   # -> TOP/SESSION_VERIFIED.json

Projection: ADAPT-MATCH-0's measured-and-frozen per-stage model (its CU_PROJECTION.json: setup, staging, gates, 509 s
per training, a DEV pass of 717 x (2.38 + 1.25 x models) s, 900 s export, 8.724 CU/h; wall cap 1.25 x chain).
Confirmation sessions need RECIPE_SELECTION.json selecting the session's recipe (select_recipe.py, after development).
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import d1_common as D  # noqa: E402

N = D.N
import match_schedule as MS  # noqa: E402
sys.path.insert(0, str(D.NDS / "chain"))
import sessions_adapt as SAD  # noqa: E402  ADAPT-NDS-1 (frozen): proposals_ok, consume_ok

REQUIRE_MARGIN, CAP_MARGIN, WALL_MARGIN = 1.10, 1.25, 1.25
MATCH_PROJ = D.MATCH / "CU_PROJECTION.json"


# ---------------------------------------------------------------- projection
def session_projection(index: int, mp: dict) -> dict:
    runs = D.session_runs(index)
    s = mp["sessions"]["2" if any(r["arm"] == "BARROW_L" for r in runs) else "1"]
    m = sum(r["infer"] for r in runs)
    train_s = len(runs) * mp["train_run_s"]
    dev_s = round(mp["dev_windows"] * (mp["per_window_base_s"] + m * mp["per_model_s"]))
    export_s = s["export_s"] + 20 * len(runs)
    chain_s = sum(s["staging_s"].values()) + s["gates_s"] + train_s + dev_s
    wall_s = s["setup_s"] + chain_s + export_s
    phase, recipe, _, _ = D.SESSIONS[index]
    return {"phase": phase, "recipe": recipe, "runs": [r["name"] for r in runs], "dev_models": m, "train_s": train_s,
            "dev_pass_s": dev_s, "chain_s": chain_s, "wall_s": wall_s, "wall_cap_s": int(WALL_MARGIN * chain_s),
            "cu": round(wall_s / 3600 * mp["cu_per_h"], 2),
            "local_peak_gb": round(len(runs) * D.WEIGHT_BYTES / 1e9 + 0.5, 2)}


def projection() -> dict:
    mp = N.read_json(MATCH_PROJ)["projection"]
    sessions = {str(i): session_projection(i, mp) for i in D.SESSIONS}
    rec = {"schema": "d1_cu_projection_v1", "utc": N.utc(),
           "source": {"path": N.rel(MATCH_PROJ), "sha256": N.sha256_file(MATCH_PROJ)},
           "projection": {k: mp[k] for k in ("cu_per_h", "per_window_base_s", "per_model_s", "train_run_s",
                                              "dev_windows")} | {"sessions": sessions, "ceilings": D.CU_CEILING}}
    rec["projection"]["paths"] = {"F1": round(sessions["1"]["cu"] + sessions["3"]["cu"] + sessions["4"]["cu"], 2),
                                  "F2": round(sum(sessions[k]["cu"] for k in ("1", "2", "13", "14")), 2)}
    N.write_json(D.CU_PROJECTION, rec)
    return rec


# ---------------------------------------------------------------- plan
def load_freeze(path: Path) -> dict:
    lock = N.read_json(Path(path).with_name("FREEZE_LOCK.json"))
    sha = N.sha256_file(path)
    if sha != lock["freeze_sha256"]:
        raise SystemExit("FREEZE.json differs from FREEZE_LOCK.json")
    f = N.read_json(path)
    f["_sha256"] = sha
    return f


def session_record(top: Path | str | None) -> dict:
    p = Path(top) / "SESSION_VERIFIED.json" if top else None
    return N.read_json(p) if p and p.is_file() else {}


def previous_ok(f: dict, index: int, previous: Path | None) -> str | None:
    if index not in D.PREVIOUS:
        return None
    prev = session_record(previous)
    want = D.PREVIOUS[index]
    if prev.get("status") != "PASS" or prev.get("index") != want or prev.get("freeze_sha256") != f["_sha256"]:
        return f"session {index} needs session {want}'s SESSION_VERIFIED.json (PASS, this freeze)"
    return None


def selection_ok(index: int) -> tuple[str | None, str | None]:
    """(violation, selection sha256): a confirmation session runs only the recipe that development selected."""
    phase, recipe, _, _ = D.SESSIONS[index]
    if phase != "confirm":
        if index == 2:
            sel = N.read_json(D.SELECTION) if D.SELECTION.is_file() else {}
            if sel.get("next") != "dev_F2":
                return "development session 2 (F2) runs only when RECIPE_SELECTION.json says F1 was not selected", None
            return None, N.sha256_file(D.SELECTION)
        return None, None
    if not D.SELECTION.is_file():
        return "confirmation needs RECIPE_SELECTION.json (select_recipe.py after development)", None
    sel = N.read_json(D.SELECTION)
    if sel.get("selected") != recipe:
        return f"RECIPE_SELECTION.json selects {sel.get('selected')}, not {recipe}", None
    return None, N.sha256_file(D.SELECTION)


def plan(f: dict, index: int, previous: Path | None) -> dict:
    if index not in D.SESSIONS:
        raise SystemExit(f"no D1 session {index}")
    for why in (previous_ok(f, index, previous), selection_ok(index)[0]):
        if why:
            raise SystemExit(why)
    pr = f["cu_projection"]["sessions"][str(index)]
    runs = D.session_runs(index)
    phase, recipe, _, _ = D.SESSIONS[index]
    return {"RECIPE": recipe, "PHASE": phase, "SEEDS": " ".join(str(s) for s in sorted({r["seed"] for r in runs})),
            "RUN_NAMES": " ".join(r["name"] for r in runs),
            "INFER_NAMES": " ".join(r["name"] for r in runs if r["infer"]),
            "PLAN_CU_REQUIRE": round(REQUIRE_MARGIN * pr["cu"], 1),
            "PLAN_SESSION_CU_CAP": round(CAP_MARGIN * pr["cu"], 1), "PLAN_WALL_CAP_S": int(pr["wall_cap_s"]),
            "PLAN_CHAIN_CU_CEILING": D.CU_CEILING[phase], "PLAN_LOCAL_PEAK_GB": pr["local_peak_gb"]}


# ---------------------------------------------------------------- verification after the export
def train_ok(tr: dict, run: dict, f: dict, weights_dir: Path, v: list) -> None:
    name, arm, seed = run["name"], run["arm"], run["seed"]
    if tr.get("status") != "COMPLETE" or tr.get("arm") != arm or tr.get("seed") != seed or \
            tr.get("recipe") != run["recipe"] or tr.get("updates") != MS.N_UPDATES:
        v.append(f"{name}: TRAIN_RESULT is not a complete {run['recipe']} {arm} run of seed {seed}")
    if tr.get("start", {}).get("sha256") != D.STARTS[arm]["sha256"]:
        v.append(f"{name}: start weights differ from {D.STARTS[arm]['registry_id']}")
    if tr.get("tables_sha256", {}).get("ls_table") != f["pins"]["ls_fit"]["sha256"]:
        v.append(f"{name}: Lower Saxony table differs from its pin")
    want = f["plan_checks"][str(seed)]["epoch_digests"]
    if {str(r.get("epoch")): r.get("draw_digest") for r in tr.get("epochs", [])} != want:
        v.append(f"{name}: trained draws differ from the frozen plan-check of seed {seed}")
    det = tr.get("detector", {})
    if det.get("last_update") != MS.N_UPDATES - 1 or "on_at_end" not in det:
        v.append(f"{name}: detector record incomplete")
    vf = weights_dir / name / "VERIFIED.json"
    if not vf.is_file() or N.read_json(vf).get("sha256") != tr.get("final_sha256") or \
            not (weights_dir / name / "final.pt").is_file():
        v.append(f"{name}: final weights not verified locally")


def verify(top: Path, f: dict, index: int, weights_dir: Path, previous: Path | None) -> dict:
    v, res = [], top / "v1" / "results"
    exp = N.read_json(top / "v1" / "EXPORT.json") if (top / "v1" / "EXPORT.json").is_file() else {}
    if exp.get("status") != "VERIFIED":
        v.append("EXPORT.json is not VERIFIED")
    if not (res / f"session_{index}.COMPLETE").is_file():
        v.append(f"session_{index}.COMPLETE missing")
    for why in (previous_ok(f, index, previous), selection_ok(index)[0]):
        v += [why] if why else []
    trained, detectors = {}, {}
    for run in D.session_runs(index):
        p = res / "runs" / run["name"] / "TRAIN_RESULT.json"
        tr = N.read_json(p) if p.is_file() else {}
        train_ok(tr, run, f, weights_dir, v)
        cur = res / "runs" / run["name"] / "curves.jsonl"
        if not cur.is_file() or N.sha256_file(cur) != tr.get("curves_sha256"):
            v.append(f"{run['name']}: curves.jsonl missing or differs from TRAIN_RESULT")
        trained[run["name"]] = tr.get("final_sha256")
        detectors[run["name"]] = tr.get("detector")
    inf = D.ROOT / f["pins"]["dev_infer_manifest"]["path"]
    tiles = {t["id"] for t in N.read_json(inf)["tiles"]}
    proposals = {}
    for run in D.session_runs(index):
        if not run["infer"]:
            continue
        name = run["name"]
        h = SAD.proposals_ok(res / "dev" / f"PROPOSALS_{name}_PRIVATE.json", name, trained.get(name) or "?",
                             f["pins"]["dev_infer_manifest"]["sha256"], tiles, v)
        SAD.consume_ok(res / "dev" / f"CONSUME_{name}.json", v)
        proposals[f"dev/PROPOSALS_{name}_PRIVATE.json"] = h
    phase, recipe, _, _ = D.SESSIONS[index]
    rec = {"index": index, "phase": phase, "recipe": recipe, "status": "FAIL" if v else "PASS", "violations": v,
           "freeze_sha256": f["_sha256"], "selection_sha256": selection_ok(index)[1], "trained": trained,
           "detectors": detectors, "dev_models": [r["name"] for r in D.session_runs(index) if r["infer"]],
           "proposals": proposals, "top": N.rel(top),
           "previous": session_record(previous).get("top") if index in D.PREVIOUS else None,
           "utc": N.utc(), "attribution": D.ATTRIBUTION}
    N.write_json(top / "SESSION_VERIFIED.json", rec)
    if v:
        raise SystemExit(f"session {index} not verified: {v[:5]}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("projection", "plan", "verify"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--previous", type=Path)
    a = ap.parse_args()
    if a.cmd == "projection":
        print(json.dumps(projection()["projection"]["paths"]))
    elif a.cmd == "plan":
        p = plan(load_freeze(Path(a.args[0])), int(a.args[1]), a.previous)
        print("\n".join(f"{k}={shlex.quote(str(v))}" for k, v in p.items()))
    else:
        top, f, idx, wdir = Path(a.args[0]), load_freeze(Path(a.args[1])), int(a.args[2]), Path(a.args[3])
        rec = verify(top, f, idx, wdir, a.previous)
        print(json.dumps({k: rec[k] for k in ("index", "recipe", "status", "trained")}))


if __name__ == "__main__":
    main()
