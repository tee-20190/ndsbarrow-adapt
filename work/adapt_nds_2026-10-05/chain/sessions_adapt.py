"""ADAPT-NDS-1 sequential sessions (packet "Resources and procedure": sequential-first).

  session 1  seed 2026092802: train A and C; CONFIRM pass of A, C and every zero-shot reference (J at the three seeds,
             ADAF barrow, ADAF AO, CONT e41); DEV (Verden) pass of A and C. Verified before sessions 2-3.
  session 2  seed 2026092803: A and C; CONFIRM pass of A and C; DEV pass of A and C.
  session 3  seed 2026092804: as session 2; plus, only when the CU rule allows (dropped first), the seed-2026092802
             Welsh/Irish DEV pass of A and C from their verified local weights (packet secondary, descriptive).

  python -B sessions_adapt.py plan FREEZE INDEX [--previous TOP]     # shell assignments for the driver (eval)
  python -B sessions_adapt.py verify TOP FREEZE INDEX WEIGHTS_DIR [--previous TOP]   # -> TOP/SESSION_VERIFIED.json
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import nds_common as N  # noqa: E402
import nds_schedule as SCH  # noqa: E402

DB = "/content/DB"
REQUIRE_MARGIN, CAP_MARGIN, WI_MARGIN = 1.10, 1.25, 1.10


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


def trained_names(index: int) -> list[str]:
    return [N.run_name(a, N.SEEDS[index - 1]) for a in N.ARMS]


def confirm_refs(index: int) -> list[str]:
    return [N.j_name(s) for s in N.SEEDS] + ["ADAF_barrow", "ADAF_AO", "CONT"] if index == 1 else []


def ref_spec(f: dict, name: str) -> str:
    if name in ("ADAF_AO", "CONT"):            # HR1 infer_dev.model_spec pins (equal to the freeze's, checked there)
        return name
    m = f["models"][name]
    return f"{name}={DB}/{m['path']}@{m['sha256']}"


def spent_so_far() -> float:
    if not N.CU_SPEND.is_file():
        return 0.0
    per = {}
    for x in N.CU_SPEND.read_text(encoding="utf-8").splitlines():
        if x.strip():
            r = json.loads(x)
            per[r["attempt"]] = r
    return round(sum(float(r.get("spent", 0.0)) for r in per.values()), 2)


def wi_decision(f: dict, index: int, weights_dir: Path) -> tuple[bool, str, dict]:
    """Session 3 only: the seed-02 Welsh/Irish DEV pass runs when spent + 1.10 x (session 3 + pass) <= 120 CU and both
    seed-02 weights are verified locally against session 1's record."""
    if index != 3:
        return False, "not session 3", {}
    pr = f["cu_projection"]
    spent = spent_so_far()
    need = WI_MARGIN * (pr["sessions"]["3"]["cu"] + pr["optional_wi_dev_seed_2026092802"]["cu"])
    if spent + need > N.CU_CEILING:
        return False, f"CU: spent {spent} + {need:.2f} > {N.CU_CEILING}", {}
    s1 = session_record(f.get("_session1_top"))
    shas = {}
    for name in trained_names(1):
        d = weights_dir / name
        v = d / "VERIFIED.json"
        want = s1.get("trained", {}).get(name)
        if not want or not v.is_file() or N.read_json(v).get("sha256") != want or not (d / "final.pt").is_file():
            return False, f"{name}: no verified local weights matching session 1", {}
        shas[name] = want
    return True, f"CU: spent {spent} + {need:.2f} <= {N.CU_CEILING}", shas


def plan(f: dict, index: int, previous: Path | None, weights_dir: Path) -> dict:
    if index not in (1, 2, 3):
        raise SystemExit(f"session index {index} outside 1-3")
    if index > 1:
        prev = session_record(previous)
        if prev.get("status") != "PASS" or prev.get("index") != index - 1 or prev.get("freeze_sha256") != f["_sha256"]:
            raise SystemExit(f"session {index} needs session {index - 1}'s SESSION_VERIFIED.json (PASS, this freeze)")
        f["_session1_top"] = N.ROOT / prev["session1_top"] if prev.get("session1_top") else None
    pr = f["cu_projection"]["sessions"][str(index)]
    wi, why, wi_shas = wi_decision(f, index, weights_dir)
    wi_cu = f["cu_projection"]["optional_wi_dev_seed_2026092802"]["cu"] if wi else 0.0
    wi_s = f["cu_projection"]["optional_wi_dev_seed_2026092802"]["wall_s"] if wi else 0
    refs = confirm_refs(index)
    return {"SEED": N.SEEDS[index - 1], "ARMS": " ".join(N.ARMS), "PHASE": f"session{index}",
            "RUN_NAMES": " ".join(trained_names(index)),
            "CONFIRM_REF_NAMES": " ".join(refs), "CONFIRM_REF_SPECS": " ".join(ref_spec(f, n) for n in refs),
            "N_CONFIRM_MODELS": len(refs) + len(N.ARMS),
            "WI_DEV": int(wi), "WI_REASON": why, "WI_NAMES": " ".join(wi_shas),
            "WI_SPECS": " ".join(f"{n}={DB}/{N.rel(weights_dir / n / 'final.pt')}@{h}" for n, h in wi_shas.items()),
            "PLAN_CU_REQUIRE": round(REQUIRE_MARGIN * (pr["cu"] + wi_cu), 1),
            "PLAN_SESSION_CU_CAP": round(CAP_MARGIN * (pr["cu"] + wi_cu), 1),
            "PLAN_WALL_CAP_S": int(pr["wall_cap_s"] + CAP_MARGIN * wi_s)}


# ---------------------------------------------------------------- verification after the export
def proposals_ok(path: Path, name: str, want_ckpt: str, msha: str, tiles: set, v: list) -> str | None:
    if not path.is_file():
        v.append(f"{path.parent.name}/{path.name}: missing")
        return None
    d = N.read_json(path)
    if d.get("model") != name or d.get("checkpoint_sha256") != want_ckpt or d.get("manifest_sha256") != msha:
        v.append(f"{path.parent.name}/{path.name}: model, weights or manifest differ from the binding")
    if set(d.get("objects", {})) != tiles:
        v.append(f"{path.parent.name}/{path.name}: tiles differ from the inference manifest")
    return N.sha256_file(path)


def consume_ok(path: Path, v: list) -> None:
    if not path.is_file() or N.read_json(path).get("status") != "PASS":
        v.append(f"{path.parent.name}/{path.name}: consume record missing or not PASS")


def train_ok(tr: dict, name: str, arm: str, seed: int, f: dict, weights_dir: Path, v: list) -> None:
    if tr.get("status") != "COMPLETE" or tr.get("arm") != arm or tr.get("seed") != seed or \
            tr.get("updates") != SCH.N_UPDATES:
        v.append(f"{name}: TRAIN_RESULT is not a complete {arm} run of seed {seed} with {SCH.N_UPDATES} updates")
    if tr.get("start", {}).get("sha256") != f["start_weights"][str(seed)]["sha256"]:
        v.append(f"{name}: start weights differ from J_{seed}")
    if tr.get("english_plan_sha256") != f["pins"][f"english_e45_{seed}"]["sha256"]:
        v.append(f"{name}: English plan differs from its pin")
    want_tables = {"w_table": "w_fit", "i_table": "i_fit"} | ({"ls_table": "ls_fit"} if arm == "A" else {})
    for k, pin in want_tables.items():
        if tr.get("tables_sha256", {}).get(k) != f["pins"][pin]["sha256"]:
            v.append(f"{name}: {k} differs from its pin")
    if arm == "C" and "ls_table" in tr.get("tables_sha256", {}):
        v.append(f"{name}: the control read the Lower Saxony table")
    vf = weights_dir / name / "VERIFIED.json"
    if not vf.is_file() or N.read_json(vf).get("sha256") != tr.get("final_sha256") or \
            not (weights_dir / name / "final.pt").is_file():
        v.append(f"{name}: final weights not verified locally")


def shared_draws_ok(a: dict, c: dict, v: list) -> list[dict]:
    """Every update outside the Lower Saxony pairs had the same source, window, offsets, dihedral and update index in
    A and C (the trainers' per-epoch shared digests), and the realised update counts equal nds_schedule's."""
    rows = []
    ea = {r["epoch"]: r for r in a.get("plan_summary", {}).get("epochs", [])}
    ec = {r["epoch"]: r for r in c.get("plan_summary", {}).get("epochs", [])}
    for e in SCH.EPOCHS:
        same = e in ea and e in ec and ea[e]["shared_digest"] == ec[e]["shared_digest"]
        rows.append({"epoch": e, "shared_digest_equal": same})
        if not same:
            v.append(f"epoch {e}: A and C shared-step digests differ")
    for arm, tr in (("A", a), ("C", c)):
        got = {k: sum(r["updates"][k] for r in tr.get("plan_summary", {}).get("epochs", [])) for k in SCH.KINDS}
        if got != SCH.kind_counts(arm)["updates"]:
            v.append(f"{arm}: realised update counts {got} differ from nds_schedule")
    return rows


def verify(top: Path, f: dict, index: int, weights_dir: Path, previous: Path | None) -> dict:
    v, res, seed = [], top / "v1" / "results", N.SEEDS[index - 1]
    exp = N.read_json(top / "v1" / "EXPORT.json") if (top / "v1" / "EXPORT.json").is_file() else {}
    if exp.get("status") != "VERIFIED":
        v.append("EXPORT.json is not VERIFIED")
    if not (res / f"session_{index}.COMPLETE").is_file():
        v.append(f"session_{index}.COMPLETE missing")
    if index > 1:
        prev = session_record(previous)
        if prev.get("status") != "PASS" or prev.get("index") != index - 1 or prev.get("freeze_sha256") != f["_sha256"]:
            v.append(f"previous session {index - 1} not verified under this freeze")
    trained, tr = {}, {}
    for arm in N.ARMS:
        name = N.run_name(arm, seed)
        p = res / "runs" / name / "TRAIN_RESULT.json"
        tr[arm] = N.read_json(p) if p.is_file() else {}
        train_ok(tr[arm], name, arm, seed, f, weights_dir, v)
        trained[name] = tr[arm].get("final_sha256")
        if not (res / "runs" / name / "HOLDOUT_OBJECTS_PRIVATE.json").is_file():
            v.append(f"{name}: English holdout objects missing")
    shared = shared_draws_ok(tr["A"], tr["C"], v)
    ckpt = {n: m["sha256"] for n, m in f["models"].items()} | trained
    lock = N.read_json(N.ROOT / f["confirm"]["build_lock"])
    if lock.get("freeze_sha256") != f["_sha256"]:
        v.append("CONFIRM_BUILD_LOCK.json was made under another freeze")
    inf_c = N.ROOT / f["confirm"]["infer_manifest"]
    inf_d = N.ROOT / f["pins"]["dev_infer_manifest"]["path"]
    tiles_c = {t["id"] for t in N.read_json(inf_c)["tiles"]}
    tiles_d = {t["id"] for t in N.read_json(inf_d)["tiles"]}
    confirm_models, proposals = list(trained) + confirm_refs(index), {}
    for name in confirm_models:
        h = proposals_ok(res / "confirm" / f"PROPOSALS_{name}_PRIVATE.json", name, ckpt.get(name, "?"),
                         N.sha256_file(inf_c), tiles_c, v)
        consume_ok(res / "confirm" / f"CONSUME_{name}.json", v)
        proposals[f"confirm/PROPOSALS_{name}_PRIVATE.json"] = h
    for name in trained:
        h = proposals_ok(res / "dev" / f"PROPOSALS_{name}_PRIVATE.json", name, trained[name],
                         f["pins"]["dev_infer_manifest"]["sha256"], tiles_d, v)
        consume_ok(res / "dev" / f"CONSUME_{name}.json", v)
        proposals[f"dev/PROPOSALS_{name}_PRIVATE.json"] = h
    wi_models = []
    if (res / "WI_SCHEDULED").is_file():
        wi_models = (res / "WI_SCHEDULED").read_text(encoding="utf-8").split()
        prev_rec = session_record(previous)
        s1 = session_record(N.ROOT / prev_rec["session1_top"]) if prev_rec.get("session1_top") else {}
        for name in wi_models:
            for reg in ("wales", "ireland"):
                p = res / reg / f"PROPOSALS_{name}_PRIVATE.json"
                if not p.is_file():
                    v.append(f"{reg}/{p.name}: scheduled but missing")
                    continue
                if N.read_json(p).get("checkpoint_sha256") != s1.get("trained", {}).get(name, "?"):
                    v.append(f"{reg}/{p.name}: weights differ from session 1's {name}")
                proposals[f"{reg}/{p.name}"] = N.sha256_file(p)
    rec = {"index": index, "seed": seed, "status": "FAIL" if v else "PASS", "violations": v,
           "freeze_sha256": f["_sha256"], "trained": trained, "confirm_models": confirm_models,
           "dev_models": list(trained), "wi_models": wi_models, "shared_draws": shared, "proposals": proposals,
           "top": N.rel(top), "session1_top": N.rel(top) if index == 1 else session_record(previous).get("session1_top"),
           "utc": N.utc(), "attribution": N.ATTRIBUTION}
    N.write_json(top / "SESSION_VERIFIED.json", rec)
    if v:
        raise SystemExit(f"session {index} not verified: {v[:5]}")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "verify"))
    ap.add_argument("args", nargs="+")
    ap.add_argument("--previous", type=Path)
    ap.add_argument("--weights-dir", type=Path, default=N.WEIGHTS_DIR)
    a = ap.parse_args()
    if a.cmd == "plan":
        p = plan(load_freeze(Path(a.args[0])), int(a.args[1]), a.previous, a.weights_dir)
        print("\n".join(f"{k}={shlex.quote(str(v))}" for k, v in p.items()))
    else:
        top, f, idx, wdir = Path(a.args[0]), load_freeze(Path(a.args[1])), int(a.args[2]), Path(a.args[3])
        rec = verify(top, f, idx, wdir, a.previous)
        print(json.dumps({k: rec[k] for k in ("index", "status", "trained")}))


if __name__ == "__main__":
    main()
