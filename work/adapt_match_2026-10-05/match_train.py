"""ADAPT-MATCH-0 trainer: AO_L and BARROW_L, the public ADAF AO and ADAF barrow HRNets continued on ADAPT-NDS-1's
Lower Saxony FIT windows with the J recipe (PACKET.md "Arms").

Reused unchanged (imported read-only, as ADAPT-NDS-1's nds_train imports them): nds_train.update (adapt_train.update
with nds_schedule.lr_at: J's cosine 1e-4 -> 1e-6, no warm-up, compressed to 5,116 updates; r2_train.masked_loss =
BCE + 1 - soft Dice over non-IGNORE pixels; explicit device), nds_train.ls_rows (the Lower Saxony AdaptSteps rows: one
positive window per register group, background windows), nds_train.check_constants; adapt_train AdaptSteps + loader
(void policy V, 512 px crops, +-160 px offsets, dihedral, 3 workers); r2_train.load_model (official AiTLAS HRNet with a
2-channel HRNetSegHead, strict state dict: the loader of every J and ADAF inference), r2_train.set_train_mode (frozen
BatchNorm, batch 1), r2_train.WD and plans.LR_MAX (a fresh AdamW, as J's continuation resets it); paired_train.save;
HR1 welsh_train._resume / env_record / step0 (through JOINT-1 joint_train); JOINT-1 joint_train.staged / torch_flags.
Differences from A, all from the packet: the start (the arm's pinned ADAF tar, size and sha256 checked) and the plan
(match_schedule.local_plan: every update a Lower Saxony step, no English, Welsh or Irish replay). Batch 1, fp32, TF32
off, torch.manual_seed(seed), final weights only (no checkpoint selection, no retention pass).

  python -B match_train.py plan-check SEED OUT.json --ls-table T [--ls-win W --ls-mask M]     (CPU; no forward)
  python -B match_train.py step0 OUT.json            # HR1/JOINT-1 step-0 environment gate (CONT e41, declared reference)
  python -B match_train.py train SEED ARM --ls-table T --ls-win W --ls-mask M --run-root R
  python -B match_train.py smoke ARM N OUT.json --ls-table T --ls-win W --ls-mask M --scratch S [--device cpu]
                                 [--run-root R]                                                  (staged windows only)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import match_common as M  # noqa: E402
import match_schedule as MS  # noqa: E402

N = M.N
import nds_schedule as SCH  # noqa: E402  frozen (ADAPT-NDS-1)
import nds_train as NT  # noqa: E402  frozen ADAPT-NDS-1 trainer: update, ls_rows, check_constants
from nds_train import JT1, at, plans, pt, rt, wt  # noqa: E402  frozen recipe modules as ADAPT-NDS-1 imports them

LS_REGION = NT.LS_REGION


# ---------------------------------------------------------------- inputs
def check_run(seed: int, arm: str) -> str:
    return M.run_name(arm, seed)


def ls_table(path: Path | str) -> dict:
    t = N.read_json(path)
    if t.get("role") != "FIT" or "Lower Saxony FIT" not in t.get("region", ""):
        raise N.RefusedTerrain("the Lower Saxony table is not ADAPT-NDS-1's FIT table")
    return t


def ls_context(paths: dict) -> dict:
    """AdaptSteps rows of the Lower Saxony FIT windows (nds_train.ls_rows, row offset 0) with the pools of A."""
    t = ls_table(paths["ls_table"])
    rows, pos, bg = NT.ls_rows(t, Path(paths.get("ls_win") or "ls_windows"), Path(paths.get("ls_mask") or "ls_masks"),
                               0)
    if len(set(pos)) != len(pos) or not pos or not bg:
        raise SystemExit("Lower Saxony pools empty or a positive window repeats (one window per group expected)")
    if len({r["uid"] for r in rows}) != len(rows):
        raise SystemExit("duplicate Lower Saxony window")
    return {"rows": rows, "pos": pos, "bg": bg}


def schedule(seed: int, epoch: int, ctx: dict) -> dict:
    return MS.local_plan(seed, epoch, ctx["pos"], ctx["bg"])


def draw_digest(wmap: dict, rows: list[dict]) -> str:
    """nds_train's per-epoch draw digest form: step -> [window uid, dy, dx, k]."""
    return N.digest({str(s): [rows[v[0]]["uid"], *v[1:]] for s, v in sorted(wmap.items())})


def role_counts(wmap: dict, rows: list[dict]) -> dict:
    out = {"positive": 0, "background": 0}
    for v in wmap.values():
        out[rows[v[0]]["role"]] += 1
    return out


def plan_summary(seed: int, ctx: dict) -> dict:
    out = {"seed": seed, "epochs": []}
    for e in MS.EPOCHS:
        wmap = schedule(seed, e, ctx)
        out["epochs"].append({"epoch": e, "updates": role_counts(wmap, ctx["rows"]), "lower_saxony_updates": len(wmap),
                              "draw_digest": draw_digest(wmap, ctx["rows"])})
    return out


# ---------------------------------------------------------------- model and passes
def start_weights(arm: str) -> Path:
    s = M.STARTS[arm]
    p = M.start_path(arm)
    if not p.is_file() or p.stat().st_size != s["bytes"] or N.sha256_file(p) != s["sha256"]:
        raise SystemExit(f"start weights of {arm} missing or differ from the registry ({s['path']})")
    return p


def fresh_model(arm: str, device: str = "cuda"):
    """The arm's ADAF tar through r2_train.load_model, train mode with frozen BatchNorm, a fresh AdamW (lr plans.LR_MAX,
    weight decay r2_train.WD): nds_train.fresh_model with the start replaced."""
    import torch
    model = rt.load_model(start_weights(arm), device=device)
    rt.set_train_mode(model)
    return model, torch.optim.AdamW(model.parameters(), lr=plans.LR_MAX, weight_decay=rt.WD)


def new_sums() -> dict:
    return {"positive_loss": 0.0, "background_loss": 0.0, "n_positive": 0, "n_background": 0, "skipped": 0}


def epoch_pass(arm: str, seed: int, epoch: int, c: dict, name: str, started: float, device: str = "cuda",
               limit: int | None = None) -> dict:
    """One epoch of Lower Saxony updates (nds_train.epoch_pass with the all-local map)."""
    t0 = time.monotonic()
    rows = c["ctx"]["rows"]
    wmap = schedule(seed, epoch, c["ctx"])
    n = MS.STEPS_PER_EPOCH if limit is None else limit
    wmap = {s: v for s, v in wmap.items() if s < n}
    if set(wmap) != set(range(n)):
        raise SystemExit("the all-local plan does not cover every step")
    sums, losses = new_sums(), []
    for s, x, y, _ in at.loader(at.AdaptSteps([], [None] * n, rows, wmap)):
        role = rows[wmap[int(s)][0]]["role"]
        r = NT.update(c["model"], c["opt"], SCH.t_index(epoch, int(s)), x, y, device)
        losses.append(None if r["skipped"] else r["loss"])
        if r["skipped"]:
            sums["skipped"] += 1
        else:
            sums[f"{role}_loss"] += r["loss"]
            sums[f"n_{role}"] += 1
        if int(s) % at.LOG_EVERY == 0:
            print(json.dumps({"run": name, "epoch": epoch, "step": int(s), "role": role, **r,
                              "elapsed_s": round(time.monotonic() - started, 1)}), flush=True)
    row = {"epoch": epoch, "lr_end": c["opt"].param_groups[0]["lr"], "skipped_all_ignore": sums["skipped"],
           "epoch_s": time.monotonic() - t0, "updates": role_counts(wmap, rows), "lower_saxony_updates": len(wmap),
           **{f"mean_{k}_loss": (sums[f"{k}_loss"] / sums[f"n_{k}"] if sums[f"n_{k}"] else None) for k in MS.ROLES},
           "draw_digest": draw_digest(wmap, rows)}
    row["losses_digest"] = N.digest(losses) if limit is not None else None
    c["losses"] = losses
    return row


def save_final(c: dict, run: Path, arm: str, seed: int, epoch: int, **extra) -> str:
    final = run / "final.pt"
    pt.save({"state_dict": c["model"].state_dict(), "epoch": epoch, "seed": seed, "arm": arm,
             "start": M.STARTS[arm]["sha256"], **extra}, final)
    return rt.sha(final)


# ---------------------------------------------------------------- training (VM, GPU)
def train_context(paths: dict) -> dict:
    NT.check_constants()
    return {"ctx": ls_context(paths)}


def train(seed: int, arm: str, paths: dict, run_root: Path) -> dict:
    """Resumable per epoch (last.pt), as nds_train.train; no English retention pass (not a packet measure)."""
    import torch
    name = check_run(seed, arm)
    c = train_context(paths)
    run = run_root / name
    run.mkdir(parents=True, exist_ok=True)
    last, log_path = run / "last.pt", run / "epochs.jsonl"
    if (run / "TRAIN_RESULT.json").exists():
        return N.read_json(run / "TRAIN_RESULT.json")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    c["model"], c["opt"] = fresh_model(arm)
    torch.manual_seed(seed)
    done = wt._resume(c["model"], c["opt"], last)
    for epoch in MS.EPOCHS:
        if epoch < done:
            continue
        row = epoch_pass(arm, seed, epoch, c, name, started)
        if epoch == MS.EPOCHS[-1]:
            row["final_sha256"] = save_final(c, run, arm, seed, epoch)
        pt.save({"state_dict": c["model"].state_dict(), "optimizer": c["opt"].state_dict(), "epoch": epoch,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}, last)
        with log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({"run": name, **row}), flush=True)
    rows_log = [json.loads(x) for x in log_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    result = result_record(arm, seed, run, rows_log, c, paths, started)
    N.write_json(run / "TRAIN_RESULT.json", result)
    last.unlink(missing_ok=True)
    return result


def result_record(arm: str, seed: int, run: Path, rows_log: list, c: dict, paths: dict, started: float) -> dict:
    import torch
    final = run / "final.pt"
    return {"run": run.name, "arm": arm, "seed": seed, "status": "COMPLETE", "final": str(final),
            "final_sha256": rt.sha(final), "final_bytes": final.stat().st_size, "epochs": rows_log,
            "plan_summary": plan_summary(seed, c["ctx"]),
            "start": {"registry_id": M.STARTS[arm]["registry_id"], "path": M.STARTS[arm]["path"],
                      "sha256": M.STARTS[arm]["sha256"]},
            "updates": MS.N_UPDATES, "lower_saxony_updates": MS.N_UPDATES,
            "lr": "cosine 1e-4 -> 1e-6 over 5,116 updates (J's, compressed; no warm-up; nds_schedule.lr_at)",
            "tables_sha256": {"ls_table": N.sha256_file(paths["ls_table"])},
            "script_sha256": N.sha256_file(Path(__file__)), "schedule_sha256": N.sha256_file(Path(MS.__file__)),
            "nds_train_sha256": N.sha256_file(Path(NT.__file__)), "nds_schedule_sha256": N.sha256_file(Path(SCH.__file__)),
            "frozen_trainer_sha256": N.sha256_file(Path(at.__file__)), "wall_s": time.monotonic() - started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
            "attribution": M.ATTRIBUTION, **(wt.env_record() if torch.cuda.is_available() else {})}


# ---------------------------------------------------------------- smoke (staged windows; GPU locally or CPU)
def staged_table(paths: dict) -> Path:
    t = JT1.staged(ls_table(paths["ls_table"]), Path(paths["ls_win"]), Path(paths["ls_mask"]))
    if not t["positive_windows"] or not t["background_windows"]:
        raise SystemExit("smoke: no staged Lower Saxony positive or background window")
    p = Path(paths["scratch"]) / "STAGED_ls_table.json"
    N.write_json(p, t, indent=None)
    return p


def smoke(arm: str, n: int, out: Path, paths: dict, device: str = "cuda", run_root: Path | None = None) -> dict:
    """First n updates of epoch 4 at seed 2026092802 on the staged windows; with run_root the model is saved as
    RUN_ROOT/SMOKE_<arm>/final.pt (train format) for the artifact route."""
    import torch
    seed = M.SEEDS[0]
    NT.check_constants()
    sp = dict(paths, ls_table=staged_table(paths))
    c = {"ctx": ls_context(sp)}
    torch.manual_seed(seed)
    c["model"], c["opt"] = fresh_model(arm, device)
    t0 = time.monotonic()
    row = epoch_pass(arm, seed, MS.EPOCHS[0], c, f"SMOKE_{arm}", t0, device=device, limit=n)
    rec = {"arm": arm, "seed": seed, "n": n, "device": device, "row": row, "losses": c["losses"],
           "s_per_update": (time.monotonic() - t0) / max(n, 1), "start_sha256": M.STARTS[arm]["sha256"],
           "staged_windows": len(c["ctx"]["rows"]), "all_finite": all(v is None or math.isfinite(v) for v in c["losses"])}
    if run_root is not None:
        run = Path(run_root) / f"SMOKE_{arm}"
        run.mkdir(parents=True, exist_ok=True)
        rec["final_sha256"] = save_final(c, run, arm, seed, MS.EPOCHS[0], smoke_updates=n)
        rec["final_bytes"] = (run / "final.pt").stat().st_size
        N.write_json(run / "TRAIN_RESULT.json", {"run": run.name, "arm": arm, "seed": seed, "tag": "smoke",
                                                "status": "SMOKE", "final": str(run / "final.pt"),
                                                "final_sha256": rec["final_sha256"],
                                                "final_bytes": rec["final_bytes"]})
    N.write_json(out, rec)
    if not rec["all_finite"]:
        raise SystemExit("smoke: non-finite loss")
    return rec


# ---------------------------------------------------------------- plan-check (CPU, no forward)
def plan_check(seed: int, paths: dict, out: Path) -> dict:
    """The all-local plan of one seed on the real table: every step a Lower Saxony step, positive at 2p and background
    at 2p + 1, each complete pass a permutation of the pool, exposure as the closed form, pass 0 in A's Lower Saxony
    order (nds_schedule.ls_plan, frozen), update indices 0..5,115 once, AO_L and BARROW_L plans identical."""
    if seed not in M.SEEDS:
        raise SystemExit(f"plan-check: seed {seed} outside the design")
    NT.check_constants()
    ctx = ls_context(paths)
    rows, pool, bg = ctx["rows"], ctx["pos"], ctx["bg"]
    plan = MS.run_plan(seed, pool, bg)
    bad = []
    for e in MS.EPOCHS:
        wmap = plan[e]
        if set(wmap) != set(range(MS.STEPS_PER_EPOCH)):
            bad.append(f"epoch {e}: not every step is a Lower Saxony step")
        wrong = [s for s, v in wmap.items() if rows[v[0]]["role"] != MS.ROLES[s % 2] or rows[v[0]]["region"] != LS_REGION]
        bad += [f"epoch {e}: {len(wrong)} steps break the positive/background alternation"] if wrong else []
        off = [s for s, v in wmap.items() if not (-SCH.OFFSET_PX <= v[1] <= SCH.OFFSET_PX and
                                                  -SCH.OFFSET_PX <= v[2] <= SCH.OFFSET_PX and 0 <= v[3] < 8)]
        bad += [f"epoch {e}: {len(off)} offsets or dihedral outside r2_train.draw"] if off else []
    t_idx = sorted(SCH.t_index(e, s) for e in MS.EPOCHS for s in plan[e])
    bad += [] if t_idx == list(range(MS.N_UPDATES)) else ["update indices are not 0..5,115 once each"]
    bad += MS.pass_complete(plan, pool, bg)
    exp = MS.exposure(plan, pool, bg)
    want = MS.expected_exposure(len(pool), len(bg))
    bad += [f"{r}: exposure {exp[r]['histogram']} != {want[r]}" for r in MS.ROLES if exp[r]["histogram"] != want[r]]
    ours = [MS.pair_windows(seed, n, pool, bg) for n in range(SCH.ls_pair_count())]
    a_seq = MS.a_order(seed, pool, bg)
    bad += [] if ours == a_seq else ["pass 0 differs from A's Lower Saxony order"]
    summ = {arm: plan_summary(seed, ctx) for arm in M.ARMS}
    if summ["AO_L"] != summ["BARROW_L"]:
        bad.append("AO_L and BARROW_L plans differ")
    rec = {"schema": "adapt_match_plan_check_v1", "seed": seed, "status": "FAIL" if bad else "PASS", "violations": bad,
           "updates": MS.N_UPDATES, "pairs": MS.total_pairs(), "pool": {"positive": len(pool), "background": len(bg)},
           "passes": {r: divmod(MS.total_pairs(), size) for r, size in (("positive", len(pool)), ("background", len(bg)))},
           "exposure": exp, "a_exposure": {"positive": {"1": len(pool)}, "background": {"1": len(bg)}},
           "a_pairs_lower_saxony": SCH.ls_pair_count(), "pass0_equals_a_order": ours == a_seq,
           "lr": MS.lr_trace(), "plan_summary": summ["AO_L"],
           "plan_digest": N.digest([e["draw_digest"] for e in summ["AO_L"]["epochs"]]),
           "ls_table_sha256": N.sha256_file(paths["ls_table"]), "utc": N.utc(), "attribution": M.ATTRIBUTION}
    N.write_json(out, rec)
    if bad:
        raise SystemExit(f"plan-check failed: {bad}")
    return rec


# ---------------------------------------------------------------- CLI
def parse_paths(a) -> dict:
    return {k: getattr(a, k) for k in ("ls_table", "ls_win", "ls_mask", "scratch") if getattr(a, k, None) is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan-check", "step0", "train", "smoke"))
    ap.add_argument("args", nargs="*")
    for k in ("ls-table", "ls-win", "ls-mask", "run-root", "scratch"):
        ap.add_argument(f"--{k}", type=Path)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd in ("step0", "train", "smoke") and a.device == "cuda":
        JT1.J.torch_flags()
    paths = parse_paths(a)
    if a.cmd == "plan-check":
        r = plan_check(int(a.args[0]), paths, Path(a.args[1]))
        print(json.dumps({k: r[k] for k in ("seed", "status", "exposure", "pass0_equals_a_order", "plan_digest")}))
    elif a.cmd == "step0":
        print(json.dumps(JT1.step0(Path(a.args[0]))))
    elif a.cmd == "train":
        r = train(int(a.args[0]), a.args[1], paths, a.run_root)
        print(json.dumps({k: r[k] for k in ("run", "status", "final_sha256", "wall_s")}))
    else:
        if "scratch" not in paths:
            paths["scratch"] = Path(a.args[2]).parent
        r = smoke(a.args[0], int(a.args[1]), Path(a.args[2]), paths, a.device, a.run_root)
        print(json.dumps({k: r[k] for k in ("arm", "n", "device", "s_per_update", "all_finite")}))


if __name__ == "__main__":
    main()
