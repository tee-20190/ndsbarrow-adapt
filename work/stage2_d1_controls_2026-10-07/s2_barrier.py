"""STAGE2-D1-CONTROLS technical checks of a finished control run: the first-run barrier and the per-run verification
(ROUND4_REPLY.md R4-2 "Use one session with an explicit first-run barrier"; ROUND4_REVIEW.md "Adopted" item 3).

No archaeological score is computed here or anywhere in the chain. A technically valid checkpoint that predicts only
background passes: these checks concern identities, executed updates, target handling, checkpoint integrity and
readability, never recall.

  vm-check NAME RUN_DIR PINS OUT.json [--smoke N] [--start-diff] [--device D] [--w-win ...]   (VM, fresh process)
      identities  every trainer / schedule / recipe module and every table the run read equals the session pins
                  (PINS_SESSION.json: repo path -> sha256) and the hashes recorded in TRAIN_RESULT.json; the bundle
                  manifests (MANIFEST_code / MANIFEST_english) are hashed and every English raw window and mask the run
                  read is listed there with its on-disk sha256; the Welsh / Irish files it read are hashed (digest
                  recorded); the plan summary re-derives from the pinned tables (run mode)
      updates     curves.jsonl holds t = 0 .. U-1 once each (U = 5,116; smoke: N), roles alternate positive /
                  background, region counts equal arm C's, every lr = d1_train.lr_at("F1", t) exactly, and the AdamW
                  step counter = U - skipped all-IGNORE updates (expected 0, as in D1)
      targets     F1 target handling on every update: Dice used iff the valid target has a positive pixel; the
                  update's target counts (sum_g, n_valid) are bounded by the staged mask's crop at the same (dy, dx)
                  (equal unless void policy V removed pixels); per-role empty-target share and foreground-fraction
                  quantiles recorded
      checkpoint  sha256 and size = TRAIN_RESULT.json; reload in this fresh process with the inference loader
                  (adaf_configs.load_models, strict); every tensor finite; with --start-diff, parameters differ from
                  the ADAF barrow start (updates executed)
      fixture     the reloaded model on FIXTURE (four English R2 FIT windows, fixed rule: the first two positive and
                  the first two negative non-holdout windows in id order) through the inference path
                  (tta_run.objects_multi): every probability finite and in [0, 1]; map sha256 and object counts
                  recorded (no score)
      environment python, numpy, torch, CUDA, GPU, TF32 / cuDNN flags
  wait-ack NAME TRAIN_RESULT ACK TIMEOUT_S OUT.json                                            (VM, barrier only)
      durable readback: waits for the driver's ACK (uploaded only after the local copy under outputs/.../weights
      was re-read from disk and its sha256 / size matched); refuses a mismatching ACK; times out -> barrier FAIL
  ack RUN WEIGHTS_DIR TRAIN_RESULT OUT.json                                                     (local, driver)
      re-reads the local final.pt from disk, recomputes sha256 and size, compares them with VERIFIED.json and
      TRAIN_RESULT.json, writes the ACK
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2_common as S  # noqa: E402

N = S.N
ROOT = S.ROOT
FIXTURE_N = {"positive": 2, "negative": 2}
QUANT = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
INFER_PATHS = ("work/campaign_2026-09-22", "work/tta_ensemble_2026-09-24", "work/aim_train_audit_2026-09-24")


def _rel(p: Path) -> str:
    return N.rel(p) if Path(p).resolve().is_relative_to(ROOT) else str(p)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- identities
def module_files() -> dict[str, Path]:
    import d1_train as T1
    import nds_schedule as SCH
    import nds_train as NT
    import s2_schedule as SS
    import s2_train as T
    from nds_train import at
    return {"script_sha256": Path(T.__file__), "schedule_sha256": Path(SS.__file__), "common_sha256": Path(S.__file__),
            "d1_train_sha256": Path(T1.__file__), "nds_train_sha256": Path(NT.__file__),
            "nds_schedule_sha256": Path(SCH.__file__), "frozen_trainer_sha256": Path(at.__file__)}


def table_files(seed: int | None) -> dict[str, Path]:
    import s2_schedule as SS
    out = {"w_table": SS.J.W_FIT, "i_table": SS.J.I_FIT, "english_manifest": SS.plans.MANIFEST,
           "subsets_s2": S.SUBSETS_S2}
    if seed is not None:
        out["english_plan"] = S.english_plan_path(seed)
    return out


def identities(tr: dict, pins: dict, seed: int | None, v: list) -> dict:
    out = {}
    for key, p in module_files().items():
        h, rel = sha(p), N.rel(p)
        out[rel] = h
        if pins.get(rel) != h:
            v.append(f"{rel}: differs from the session pins")
        if seed is not None and tr.get(key) != h:
            v.append(f"{rel}: differs from TRAIN_RESULT.json {key}")
    for key, p in table_files(seed).items():
        h, rel = sha(p), N.rel(p)
        out[rel] = h
        want = pins.get(rel) or bundle_manifest("code").get(rel) or bundle_manifest("english").get(rel)
        if want != h:
            v.append(f"{rel}: data manifest / table differs from the session pins and bundle manifests")
        if seed is not None and tr.get("tables_sha256", {}).get(key) != h:
            v.append(f"{key}: differs from TRAIN_RESULT.json tables_sha256")
    for b in ("code", "english"):
        m = ROOT / f"MANIFEST_{b}.sha256"
        out[f"MANIFEST_{b}.sha256"] = sha(m) if m.is_file() else None
    return out


def bundle_manifest(name: str = "english") -> dict[str, str]:
    """repo path -> sha256 from the VM's MANIFEST_<name>.sha256 (written by bundle_stream.py, verified at unpack)."""
    out = {}
    m = ROOT / f"MANIFEST_{name}.sha256"
    if m.is_file():
        for ln in m.read_text(encoding="utf-8").splitlines():
            h, _, rel = ln.partition("  ")
            out[rel.strip().lstrip("./").lstrip("*")] = h
    return out


def data_identity(steps: list[dict], windows: list[dict], v: list) -> dict:
    """English raw windows + masks read by the run: listed in the english bundle manifest with their on-disk sha256;
    Welsh / Irish files: hashed (digest of uid -> (tif sha, mask sha))."""
    import s2_schedule as SS
    man = bundle_manifest()
    byid = {w["id"]: w for w in windows}
    eng = sorted({r["uid"].split(":", 1)[1] for r in steps if r["kind"] == "english"})
    unlisted = bad = 0
    for wid in eng:
        for p in (SS.rt.RAW / f"{wid}.tif", ROOT / byid[wid]["mask_path"]):
            rel = N.rel(p)
            h = sha(p)
            if man and rel not in man:
                unlisted += 1
            elif man and man[rel] != h:
                bad += 1
    if bad or unlisted:
        v.append(f"English data identity: {bad} files differ from and {unlisted} are missing in MANIFEST_english")
    reg = {}
    for r in steps:
        if r["kind"] in ("welsh", "irish") and r["uid"] not in reg:
            m = Path(r["mask"])
            reg[r["uid"]] = sha(m if m.is_absolute() else ROOT / m)
    return {"english_windows_read": len(eng), "english_manifest_entries": len(man),
            "english_files_checked": 2 * len(eng), "regional_windows_read": len(reg),
            "regional_masks_digest": N.digest(dict(sorted(reg.items())))}


# ---------------------------------------------------------------- updates and targets
def curves(run: Path) -> list[dict]:
    return [json.loads(x) for x in (run / "curves.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def _q(x: list[float]) -> dict:
    import numpy as np
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in QUANT} if x else {}


def update_checks(recs: list[dict], tr: dict, n_updates: int, smoke: bool, v: list) -> dict:
    import d1_train as T1
    import nds_schedule as SCH
    ts = [r["t"] for r in recs]
    if sorted(ts) != list(range(n_updates)) or len(ts) != n_updates:
        v.append(f"update indices are not 0..{n_updates - 1} once each ({len(ts)} records)")
    alt = sum(1 for r in recs if r["role"] != ("positive", "background")[r["step"] % 2])
    lr_bad = sum(1 for r in recs if r["lr"] != T1.lr_at(S.D1_RECIPE, r["t"]))
    v += [f"{alt} updates break the positive/background alternation"] if alt else []
    v += [f"{lr_bad} updates ran off the F1 learning-rate schedule"] if lr_bad else []
    kinds = {k: sum(1 for r in recs if r["kind"] == k) for k in SCH.KINDS}
    if not smoke and kinds != SCH.kind_counts("C")["updates"]:
        v.append(f"region counts {kinds} differ from arm C's")
    skipped = sum(1 for r in recs if r.get("skipped"))
    oc = tr.get("optimizer_step_counter", {})
    if oc.get("max") != n_updates - skipped or oc.get("min") != n_updates - skipped:
        v.append(f"AdamW step counter {oc} != {n_updates} updates - {skipped} skipped")
    return {"update_records": len(recs), "skipped_all_ignore": skipped, "optimizer_step_counter": oc,
            "lr_first": recs[0]["lr"] if recs else None, "lr_last": recs[-1]["lr"] if recs else None,
            "lr_schedule_exact": lr_bad == 0, "kind_updates": kinds}


def target_checks(recs: list[dict], steps: list[dict] | None, v: list) -> dict:
    """F1 target handling + per-role crop statistics; with steps, bounds against the staged masks' crops."""
    import numpy as np
    import s2_schedule as SS
    wrong_mode = sum(1 for r in recs if not r.get("skipped") and bool(r["dice_used"]) != (r["sum_g"] > 0))
    loss_bad = sum(1 for r in recs if not r.get("skipped") and
                   abs(r["loss_used"] - (r["bce"] + ((1 - r["dice"]) if r["dice_used"] else 0.0))) > 1e-5)
    v += [f"{wrong_mode} updates used Dice against F1's target rule"] if wrong_mode else []
    v += [f"{loss_bad} updates' loss differs from F1's form"] if loss_bad else []
    stats = {}
    for role in ("positive", "background"):
        rr = [r for r in recs if r["role"] == role and not r.get("skipped")]
        fg = [r["sum_g"] / r["n_valid"] for r in rr if r["n_valid"] > 0]
        stats[role] = {"updates": len(rr), "empty_target": sum(1 for r in rr if r["sum_g"] == 0),
                       "empty_target_share": (sum(1 for r in rr if r["sum_g"] == 0) / len(rr)) if rr else None,
                       "fg_fraction_mean": float(np.mean(fg)) if fg else None, "fg_fraction_quantiles": _q(fg)}
    out = {"dice_rule_violations": wrong_mode, "loss_form_violations": loss_bad, "crop_stats": stats}
    if steps is not None:
        by_t = {r["t"]: r for r in steps}
        over = equal = 0
        cache: dict = {}
        for r in recs:
            st = by_t[r["t"]]
            m = Path(st["mask"]) if Path(st["mask"]).is_absolute() else ROOT / st["mask"]
            if m not in cache:
                cache = {m: np.load(m)["label"]} if len(cache) > 256 else {**cache, m: np.load(m)["label"]}
            c = SS.rt.crop(cache[m], st["dy"], st["dx"])
            nv, npos = int((c != SS.rt.IGNORE).sum()), int((c == 1).sum())
            if r.get("skipped"):
                continue
            if r["n_valid"] > nv or r["sum_g"] > npos:
                over += 1
            equal += int(r["n_valid"] == nv and r["sum_g"] == npos)
        if over:
            v.append(f"{over} updates' targets exceed the staged mask crop at their (dy, dx): wrong data")
        out |= {"mask_bound_violations": over, "updates_equal_to_mask_crop": equal}
    return out


# ---------------------------------------------------------------- checkpoint, reload, fixture
def infer_modules():
    for p in INFER_PATHS:
        if str(ROOT / p) not in sys.path:
            sys.path.insert(0, str(ROOT / p))
    import tta_run as tr
    return tr


def fixture_windows() -> list[str]:
    import s2_schedule as SS
    windows = SS.plans.load_windows()[0]
    out = []
    for role, n in FIXTURE_N.items():
        out += sorted(w["id"] for w in windows if w["role"] == role and not w["holdout"])[:n]
    return out


def predictor(models: dict, device: str):
    """holdout_eval.predictor's op sequence with an explicit device (the VM uses cuda)."""
    import numpy as np
    import torch
    seen = {"patches": 0, "nonfinite": 0, "out_of_range": 0}

    def predict(a: np.ndarray) -> dict[str, np.ndarray]:
        x = torch.tensor(np.repeat(a[None, :, :], 3, axis=0), dtype=torch.float32, device=device)[None]
        with torch.inference_mode():
            out = {k: torch.sigmoid(m(x)["out"])[0, 1].cpu().numpy() for k, m in models.items()}
        for y in out.values():
            seen["patches"] += 1
            seen["nonfinite"] += int((~np.isfinite(y)).sum())
            seen["out_of_range"] += int(((y < 0) | (y > 1)).sum())
        return out
    return predict, seen


def checkpoint_checks(run: Path, tr: dict, device: str, start_diff: bool, v: list) -> dict:
    import torch
    f = run / "final.pt"
    h, size = sha(f), f.stat().st_size
    if h != tr.get("final_sha256") or size != tr.get("final_bytes", size):
        v.append("final.pt differs from TRAIN_RESULT.json (sha256 / size)")
    state = torch.load(f, map_location="cpu", weights_only=True)["state_dict"]
    nonfinite = [k for k, t in state.items() if t.is_floating_point() and not bool(torch.isfinite(t).all())]
    v += [f"{len(nonfinite)} checkpoint tensors are not finite"] if nonfinite else []
    out = {"sha256": h, "bytes": size, "tensors": len(state), "nonfinite_tensors": len(nonfinite)}
    if start_diff:
        import match_train as MT
        import s2_schedule as SS
        start = SS.rt.load_model(MT.start_weights(S.ARM), device="cpu").state_dict()
        changed = sum(1 for k, t in state.items() if t.is_floating_point() and k in start
                      and not torch.equal(t, start[k].to(t.dtype)))
        out["tensors_changed_from_start"] = changed
        if changed == 0:
            v.append("no parameter differs from the ADAF barrow start: updates were not executed")
        del start
    del state
    tr_mod = infer_modules()
    tr_mod.ac.check_frozen()
    models, _ = tr_mod.ac.load_models([f], device=device)
    out["reload"] = "adaf_configs.load_models strict=True"
    out["fixture"] = fixture(models[0], device, v)
    return out


def fixture(model, device: str, v: list) -> dict:
    import s2_schedule as SS
    tr_mod = infer_modules()
    predict, seen = predictor({"m": model}, device)
    rows = []
    for wid in fixture_windows():
        rec = tr_mod.objects_multi(SS.rt.RAW / f"{wid}.tif", predict, ["m"])
        rows.append({"map_sha256": rec["arms"]["m"]["map_sha256"], "objects": len(rec["arms"]["m"]["objects"]),
                     "patches": rec["patches"]})
    if seen["nonfinite"] or seen["out_of_range"] or not seen["patches"]:
        v.append(f"fixture outputs invalid: {seen}")
    return {"windows": len(rows), "rule": "first 2 positive + first 2 negative non-holdout English FIT windows by id",
            "outputs": rows, "probabilities": seen,
            "fixture_digest": N.digest([r["map_sha256"] for r in rows])}


def environment() -> dict:
    import numpy as np
    import torch
    env = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
           "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark,
           "cudnn_deterministic": torch.backends.cudnn.deterministic, "precision": "fp32"}
    return env


# ---------------------------------------------------------------- the commands
def run_steps(seed: int, paths: dict) -> tuple[list[dict], list[dict], dict]:
    import s2_train as T
    c = T.context(seed, paths)
    steps = [r for e in S.EPOCHS for r in T.epoch_steps(seed, e, c)[2]]
    return steps, c["windows"], T.plan_summary(seed, c)


def vm_check(name: str, run: Path, pins_path: Path, out: Path, smoke: int | None, start_diff: bool, device: str,
             paths: dict | None, data_ident: bool = False) -> dict:
    t0 = time.monotonic()
    v: list[str] = []
    tr = N.read_json(run / "TRAIN_RESULT.json")
    pins = N.read_json(pins_path)["files"]
    seed = None if smoke else int(tr.get("seed", -1))
    if smoke:
        if tr.get("status") != "SMOKE" or tr.get("seed") != S.SMOKE_SEED:
            v.append("smoke TRAIN_RESULT is not the smoke of the first control seed")
    else:
        if name != S.run_name(seed) or tr.get("status") != "COMPLETE" or tr.get("recipe") != S.D1_RECIPE or \
                tr.get("updates") != S.N_UPDATES or tr.get("local_labels_read") is not False or \
                tr.get("lower_saxony_updates") != 0 or tr.get("start", {}).get("sha256") != S.START["sha256"]:
            v.append("TRAIN_RESULT is not a complete F1 control from ADAF barrow without local labels")
    if N.sha256_file(run / "curves.jsonl") != tr.get("curves_sha256"):
        v.append("curves.jsonl differs from TRAIN_RESULT.json")
    rec: dict = {"schema": "s2_barrier_vm_v1", "run": name, "mode": "smoke" if smoke else "run"}
    rec["identities"] = identities(tr, pins, seed, v)
    steps = None
    if not smoke and paths:
        steps, windows, ps = run_steps(seed, paths)
        if ps != tr.get("plan_summary"):
            v.append("plan summary does not re-derive from the pinned tables")
        if data_ident:
            rec["data_identity"] = data_identity(steps, windows, v)
    recs = curves(run)
    rec["updates"] = update_checks(recs, tr, smoke or S.N_UPDATES, bool(smoke), v)
    rec["targets"] = target_checks(recs, steps, v)
    rec["checkpoint"] = checkpoint_checks(run, tr, device, start_diff, v)
    rec["environment"] = environment()
    rec["recorded_environment_train"] = {k: tr.get(k) for k in ("torch", "cuda", "gpu", "nvidia_tf32_override",
                                                                "matmul_allow_tf32", "cudnn_allow_tf32",
                                                                "cudnn_benchmark", "cudnn_deterministic")}
    rec |= {"status": "FAIL" if v else "PASS", "violations": v, "wall_s": round(time.monotonic() - t0, 1),
            "utc": N.utc(), "note": "technical checks only; no archaeological score"}
    N.write_json(out, rec)
    if v:
        raise SystemExit(f"barrier check of {name} FAILED: {v[:4]}")
    return rec


def wait_ack(name: str, tr_path: Path, ack: Path, timeout_s: float, out: Path, poll_s: float = 5.0) -> dict:
    tr = N.read_json(tr_path)
    t0 = time.monotonic()
    while not ack.is_file() and time.monotonic() - t0 < timeout_s:
        time.sleep(poll_s)
    rec = {"schema": "s2_barrier_ack_v1", "run": name, "waited_s": round(time.monotonic() - t0, 1), "utc": N.utc()}
    v = []
    if not ack.is_file():
        v.append(f"no durable-readback ACK within {timeout_s} s")
    else:
        a = json.loads(ack.read_text(encoding="utf-8"))
        rec["ack"] = a
        if a.get("status") != "PASS" or a.get("run") != name or a.get("readback_sha256") != tr.get("final_sha256") \
                or a.get("readback_bytes") != tr.get("final_bytes"):
            v.append("ACK does not confirm a local readback equal to TRAIN_RESULT.json")
    rec |= {"status": "FAIL" if v else "PASS", "violations": v}
    N.write_json(out, rec)
    if v:
        raise SystemExit(f"durable readback of {name} FAILED: {v}")
    return rec


def ack(run: str, weights_dir: Path, tr_path: Path, out: Path) -> dict:
    d = weights_dir / run
    ver = N.read_json(d / "VERIFIED.json")
    tr = N.read_json(tr_path)
    h, size = sha(d / "final.pt"), (d / "final.pt").stat().st_size        # independent re-read from local disk
    ok = h == ver.get("sha256") == tr.get("final_sha256") and size == ver.get("bytes") and \
        size == tr.get("final_bytes", size)
    rec = {"schema": "s2_barrier_local_ack_v1", "run": run, "status": "PASS" if ok else "FAIL",
           "readback_sha256": h, "readback_bytes": size, "verified_json_sha256": N.sha256_file(d / "VERIFIED.json"),
           "local_path": _rel(d / "final.pt"), "utc": N.utc()}
    N.write_json(out, rec)
    if not ok:
        raise SystemExit(f"local readback of {run} differs from VERIFIED.json / TRAIN_RESULT.json")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("vm-check", "wait-ack", "ack"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--smoke", type=int)
    ap.add_argument("--start-diff", action="store_true")
    ap.add_argument("--data-identity", action="store_true")
    ap.add_argument("--device", default="cuda")
    for k in ("w-win", "w-mask", "i-win", "i-mask"):
        ap.add_argument(f"--{k}", type=Path)
    a = ap.parse_args()
    x = a.args
    if a.cmd == "vm-check":
        if a.device == "cuda":
            import joint_common as JC
            JC.torch_flags()
        paths = None
        if a.w_win:
            import s2_schedule as SS
            paths = {**SS.replay_paths(), "w_win": a.w_win, "w_mask": a.w_mask, "i_win": a.i_win, "i_mask": a.i_mask}
        r = vm_check(x[0], Path(x[1]), Path(x[2]), Path(x[3]), a.smoke, a.start_diff, a.device, paths, a.data_identity)
        print(json.dumps({k: r[k] for k in ("run", "mode", "status", "wall_s")}))
    elif a.cmd == "wait-ack":
        print(json.dumps(wait_ack(x[0], Path(x[1]), Path(x[2]), float(x[3]), Path(x[4]))["status"]))
    else:
        print(json.dumps(ack(x[0], Path(x[1]), Path(x[2]), Path(x[3]))))


if __name__ == "__main__":
    main()
