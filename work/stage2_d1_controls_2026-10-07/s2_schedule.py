"""STAGE2-D1-CONTROLS step plans (CPU; no forward pass): the control's step map and the composition records.

The control's 5,116 updates are ADAPT-NDS-1 arm C's step map at the control seed (PACKET.md "Design"):
  English steps   plans.real_plan(seed, e, fit_pos, fit_neg) for e = 4, 5 (the generator of J's PLAN files; C's
                  ENGLISH_E45 files at seeds 02-04 are this function's output), on every step outside the regional map;
  Welsh pairs     HR1 A_W30 draws (welsh_train.welsh_plan) on pairs with (p - e) % 4 == 0, the seed's 30-monument subset;
  Irish pairs     HR3 A_I30 draw streams (joint_train.irish_plan) on pairs with (p - e) % 4 == 2, the seed's subset;
  composed by nds_train.schedule("C", ...) (nds_schedule.compose: C = J's map, no Lower Saxony row is ever added).
Pair p = (positive step 2p, background step 2p + 1) in every source; update index t = (e - 4) x 2,558 + step.

Subsets: JOINT-1 recorded the 30-monument Welsh and Irish subsets for seeds 2026092800-06 (SUBSETS_PRIVATE.json,
audit_mixture.subset: default_rng([seed, 30]).choice(sorted(monuments), 30, replace=False)). Seeds 07-14 are derived by
the same rule; subsets() fails closed unless the rule reproduces every recorded seed exactly.

D1's plan (for the side-by-side composition): match_schedule.run_plan on the Lower Saxony FIT table (D1's own plan-check).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

import s2_common as S

N = S.N
J = S.J                                                        # JOINT-1 joint_common (W_FIT, I_FIT, SUBSETS, K)
import match_schedule as MS  # noqa: E402  ADAPT-MATCH-0 (frozen): D1's all-local plan, ROLES, EPOCHS
import match_train as MT  # noqa: E402  ADAPT-MATCH-0 (frozen): ls_context
import nds_schedule as SCH  # noqa: E402  ADAPT-NDS-1 (frozen): t_index, kind_counts, N_UPDATES
import nds_train as NT  # noqa: E402  ADAPT-NDS-1 (frozen): region_context, schedule, kinds_of
from nds_train import JT1, plans, rt  # noqa: E402  frozen recipe modules as ADAPT-NDS-1 imports them

ROLES = MS.ROLES                                               # ("positive", "background"): even / odd steps
ENGLISH_ROLE = {"positive": "positive", "negative": "background"}
ENGLISH_GENERATOR = ("plans.real_plan(seed, epoch, fit_pos, fit_neg) (work/counterfactual_2026-09-28/plans.py, the "
                     "generator of J's PLAN files)")


# ---------------------------------------------------------------- subsets (JOINT-1 rule, extended to seeds 07-14)
def subset_rule(monuments: list, seed: int) -> list[str]:
    """audit_mixture.subset, verbatim expression (work/joint_campaign_2026-10-03/audit_mixture.py:38-40)."""
    rng = np.random.default_rng([seed, S.SUBSET_TAG])
    return sorted(str(x) for x in rng.choice(sorted(monuments), J.K, replace=False))


def subsets(w_table: dict, i_table: dict, recorded: dict | None = None) -> dict:
    """{"wales": {seed: [...]}, "ireland": {...}} for the recorded seeds 00-06 and the ten control seeds."""
    recorded = recorded if recorded is not None else N.read_json(J.SUBSETS)
    out: dict = {"wales": {}, "ireland": {}}
    for region, table in (("wales", w_table), ("ireland", i_table)):
        mons = sorted(table["monuments"])
        for seed in sorted(set(S.SUBSET_RECORDED_SEEDS) | set(S.SEEDS)):
            out[region][str(seed)] = subset_rule(mons, seed)
        for seed in S.SUBSET_RECORDED_SEEDS:
            if sorted(map(str, recorded[region][str(seed)])) != out[region][str(seed)]:
                raise SystemExit(f"subset rule does not reproduce JOINT-1's recorded {region} subset of {seed}")
    return out


# ---------------------------------------------------------------- English plans (C's generator, epochs 4-5)
def english_record(seed: int, fit_pos: list[int], fit_neg: list[int]) -> dict:
    return {"schema": "adapt_nds_english_e45_v1", "seed": seed, "generator": ENGLISH_GENERATOR,
            "plans_sha256": N.sha256_file(Path(plans.__file__)), "manifest_sha256": plans.MANIFEST_SHA256,
            "epochs": {str(e): [list(map(int, r)) for r in plans.real_plan(seed, e, fit_pos, fit_neg)]
                       for e in S.EPOCHS}}


def english_real(seed: int, fit_pos: list[int], fit_neg: list[int], pinned: Path | None = None) -> dict[int, list]:
    """The English plan of epochs 4-5; with `pinned`, regenerated and compared to the pinned file (fails closed)."""
    rec = english_record(seed, fit_pos, fit_neg)
    if pinned is not None and N.read_json(pinned)["epochs"] != rec["epochs"]:
        raise SystemExit(f"English plan of seed {seed} differs from its pinned file {pinned}")
    out = {}
    for e in S.EPOCHS:
        if len(rec["epochs"][str(e)]) != SCH.STEPS_PER_EPOCH:
            raise SystemExit("English plan length changed")
        out[e] = rec["epochs"][str(e)]
    return out


# ---------------------------------------------------------------- contexts
def replay_paths(w_win: str = "w_windows", w_mask: str = "w_masks", i_win: str = "i_windows",
                 i_mask: str = "i_masks") -> dict:
    """nds_train path dict without any Lower Saxony key (the control never reads the FIT table)."""
    return {"w_table": J.W_FIT, "w_win": Path(w_win), "w_mask": Path(w_mask), "i_table": J.I_FIT,
            "i_win": Path(i_win), "i_mask": Path(i_mask)}


def control_context(paths: dict, sub: dict) -> dict:
    """C's Welsh + Irish rows (nds_train.region_context("C"): identical indices to C) with the extended subsets."""
    if any(k.startswith("ls_") for k in paths):
        raise SystemExit("control context refused: a Lower Saxony path was supplied")
    ctx = NT.region_context(S.NDS_ARM, paths)
    if ctx["rr"]["ls_pos"] or ctx["rr"]["ls_bg"]:
        raise SystemExit("control context holds Lower Saxony rows")
    ctx["sub"] = sub
    return ctx


def control_map(seed: int, epoch: int, ctx: dict) -> dict:
    return NT.schedule(S.NDS_ARM, seed, epoch, ctx["rr"], ctx["sub"])


def draw_digest(wmap: dict, rows: list[dict]) -> str:
    """nds_train's per-epoch draw digest form (step -> [uid, dy, dx, k]); equals C's TRAIN_RESULT plan_summary."""
    return N.digest({str(s): [rows[v[0]]["uid"], *v[1:]] for s, v in sorted(wmap.items())})


# ---------------------------------------------------------------- per-step tables (source, window, role, crop)
def control_steps(seed: int, epoch: int, real: list, wmap: dict, rows: list[dict], windows: list[dict]) -> list[dict]:
    kinds = NT.kinds_of(wmap, rows)
    out = []
    for s in range(SCH.STEPS_PER_EPOCH):
        if s in wmap:
            wi, dy, dx, k = wmap[s]
            r = rows[wi]
            role, uid, mask = r["role"], r["uid"], r["mask_path"]
        else:
            wi, dy, dx, k = real[s]
            w = windows[wi]
            role, uid, mask = ENGLISH_ROLE[w["role"]], f"england:{w['id']}", w["mask_path"]
        out.append({"t": SCH.t_index(epoch, s), "step": s, "kind": kinds[s], "role": role, "uid": uid,
                    "mask": mask, "dy": int(dy), "dx": int(dx), "k": int(k)})
    return out


def d1_steps(seed: int, epoch: int, ls: dict) -> list[dict]:
    wmap = MS.local_plan(seed, epoch, ls["pos"], ls["bg"])
    out = []
    for s in range(MS.STEPS_PER_EPOCH):
        wi, dy, dx, k = wmap[s]
        r = ls["rows"][wi]
        out.append({"t": SCH.t_index(epoch, s), "step": s, "kind": "lower_saxony", "role": r["role"], "uid": r["uid"],
                    "mask": r["mask_path"], "dy": int(dy), "dx": int(dx), "k": int(k)})
    return out


# ---------------------------------------------------------------- composition (plan level; masks optional)
def composition(steps: list[dict]) -> dict:
    """Coordinate-free aggregates of one run's 5,116 steps."""
    bad = []
    if sorted(r["t"] for r in steps) != list(range(S.N_UPDATES)):
        bad.append("update indices are not 0..5,115 once each")
    alt = [r["t"] for r in steps if r["role"] != ROLES[r["step"] % 2]]
    bad += [f"{len(alt)} steps break the positive/background alternation"] if alt else []
    off = [r["t"] for r in steps if not (-SCH.OFFSET_PX <= r["dy"] <= SCH.OFFSET_PX and
                                         -SCH.OFFSET_PX <= r["dx"] <= SCH.OFFSET_PX and 0 <= r["k"] < 8)]
    bad += [f"{len(off)} crops outside r2_train.draw"] if off else []
    by_role = Counter(r["role"] for r in steps)
    by_kind_role = Counter(f"{r['kind']}/{r['role']}" for r in steps)
    expo: dict = {}
    for kr in sorted(by_kind_role):
        c = Counter(r["uid"] for r in steps if f"{r['kind']}/{r['role']}" == kr)
        h = Counter(c.values())
        expo[kr] = {"draws": sum(c.values()), "distinct_windows": len(c), "max_draws_per_window": max(c.values()),
                    "histogram": {str(k): v for k, v in sorted(h.items())}}
    return {"updates": len(steps), "roles": dict(by_role), "kind_role_updates": dict(sorted(by_kind_role.items())),
            "exposure": expo, "status": "FAIL" if bad else "PASS", "violations": bad}


def mask_composition(steps: list[dict], root: Path = S.ROOT) -> dict:
    """Target content of each step's 512 px crop from its mask alone (no terrain: void policy V not applied, so
    NoData pixels count as their mask value). Per role: share of steps whose valid target has >= 1 positive pixel
    (F1 applies Dice only to those), mean positive share of valid pixels, all-IGNORE steps (skipped by the update)."""
    acc: dict = {}
    cache: dict = {}
    for r in steps:
        p = Path(r["mask"]) if Path(r["mask"]).is_absolute() else root / r["mask"]
        if p not in cache:
            if len(cache) > 4096:
                cache.clear()
            cache[p] = np.load(p)["label"]
        lab = rt.crop(cache[p], r["dy"], r["dx"])
        valid = lab != rt.IGNORE
        nv, npos = int(valid.sum()), int((lab == 1).sum())
        a = acc.setdefault(r["role"], {"steps": 0, "all_ignore": 0, "nonempty": 0, "pos_share_sum": 0.0})
        a["steps"] += 1
        if nv == 0:
            a["all_ignore"] += 1
            continue
        a["nonempty"] += npos > 0
        a["pos_share_sum"] += npos / nv
    out = {}
    for role, a in acc.items():
        n = a["steps"] - a["all_ignore"]
        out[role] = {"steps": a["steps"], "all_ignore_skipped": a["all_ignore"],
                     "share_with_positive_pixels": a["nonempty"] / n if n else None,
                     "mean_positive_pixel_share": a["pos_share_sum"] / n if n else None}
    return out


def smoke_windows(seed: int, n: int, ctx: dict) -> dict:
    """Welsh and Irish window ids of the regional steps among the first n updates of epoch 4 at the seed (the windows
    the GPU smoke stages with --only; the smoke then draws from those staged monuments, as nds_train.smoke)."""
    rows, out = ctx["rr"]["rows"], {"wales": set(), "ireland": set()}
    for s, v in control_map(seed, S.EPOCHS[0], ctx).items():
        if s < n:
            out[rows[v[0]]["region"]].add(rows[v[0]]["id"])
    return {k: sorted(v) for k, v in out.items()}


def ls_context(ls_table: Path, ls_mask: str = "ls_masks") -> dict:
    return MT.ls_context({"ls_table": ls_table, "ls_win": Path("ls_windows"), "ls_mask": Path(ls_mask)})
