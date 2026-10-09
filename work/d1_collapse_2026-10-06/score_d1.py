"""D1-COLLAPSE-1 scoring, recipe selection and the frozen verdict (PACKET "Collapsed", "Acceptable performance loss",
"Sequence", "Exit rule"). Local only (labels never travel).

  python -B score_d1.py runs FREEZE TOP [TOP ...]            # per-run table of verified sessions -> stdout JSON
  python -B score_d1.py select FREEZE DEV1_TOP [DEV2_TOP]    # -> RECIPE_SELECTION.json (create-only)
  python -B score_d1.py verdict FREEZE BARROW_TOP AO_TOP     # -> RESULT.json (confirmation runs only)
  python -B score_d1.py old                                  # ADAPT-MATCH-0's six runs under the same definition

Primary = FRESH-1's barrow-group recall (mean over 0.5, 1, 2 background proposals per km^2) through ADAPT-MATCH-0's
frozen scoring path (score_match.dev_context / load_objects -> score_adapt.score_model). Collapsed iff (a) zero
Verden proposals, (b) primary < 0.5 x the arm's reference, or (c) the online detector ON at the end of training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1_common as D  # noqa: E402

N = D.N
import score_match as SM  # noqa: E402  ADAPT-MATCH-0 (frozen): dev_context, load_objects; SM.SA = score_adapt

SCHEMA = "d1_collapse_result_v1"


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact (Clopper-Pearson) one-sided 95 % upper bound on a binomial rate (k = 0: 1 - alpha ** (1 / n))."""
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(1 - alpha, k + 1, n - k))


def threshold(arm: str) -> float:
    return D.REFERENCE[arm] - D.NOISE


def collapse_reasons(arm: str, n_props: int, primary: float, det_on_end: bool) -> list[str]:
    out = []
    if n_props == 0:
        out.append("a_zero_proposals")
    if primary < D.collapse_threshold(arm):
        out.append("b_primary_below_half_reference")
    if det_on_end:
        out.append("c_detector_on_at_end")
    return out


def run_rows(f: dict, tops: list[Path]) -> dict:
    ctx, v, rows = SM.dev_context(f), [], {}
    for top in tops:
        sv = N.read_json(Path(top) / "SESSION_VERIFIED.json")
        if sv.get("status") != "PASS" or sv.get("freeze_sha256") != f["_sha256"]:
            raise SystemExit(f"{top}: SESSION_VERIFIED is not PASS under this freeze")
        runs = {r["name"]: r for r in D.session_runs(sv["index"])}
        for name, ckpt in sv["trained"].items():
            r = runs[name]
            row = {"session": sv["index"], "recipe": r["recipe"], "arm": r["arm"], "seed": r["seed"],
                   "checkpoint_sha256": ckpt, "detector": sv["detectors"][name]}
            tr = N.read_json(Path(top) / "v1/results/runs" / name / "TRAIN_RESULT.json")
            row["final_epoch"] = {k: tr["epochs"][-1].get(k) for k in
                                  ("mean_positive_loss", "mean_background_loss", "mean_positive_dice",
                                   "mean_background_sum_p")}
            if r["infer"]:
                key = f"dev/PROPOSALS_{name}_PRIVATE.json"
                objs = SM.load_objects(Path(top) / "v1/results" / key, name, sv["proposals"][key], ckpt, ctx, v)
                row["proposals"] = sum(len(x) for x in objs.values())
                row["primary"] = SM.SA.score_model(objs, ctx)["primary"]
                row["collapse_reasons"] = collapse_reasons(r["arm"], row["proposals"], row["primary"],
                                                           bool(row["detector"]["on_at_end"]))
                row["collapsed"] = bool(row["collapse_reasons"])
            rows[name] = row
    if v:
        raise SystemExit(f"scoring refused: {v[:5]}")
    return rows


def dev_rule(rows: dict, recipe: str) -> dict:
    xs = [r for r in rows.values() if r["recipe"] == recipe and r["arm"] == "BARROW_L" and r["seed"] in D.DEV_SEEDS]
    if sorted(r["seed"] for r in xs) != list(D.DEV_SEEDS):
        raise SystemExit(f"development runs of {recipe} incomplete")
    mean = sum(r["primary"] for r in xs) / len(xs)
    collapsed = sum(r["collapsed"] for r in xs)
    return {"recipe": recipe, "collapsed": collapsed, "mean_primary": mean, "threshold": threshold("BARROW_L"),
            "selected": collapsed == 0 and mean >= threshold("BARROW_L")}


def select(f: dict, tops: list[Path]) -> dict:
    if D.SELECTION.exists():
        raise SystemExit("RECIPE_SELECTION.json exists; selection is made once")
    rows = run_rows(f, tops)
    rules = [dev_rule(rows, "F1")]
    if not rules[0]["selected"] and any(r["recipe"] == "F2" for r in rows.values()):
        rules.append(dev_rule(rows, "F2"))
    chosen = next((r["recipe"] for r in rules if r["selected"]), None)
    nxt = "confirm" if chosen else ("dev_F2" if len(rules) == 1 else "stop_two_failures")
    rec = {"schema": "d1_recipe_selection_v1", "utc": N.utc(), "freeze_sha256": f["_sha256"], "rules": rules,
           "selected": chosen, "next": nxt, "development_runs": rows,
           "rule": "PACKET 'Sequence': selected iff 0/3 development BARROW_L runs collapsed and mean primary >= 0.519",
           "sessions": [N.rel(Path(t)) for t in tops]}
    N.create_only(D.SELECTION, rec)
    return rec


def verdict(f: dict, tops: list[Path]) -> dict:
    sel = N.read_json(D.SELECTION)
    rows = run_rows(f, tops)
    arms = {}
    for arm in D.ARMS:
        xs = [r for r in rows.values() if r["arm"] == arm and r["seed"] in D.CONFIRM_SEEDS and
              r["recipe"] == sel["selected"]]
        if sorted(r["seed"] for r in xs) != list(D.CONFIRM_SEEDS):
            raise SystemExit(f"confirmation runs of {arm} incomplete")
        k, mean = sum(r["collapsed"] for r in xs), sum(r["primary"] for r in xs) / len(xs)
        arms[arm] = {"collapsed": k, "n": len(xs), "collapse_rate_upper95": clopper_pearson_upper(k, len(xs)),
                     "mean_primary": mean, "threshold": threshold(arm), "reference": D.REFERENCE[arm],
                     "mean_ok": mean >= threshold(arm), "pass": k == 0 and mean >= threshold(arm),
                     "per_seed": {r["seed"]: r["primary"] for r in sorted(xs, key=lambda r: r["seed"])}}
    k_all = sum(a["collapsed"] for a in arms.values())
    rec = {"schema": SCHEMA, "utc": N.utc(), "freeze_sha256": f["_sha256"], "recipe": sel["selected"],
           "selection_sha256": N.sha256_file(D.SELECTION), "arms": arms,
           "pooled": {"collapsed": k_all, "n": 20, "collapse_rate_upper95": clopper_pearson_upper(k_all, 20)},
           "verdict": "PASS" if all(a["pass"] for a in arms.values()) else "FAIL", "runs": rows,
           "wording": "0/10 bounds the per-run collapse rate below the stated upper bound at 95 %; it does not show "
                      "collapse is eliminated", "attribution": D.ATTRIBUTION}
    N.write_json(D.RESULT, rec)
    return rec


def old_runs() -> dict:
    """ADAPT-MATCH-0's six runs under (a) and (b) (no curves, so (c) is shown from the final-epoch signature only)."""
    mf = SM.load_freeze(D.MATCH / "freeze/FREEZE.json")
    ctx, v, out = SM.dev_context(mf), [], {}
    for top in sorted((D.MC.CHAIN_RUNS).glob("run_s*/SESSION_VERIFIED.json")):
        sv = N.read_json(top)
        if sv.get("status") != "PASS":
            continue
        for name, ckpt in sv["trained"].items():
            key = f"dev/PROPOSALS_{name}_PRIVATE.json"
            objs = SM.load_objects(top.parent / "v1/results" / key, name, sv["proposals"][key], ckpt, ctx, v)
            tr = N.read_json(top.parent / "v1/results/runs" / name / "TRAIN_RESULT.json")
            arm = name.rsplit("_", 1)[0]
            p, n = SM.SA.score_model(objs, ctx)["primary"], sum(len(x) for x in objs.values())
            out[name] = {"proposals": n, "primary": p, "final_positive_loss": tr["epochs"][-1]["mean_positive_loss"],
                         "final_background_loss": tr["epochs"][-1]["mean_background_loss"],
                         "reasons_a_b": collapse_reasons(arm, n, p, False)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("runs", "select", "verdict", "old"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.cmd == "old":
        print(json.dumps(old_runs(), indent=1))
        return
    import sessions_d1 as S
    f = S.load_freeze(Path(a.args[0]))
    tops = [Path(t) for t in a.args[1:]]
    if a.cmd == "runs":
        print(json.dumps(run_rows(f, tops), indent=1))
    elif a.cmd == "select":
        print(json.dumps({k: v for k, v in select(f, tops).items() if k != "development_runs"}, indent=1))
    else:
        r = verdict(f, tops)
        print(json.dumps({k: r[k] for k in ("verdict", "recipe", "arms", "pooled")}, indent=1))


if __name__ == "__main__":
    main()
