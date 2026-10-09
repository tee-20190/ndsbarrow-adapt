"""PACKET.md decision rule: pooled SCREEN-26 + stage-2-73 common-class recovery, NEG minus control, with a paired
block bootstrap, and the rule's noise calibration on recorded arm pairs (the SCREEN_RULE_NOISE.json method, extended
to the pooled endpoint and to the interval rule).

Rule (fixed in Phase A, before any NEG output exists):
  D(d) = pooled common-class hits, NEG - control, at d proposals per km2: SCREEN K = 323 / 647 / 1,294 plus stage-2
  K = 909 / 1,818 / 3,635 for d = 0.5 / 1 / 2. Paired block bootstrap: the 26 SCREEN and 73 stage-2 blocks are
  resampled with replacement within their landscape (BOOT_N draws, seed BOOT_SEED); the statistic is the pooled hit
  difference (the global top-K selection of each landscape is held fixed, as in every earlier block bootstrap here).
  GAIN       lower bound of the two-sided LEVEL interval of D(0.5) > 0, and D(2) >= -2
  TRADE_OFF  lower bound of D(0.5) > 0, and D(2) <= -3
  HARM       otherwise, if the upper bound of D(0.5) < 0 or the upper bound of D(2) < 0
  NULL       otherwise
Calibration, acceptance fixed before computing: per-block sign-flip null over the discordant parts of the recorded
arm pairs scored on both landscapes (CONT-R2, AIMX-CONT, AIMX-R2). LEVEL = 0.90 (the briefed 90 % interval) is kept if
the null P(GAIN) is <= 0.10 for every pair; otherwise 0.95 and then 0.99 are tried in that order. The per-part null
and the power (the better arm wins each discordant part with probability p) are reported as in SCREEN_RULE_NOISE.json.

  python -B work/bgscale_2026-09-26/bg_rule.py calibrate   # CPU, recorded hit lists only -> RULE_NOISE_POOLED.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STAGES = ROOT / "work/adaf_followup_2026-09-19/english/STAGES_26_PLUS_73.json"
STAGES_SHA256 = "a0447a2d3815309d30f6a88da8da61908512cfc7455075c639225285f1661804"
SCREEN_RESULT = ROOT / "outputs/aim_train_audit_2026-09-24/SCREEN_RESULT.json"
STAGE2_RESULT = ROOT / "outputs/stage2_2026-09-24/STAGE2_RESULT.json"
CALIBRATION = HERE / "RULE_NOISE_POOLED.json"

DENSITIES = ("d0.5", "d1.0", "d2.0")
SCREEN_K = {"d0.5": 323, "d1.0": 647, "d2.0": 1294}
STAGE2_K = {"d0.5": 909, "d1.0": 1818, "d2.0": 3635}
N_SCREEN, N_STAGE2 = 26, 73
BOOT_N, BOOT_SEED = 10000, 20260926
LEVELS = (0.90, 0.95, 0.99)
NULL_GAIN_MAX = 0.10
MAX_D2_DROP = 2
CAL_PAIRS = (("CONT", "R2"), ("AIMX", "CONT"), ("AIMX", "R2"))
NULL_DRAWS, NULL_SEED = 2000, 20260927
POWER_P = (0.6, 0.7, 0.8)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- bootstrap and rule
def boot_counts(n_blocks: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, n_blocks) multiplicities of each block in n resamples of n_blocks blocks with replacement."""
    idx = rng.integers(0, n_blocks, size=(n, n_blocks))
    counts = np.zeros((n, n_blocks), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(n), n_blocks), idx.ravel()), 1.0)
    return counts


def boot_matrices(n: int = BOOT_N, seed: int = BOOT_SEED) -> np.ndarray:
    """(n, 26 + 73) block multiplicities: SCREEN and stage-2 blocks resampled within their own landscape."""
    rng = np.random.default_rng(seed)
    return np.hstack([boot_counts(N_SCREEN, n, rng), boot_counts(N_STAGE2, n, rng)])


def interval(boot: np.ndarray, level: float, axis: int = -1) -> tuple:
    a = (1 - level) / 2
    return np.quantile(boot, a, axis=axis), np.quantile(boot, 1 - a, axis=axis)


def outcome(lo05: float, hi05: float, d2: float, hi2: float) -> str:
    if lo05 > 0:
        return "GAIN" if d2 >= -MAX_D2_DROP else "TRADE_OFF"
    if hi05 < 0 or hi2 < 0:
        return "HARM"
    return "NULL"


def decide(delta: dict, level: float, mats: np.ndarray | None = None) -> dict:
    """delta[d] = per-block hit differences (26 SCREEN blocks, then 73 stage-2 blocks) for d in DENSITIES."""
    mats = boot_matrices() if mats is None else mats
    res: dict = {"level": level, "boot_n": int(mats.shape[0]), "boot_seed": BOOT_SEED}
    for d in DENSITIES:
        v = np.asarray(delta[d], dtype=np.float64)
        if v.shape != (N_SCREEN + N_STAGE2,):
            raise ValueError(f"{d}: expected {N_SCREEN + N_STAGE2} block differences, got {v.shape}")
        boot = mats @ v
        lo, hi = interval(boot, level)
        res[d] = {"D": int(round(v.sum())), "D_screen": int(round(v[:N_SCREEN].sum())),
                  "D_stage2": int(round(v[N_SCREEN:].sum())), "interval": [float(lo), float(hi)],
                  "share_gt_0": float((boot > 0).mean()), "share_lt_0": float((boot < 0).mean())}
    res["outcome"] = outcome(res["d0.5"]["interval"][0], res["d0.5"]["interval"][1], res["d2.0"]["D"],
                             res["d2.0"]["interval"][1])
    return res


def rule_text(level: float) -> str:
    return (f"GAIN: lower bound of the two-sided {level:.0%} paired block-bootstrap interval of pooled D(0.5) > 0 and "
            f"D(2) >= -{MAX_D2_DROP}; TRADE_OFF: lower bound of D(0.5) > 0 and D(2) <= -{MAX_D2_DROP + 1}; HARM: "
            f"upper bound of D(0.5) < 0 or upper bound of D(2) < 0; else NULL")


# ---------------------------------------------------------------- calibration on recorded pairs
def recorded_hits() -> tuple[dict, dict]:
    """hits[arm][d] = set of (landscape, part id); block[(landscape, part id)] = pooled block index 0..98."""
    if sha(STAGES) != STAGES_SHA256:
        raise RuntimeError("stage list changed")
    st = json.loads(STAGES.read_text(encoding="utf-8"))
    block = {}
    for i, b in enumerate(st["stage1"]):
        for lab in b["labels"]:
            block[("S", lab["id"])] = i
    for i, b in enumerate(st["stage2_conditional"]):
        for lab in b["labels"]:
            block[("T", lab["id"])] = N_SCREEN + i
    scr = json.loads(SCREEN_RESULT.read_text(encoding="utf-8"))["cohorts"]["full_26"]
    s2 = json.loads(STAGE2_RESULT.read_text(encoding="utf-8"))["budgets"]
    hits: dict = {}
    for arm in ("CONT", "R2", "AIMX"):
        hits[arm] = {d: {("S", i) for i in scr[f"K{SCREEN_K[d]}"][arm]["ids"]} |
                     {("T", i) for i in s2[d][arm]["ids"]} for d in ("d0.5", "d2.0")}
    return hits, block


def discordance(a: dict, b: dict, block: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per discordant part: (d0.5, d2.0) difference vector and its pooled block index."""
    parts = sorted(set().union(*a.values(), *b.values()))
    vec = [(int(p in a["d0.5"]) - int(p in b["d0.5"]), int(p in a["d2.0"]) - int(p in b["d2.0"])) for p in parts]
    keep = [i for i, v in enumerate(vec) if v != (0, 0)]
    return np.array([vec[i] for i in keep], dtype=np.float64), np.array([block[parts[i]] for i in keep])


def rates(d05: np.ndarray, d2: np.ndarray, mats: np.ndarray, chunk: int = 250) -> dict:
    """Rule outcomes for many simulated per-block difference rows (draws x 99) at every LEVEL."""
    res: dict[float, list[str]] = {level: [] for level in LEVELS}
    for s in range(0, len(d05), chunk):
        x05, x2 = d05[s:s + chunk], d2[s:s + chunk]
        b05, b2 = x05 @ mats.T, x2 @ mats.T
        for level in LEVELS:
            lo05, hi05 = interval(b05, level, axis=1)
            _, hi2 = interval(b2, level, axis=1)
            res[level] += [outcome(lo05[i], hi05[i], x2[i].sum(), hi2[i]) for i in range(len(x05))]
    return {f"{level:.2f}": {k: round(r.count(k) / len(r), 4) for k in ("GAIN", "TRADE_OFF", "HARM", "NULL")}
            for level, r in res.items()}


def calibrate() -> dict:
    hits, block = recorded_hits()
    mats = boot_matrices()
    rng = np.random.default_rng(NULL_SEED)
    nb = N_SCREEN + N_STAGE2
    out = {"method": "per-block (and per-part) sign-flip null over discordant parts of recorded pairs; power: the "
                     "better arm wins each discordant part with probability p (SCREEN_RULE_NOISE.json method)",
           "acceptance": f"keep the first LEVEL in {LEVELS} whose per-block null P(GAIN) <= {NULL_GAIN_MAX} for every "
                         f"pair in {[f'{a}-{b}' for a, b in CAL_PAIRS]}",
           "boot_n": BOOT_N, "boot_seed": BOOT_SEED, "null_draws": NULL_DRAWS, "null_seed": NULL_SEED,
           "inputs": {p.relative_to(ROOT).as_posix(): sha(p) for p in (STAGES, SCREEN_RESULT, STAGE2_RESULT)},
           "pairs": {}}
    for a, b in CAL_PAIRS:
        vec, blk = discordance(hits[a], hits[b], block)
        onehot = np.zeros((len(vec), nb))
        onehot[np.arange(len(vec)), blk] = 1.0
        per_block = vec.T @ onehot  # (2, 99)
        obs = decide({"d0.5": per_block[0], "d1.0": per_block[0], "d2.0": per_block[1]}, LEVELS[0], mats)
        used = np.flatnonzero(np.abs(per_block).sum(0) > 0)
        s_blk = rng.choice([-1.0, 1.0], size=(NULL_DRAWS, nb))
        s_part = rng.choice([-1.0, 1.0], size=(NULL_DRAWS, len(vec)))
        null_block = rates(s_blk * per_block[0], s_blk * per_block[1], mats)
        null_part = rates((s_part * vec[:, 0]) @ onehot, (s_part * vec[:, 1]) @ onehot, mats)
        mag = np.abs(vec)
        power = {}
        for p in POWER_P:
            s = np.where(rng.random((NULL_DRAWS, len(vec))) < p, 1.0, -1.0)
            power[str(p)] = rates((s * mag[:, 0]) @ onehot, (s * mag[:, 1]) @ onehot, mats)
        out["pairs"][f"{a}-{b}"] = {
            "observed_D05_D2": [obs["d0.5"]["D"], obs["d2.0"]["D"]],
            "observed_interval_90_D05": obs["d0.5"]["interval"], "observed_outcome_at_0.90": obs["outcome"],
            "discordant_parts_d0.5": int((vec[:, 0] != 0).sum()), "discordant_parts_d2.0": int((vec[:, 1] != 0).sum()),
            "discordant_blocks": int(len(used)),
            "null_sd_D05_D2_per_block": [round(float(np.sqrt((per_block[0] ** 2).sum())), 2),
                                         round(float(np.sqrt((per_block[1] ** 2).sum())), 2)],
            "null_per_block": null_block, "null_per_part": null_part, "power_per_part": power}
    level = next((lv for lv in LEVELS if all(v["null_per_block"][f"{lv:.2f}"]["GAIN"] <= NULL_GAIN_MAX
                                             for v in out["pairs"].values())), None)
    out["selected_level"] = level
    out["status"] = "PASS" if level is not None else "FAIL_NO_LEVEL"
    out["rule"] = rule_text(level) if level is not None else None
    CALIBRATION.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def selected_level() -> float:
    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if cal.get("status") != "PASS":
        raise RuntimeError("rule calibration did not pass; no decision level")
    return float(cal["selected_level"])


if __name__ == "__main__":
    if sys.argv[1:] == ["calibrate"]:
        r = calibrate()
        print(json.dumps({"status": r["status"], "selected_level": r["selected_level"],
                          **{k: {"obs": v["observed_D05_D2"], "discordant_blocks": v["discordant_blocks"],
                                 "null_block": v["null_per_block"], "power": v["power_per_part"]}
                             for k, v in r["pairs"].items()}}, indent=1))
    else:
        raise SystemExit(__doc__)
