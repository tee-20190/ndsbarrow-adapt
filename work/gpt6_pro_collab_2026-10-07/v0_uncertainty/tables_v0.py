"""Print the RESULT.md tables from BOOT_V0.json and IGNORE_V0.json (percentage points)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B = ("0.5", "1.0", "2.0", "mean3")


def pp(x: float) -> str:
    return f"{100 * x:+.1f}"


def ci(c: list[float]) -> str:
    return f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"


def contrast_rows(d: dict, grid: str) -> None:
    print(f"\n### grid {grid}\n")
    print("| Region | Contrast | Decoder | Budget | Point | Geography | Training | Joint |")
    print("|---|---|---|---|---:|---|---|---|")
    for reg, rv in d["regions"].items():
        for dec, dv in rv["grids"][grid]["decoders"].items():
            for name, c in dv["contrasts"].items():
                for b in B:
                    print(f"| {reg} | {name} | {dec} | {b} | {pp(c['point'][b])} | {ci(c['ci95']['geography_only'][b])} "
                          f"| {ci(c['ci95']['training_only'][b])} | {ci(c['ci95']['joint'][b])} |")


def per_run(d: dict) -> None:
    print("\n### per-run point differences (2 km grid irrelevant: all-ones), mean3 and b=0.5\n")
    for reg, rv in d["regions"].items():
        for dec, dv in rv["grids"]["2km"]["decoders"].items():
            for name, c in dv["contrasts"].items():
                if "per_run" in c:
                    s = "; ".join(f"{k}: {pp(v['0.5'])} / {pp(v['mean3'])}" for k, v in c["per_run"].items())
                    print(f"- {reg} {name} {dec}: {s}")


def systems(d: dict) -> None:
    print("\n### system points and geography CIs (2 km), b=0.5 and mean3, percent\n")
    for reg, rv in d["regions"].items():
        for dec, dv in rv["grids"]["2km"]["decoders"].items():
            for s, p in dv["systems_point"].items():
                g = dv["systems_geo_ci95"][s]
                print(f"| {reg} | {dec} | {s} | {100*p['0.5']:.1f} {ci(g['0.5'])} | {100*p['mean3']:.1f} {ci(g['mean3'])} |"
                      f" {dv['underfilled_replicate_share'].get(s, '')}")


def oracle(d: dict) -> None:
    print("\n### R3-1 table\n")
    print("| Region | System | Decoder | Supply | C | b | K | M | R | c | M/100 slots | unused | unmatched | R_oracle |")
    print("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for reg, rv in d["regions"].items():
        for s, decs in rv["oracle_table"].items():
            for dec, row in decs.items():
                for b, x in row["budgets"].items():
                    print(f"| {reg} | {s} | {dec} | {row['supply']} | {row['C']} | {b} | {x['K']} | {x['M']} | "
                          f"{x['R']:.3f} | {x['c']:.3f} | {x['matches_per_100_slots']:.1f} | {x['unused_slots']} | "
                          f"{x['submitted_unmatched']} | {x['R_oracle']:.3f} |")


def main() -> None:
    d = json.loads((HERE / "BOOT_V0.json").read_text(encoding="utf-8"))
    what = sys.argv[1:] or ["contrasts", "perrun", "systems", "oracle"]
    if "contrasts" in what:
        for g in d["regions"]["oldenburg"]["grids"]:
            contrast_rows(d, g)
    if "perrun" in what:
        per_run(d)
    if "systems" in what:
        systems(d)
    if "oracle" in what:
        oracle(d)


if __name__ == "__main__":
    main()
