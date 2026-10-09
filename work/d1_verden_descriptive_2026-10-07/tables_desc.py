"""Print the RESULT.md tables from RESULT.json (Markdown, percentage points for differences).

  python -B work/d1_verden_descriptive_2026-10-07/tables_desc.py > work/d1_verden_descriptive_2026-10-07/TABLES.md
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "RESULT.json").read_text(encoding="utf-8"))
BS = ("0.5", "1.0", "2.0", "mean3")
LAB = {"0.5": "0.5", "1.0": "1", "2.0": "2", "mean3": "mean"}
ADAF = "ADAF_barrow"


def pp(x: float) -> str:
    return f"{100 * x:+.2f}"


def ci(v: list) -> str:
    return f"[{pp(v[0])}, {pp(v[1])}]"


def main_table() -> None:
    print("| Grid | Decoder | b | Point | Positive runs | Geography only | Training only | Joint |")
    print("|---|---|---|---:|---:|---|---|---|")
    for g in R["grids"]:
        for dec in ("RAW", "NMS60"):
            c = R["grids"][g]["decoders"][dec]["contrast"]
            for b in BS:
                pos = c["positive_runs"].get(b, "")
                print(f"| {g} | {dec} | {LAB[b]} | {pp(c['point'][b])} | {pos}{'/10' if pos != '' else ''} | "
                      f"{ci(c['ci95']['geography_only'][b])} | {ci(c['ci95']['training_only'][b])} | "
                      f"{ci(c['ci95']['joint'][b])} |")


def per_run() -> None:
    c = R["grids"]["2km"]["decoders"]
    print("| Run | RAW 0.5 | RAW 1 | RAW 2 | RAW 0.5 geography 95 % | NMS60 0.5 | NMS60 1 | NMS60 2 |")
    print("|---|---:|---:|---:|---|---:|---:|---:|")
    for n, v in c["RAW"]["contrast"]["per_run"].items():
        w = c["NMS60"]["contrast"]["per_run"][n]
        g = c["RAW"]["contrast"]["per_run_geography_ci95"][n]["0.5"]
        print(f"| seed {n[-2:]} | {pp(v['0.5'])} | {pp(v['1.0'])} | {pp(v['2.0'])} | {ci(g)} | {pp(w['0.5'])} | "
              f"{pp(w['1.0'])} | {pp(w['2.0'])} |")


def pool_terms() -> None:
    for dec in ("RAW", "NMS60"):
        print(f"\n**{dec}** (N = {R['N']}; K = 211 / 422 / 845)\n")
        print("| System | Supply | C_s | G_pool,s = N − C_s | M_s(b) 0.5 / 1 / 2 | min(K_b, C_s) 0.5 / 1 / 2 | "
              "G_ranking,s(b) 0.5 / 1 / 2 |")
        print("|---|---:|---:|---:|---|---|---|")
        for n, r in R["pool_terms"].items():
            x = r[dec]
            bb = [x["budgets"][b] for b in ("0.5", "1.0", "2.0")]
            name = "ADAF barrow" if n == ADAF else f"seed {n[-2:]}"
            print(f"| {name} | {x['supply']} | {x['C']} | {x['G_pool']} | {' / '.join(str(b['M']) for b in bb)} | "
                  f"{' / '.join(str(b['min_K_C']) for b in bb)} | {' / '.join(str(b['G_ranking']) for b in bb)} |")
        d1 = [r[dec] for n, r in R["pool_terms"].items() if n != ADAF]
        rng = lambda f: f"{min(f(x) for x in d1)}–{max(f(x) for x in d1)}"  # noqa: E731
        print(f"| **D1 range** | {rng(lambda x: x['supply'])} | {rng(lambda x: x['C'])} | {rng(lambda x: x['G_pool'])}"
              f" | | | {' / '.join(rng(lambda x, b=b: x['budgets'][b]['G_ranking']) for b in ('0.5', '1.0', '2.0'))} |")


def complexes() -> None:
    s = R["v0_1_sensitivity"]
    inv = s["inventory"]
    print(f"\nAvailable complexes: {inv['available_complexes']} (members min / median / max "
          f"{inv['members_per_available_complex']['min']} / {inv['members_per_available_complex']['median']} / "
          f"{inv['members_per_available_complex']['max']}); barrow endpoint identical to Stage 1: "
          f"{s['barrow_endpoint_identical_to_stage1']}\n")
    print("| Decoder | b | ADAF barrow | D1 mean | Mean diff | Min | Max | Positive |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for k, v in s["diff_vs_adaf"].items():
        dec, b = k.split("_")
        print(f"| {dec} | {LAB[b]} | {v['adaf']:.3f} | {v['mean_d1']:.3f} | {pp(v['mean_diff'])} | {pp(v['min'])} | "
              f"{pp(v['max'])} | {v['positive']}/10 |")


if __name__ == "__main__":
    print("## Main table\n")
    main_table()
    print("\n## Per run (2 km grid)\n")
    per_run()
    print("\n## Pool terms")
    pool_terms()
    print("\n## v0.1 complex endpoint (sensitivity, point values)")
    complexes()
