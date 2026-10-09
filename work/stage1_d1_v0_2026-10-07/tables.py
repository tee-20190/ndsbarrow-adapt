"""Print RESULT.md tables from RESULT.json (STAGE1-D1-V0). python -B tables.py"""
from __future__ import annotations

import json
from pathlib import Path

R = json.loads((Path(__file__).resolve().parent / "RESULT.json").read_text(encoding="utf-8"))
B = ("0.5", "1.0", "2.0")


def table(dec: str) -> None:
    ref = R["systems"]["ADAF_barrow"][dec]
    print(f"\n{dec}: recall (matches) / diff vs ADAF barrow / matches per 100 slots; supply = in-core proposals")
    print("| System | Supply | b=0.5 | b=1 | b=2 | Δ0.5 | Δ1 | Δ2 | /100 slots 0.5 / 1 / 2 |")
    print("|---|---:|---|---|---|---:|---:|---:|---|")
    for name, sysr in R["systems"].items():
        r = sysr[dec]
        cells = [f"{r[b]['recall']:.3f} ({r[b]['matches']})" for b in B]
        dif = [f"{r[b]['recall'] - ref[b]['recall']:+.3f}" if name != "ADAF_barrow" else "–" for b in B]
        p100 = " / ".join(f"{r[b]['per_100_slots']:.1f}" for b in B)
        label = name.replace("D1F1_BARROW_L_20260928", "BARROW_L seed ")
        print(f"| {label} | {r['supply']} | {' | '.join(cells)} | {' | '.join(dif)} | {p100} |")
    s = R["summaries"]
    print("| **10-run mean** | | " + " | ".join(f"{s[f'{dec}_{float(b)}']['mean_run']:.3f}" for b in B) + " | "
          + " | ".join(f"{s[f'{dec}_{float(b)}']['mean_diff']:+.3f}" for b in B) + " | |")
    print("| runs with Δ > 0 | | | | | " + " | ".join(f"{s[f'{dec}_{float(b)}']['positive']}/10" for b in B) + " | |")


for d in ("RAW", "NMS60"):
    table(d)
print("\nK / ceiling:", {b: (R["K"][b], round(min(1, R["K"][b] / R["entities_N"]), 3)) for b in B})
