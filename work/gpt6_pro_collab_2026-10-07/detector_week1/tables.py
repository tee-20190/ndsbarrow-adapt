"""Markdown tables from UNION_DECOMP_ALL.json (counts only)."""
import json
from pathlib import Path
d = json.loads((Path(__file__).parent / "UNION_DECOMP_ALL.json").read_text())
for name, r in d.items():
    n = r["eligible_parts"]
    print(f"\n### {name} (n = {n}; {r['by_class_n']}; background {r['background_km2']:.2f} km2; scored union {r['scored_km2_union']:.2f} km2)\n")
    print("| Model | HIT / RANK / NMS / MISSING at 0.5 | at 1 | at 2 | pre-NMS parts | post-NMS parts | neg/km2 | top-K all-charged hits 0.5/1/2 |")
    print("|---|---|---|---|---:|---:|---:|---|")
    for m, v in r["models"].items():
        k = v["kinds"]
        cell = lambda x: " / ".join(str(k[x][y]) for y in ("HIT", "RANK", "NMS", "MISSING"))
        t = " / ".join(str(v["topk_all_charged_parts"][x]) for x in ("0.5", "1.0", "2.0"))
        print(f"| {m} | {cell('0.5')} | {cell('1.0')} | {cell('2.0')} | {v['pre_nms_parts']} | {v['post_nms_parts']} | {v['neg_per_km2']:.2f} | {t} |")
    u, o = r["union"], r["oracle"]
    print(f"\nUnion ({', '.join(u['models'])}): pre-NMS any {u['pre_nms_any']} ({u['pre_nms_by_class']}), post-NMS any {u['post_nms_any']}, none {u['no_model_pre_nms']}; any-model HIT at 0.5/1/2: " + " / ".join(str(u['hit_any_at_d'][x]) for x in ('0.5','1.0','2.0')))
    for x in ("0.5", "1.0", "2.0"):
        q = o[x]
        print(f"  oracle d={x}: frozen-charge upper {q['frozen_charge_upper']}, greedy NMS-feasible {q['frozen_charge_greedy_nms_feasible']}; all-charged K={q['all_charged_K']} upper {q['all_charged_upper']}, greedy NMS-feasible {q['all_charged_greedy_nms_feasible']}, greedy no-NMS {q['all_charged_greedy_no_nms']}")
    for m, v in r["models"].items():
        if r['by_class_n'].get('enclosure'):
            print("  ", m, {c: {x: v['kinds_by_class'][c][x] for x in ('0.5',)} for c in v['kinds_by_class']})
