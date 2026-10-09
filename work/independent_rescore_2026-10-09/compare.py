"""Comparison step (run only after RESCORE_RAW.json existed): my values vs the project's RESULT.json files.

Compares, per system and budget (RAW): K, matches, recall, supply, provenance sha256 and, where reported,
pool C / G_ranking. Writes COMPARE.json (aggregate only).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
B = ("0.5", "1.0", "2.0")


def load(p: str) -> dict:
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def find(o, key):
    """first dict value under `key` anywhere in o (depth-first)."""
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = find(v, key)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v, key)
            if r is not None:
                return r
    return None


def cells(mine: dict, theirs_raw: dict, prov: dict | None) -> list:
    out = []
    for b in B:
        t = theirs_raw.get(b) or theirs_raw.get(str(float(b)))
        m = mine["RAW"][b]
        out.append({"b": b, "K": (m["K"], t.get("K")), "matches": (m["matches"], t.get("matches")),
                    "recall": (m["recall"], t.get("recall")),
                    "equal": m["K"] == t.get("K") and m["matches"] == t.get("matches")
                    and abs(m["recall"] - t.get("recall")) < 1e-12})
    sup = theirs_raw.get("supply")
    out.append({"supply": (mine["supply_incore"], sup), "equal": sup is None or sup == mine["supply_incore"]})
    if prov:
        out.append({"sha256_equal": prov.get("sha256") == mine["sha256"]})
    return out


def main() -> None:
    r = json.loads((HERE / "RESCORE_RAW.json").read_text(encoding="utf-8"))
    rep = {}
    s1 = load("work/stage1_d1_v0_2026-10-07/RESULT.json")
    s2 = load("work/stage2_d1_controls_2026-10-07/RESULT.json")
    ol = load("work/oldenburg_transfer_2026-10-07/RESULT.json")
    systems_v = [r["verden"]["ADAF_barrow"]] + r["verden"]["D1"]
    rep["stage1"] = {s["model"] or "ADAF_barrow": cells(s, s1["systems"][s["model"] if s["model"] in s1["systems"] else "ADAF_barrow"]["RAW"],
                                                    s1["provenance"].get(s["model"]) or s1["provenance"]["ADAF_barrow"])
                     for s in systems_v}
    rep["stage2"] = {}
    for s in [r["verden"]["ADAF_barrow"]] + r["verden"]["D1"] + r["verden"]["CTL"]:
        name = s["model"] if s["model"] in s2["systems"] else "ADAF_barrow"
        rep["stage2"][name] = cells(s, s2["systems"][name]["RAW"], s2["provenance"].get(name))
    rep["stage2_gate"] = {k: s2["gate"].get(k) for k in ("per_seed_matches", "mean", "sum_matches", "positive", "verdict")}
    rep["stage2_three_way_RAW_0.5"] = s2["three_way"]["RAW_0.5"]
    rep["oldenburg"] = {}
    for s in [r["oldenburg"]["ADAF_barrow"]] + r["oldenburg"]["D1"]:
        name = s["model"] if s["model"] in ol["exact_scorer"] else "ADAF_barrow"
        t = ol["exact_scorer"][name]
        traw = t.get("RAW", t)
        rep["oldenburg"][name] = cells(s, traw, None)
        pt = ol["pool_terms"].get(name, {})
        rep["oldenburg"][name].append({"pool_terms_reported": pt, "my_C": s["C_pool"],
                                       "my_G_ranking": {b: s["RAW"][b]["G_ranking"] for b in B}})
    rep["oldenburg_gate"] = ol["gate"]
    rep["oldenburg_deployable"] = ol["deployable_checkpoint"]
    rep["oldenburg_parity"] = {k: ol["adaf_parity"][k] for k in ("status", "v0_match_counts_identical")}
    alleq = all(c["equal"] for sec in ("stage1", "stage2", "oldenburg") for v in rep[sec].values()
                for c in v if "equal" in c)
    shaeq = all(c["sha256_equal"] for sec in ("stage1", "stage2") for v in rep[sec].values()
                for c in v if "sha256_equal" in c)
    rep["all_cells_equal"] = alleq
    rep["all_provenance_sha256_equal"] = shaeq
    (HERE / "COMPARE.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8", newline="\n")
    print("all cells equal:", alleq, "| provenance sha equal:", shaeq)
    print("S2 gate:", rep["stage2_gate"])
    print("S2 three-way RAW 0.5:", json.dumps(rep["stage2_three_way_RAW_0.5"])[:600])
    print("OL gate:", json.dumps(rep["oldenburg_gate"])[:700])
    print("OL deployable:", json.dumps(rep["oldenburg_deployable"])[:300])
    print("OL pool ADAF:", rep["oldenburg"]["ADAF_barrow"][-1])
    print("OL parity:", rep["oldenburg_parity"])


if __name__ == "__main__":
    main()
