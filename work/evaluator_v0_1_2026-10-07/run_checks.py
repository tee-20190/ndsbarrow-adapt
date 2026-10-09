"""Evaluator v0.1 pre-freeze checks on real data (SPEC.md "Checks before the freeze"). CPU, read-only. Aggregates only.

1. Reproduction: v0.1 with no voids equals the exact Stage 1 scorer (work/stage1_d1_v0_2026-10-07/RESULT.json) for
   ADAF barrow and the ten D1 runs on Verden, every decoder, budget, match count, K and supply.
2. Readings: v0.1 on Verden ADAF barrow equals IGNORE reading (i) (historical) and differs from reading (iii) by the
   residual non-container IGNORE removal (v0_uncertainty/IGNORE_V0.json).
3. Oldenburg ADAF barrow (existing frozen proposals): v0.1 equals BENCH_V0 v0 (detector_week1/BENCH_V0.json).
4. Register role table and complex inventory (register + manifest only; no detector output) for Verden and Oldenburg.

  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
      /home/tee-/adaf-h2h-venv/bin/python -B work/evaluator_v0_1_2026-10-07/run_checks.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import eval_v01 as V

HERE = Path(__file__).resolve().parent
ROOT = V.ROOT
STAGE1 = ROOT / "work/stage1_d1_v0_2026-10-07/RESULT.json"
IGNORE = ROOT / "work/gpt6_pro_collab_2026-10-07/v0_uncertainty/IGNORE_V0.json"
WEEK1 = ROOT / "work/gpt6_pro_collab_2026-10-07/detector_week1/BENCH_V0.json"
REGIONS = {
    "verden": {"manifest": "outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json",
               "register": "outputs/fresh_region_2026-10-03/labels/denkmale_Verden_20261003T084547Z.geojson",
               "supplement": "outputs/fresh_region_2026-10-03/step0/EDGE_SUPPLEMENT_PRIVATE.json"},
    "oldenburg": {"manifest": "outputs/adapt_nds_2026-10-05/confirm/run/INDIVIDUAL_MANIFEST_PRIVATE.json",
                  "register": "outputs/adapt_nds_2026-10-05/register/denkmale_Oldenburg_20261005T010102Z.geojson",
                  "supplement": "outputs/adapt_nds_2026-10-05/register/supplement_Oldenburg_20261005T010102Z.json"},
}
OLD_ADAF = "outputs/adapt_nds_2026-10-05/score/confirm/individual/PROPOSALS_ADAF_barrow_PRIVATE.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def records(region: str) -> list[dict]:
    reg = V.load_json(ROOT / REGIONS[region]["register"])["features"]
    sup = V.load_json(ROOT / REGIONS[region]["supplement"])["records"]
    return [{"id": f["id"], "geometry": f.get("geometry"), "properties": f.get("properties") or {}} for f in reg + sup]


def stage1_reproduction(st: V.Setup) -> dict:
    res = V.load_json(STAGE1)
    checked, bad = 0, []
    for name, prov in res["provenance"].items():
        got = V.score_barrow(V.load_json(ROOT / prov["path"])["objects"], st)
        for dec in V.DECODERS:
            want = res["systems"][name][dec]
            pairs = [("supply", got[dec]["supply"], want["supply"])]
            pairs += [(f"{b}.{k}", got[dec][str(b)][k], want[str(b)][k])
                      for b in V.B for k in ("K", "matches", "recall", "ceiling_c", "per_100_slots", "slots_filled")]
            pairs.append(("primary", got[dec]["primary"], want["primary"]))
            for key, g, w in pairs:
                checked += 1
                if g != w:
                    bad.append([name, dec, key, g, w])
    return {"values_checked": checked, "mismatches": len(bad), "identical": not bad, "first_mismatches": bad[:5]}


def readings(st: V.Setup) -> dict:
    res = V.load_json(STAGE1)
    got = V.score_barrow(V.load_json(ROOT / res["provenance"]["ADAF_barrow"]["path"])["objects"], st)
    var = V.load_json(IGNORE)["regions"]["verden"]["variants"]
    out = {}
    for dec in V.DECODERS:
        v01 = {str(b): got[dec][str(b)]["recall"] for b in V.B}
        r1 = {str(b): var["historical"]["decoders"][dec]["systems_point"]["ADAF_barrow"][str(b)] for b in V.B}
        r3 = {str(b): var["precedence"]["decoders"][dec]["systems_point"]["ADAF_barrow"][str(b)] for b in V.B}
        out[dec] = {"v01": v01, "reading_i": r1, "reading_iii": r3,
                    "v01_equals_reading_i": all(v01[k] == r1[k] for k in v01),
                    "v01_minus_reading_iii_points": {k: 100 * (v01[k] - r3[k]) for k in v01}}
    out["reading_iii_K"] = var["precedence"]["K"]
    out["v01_K"] = {str(b): V.budget(b, st.area_km2) for b in V.B}
    return out


def oldenburg_adaf(st: V.Setup) -> dict:
    got = V.score_barrow(V.load_json(ROOT / OLD_ADAF)["objects"], st)
    want = V.load_json(WEEK1)["oldenburg"]["systems"]["ADAF_barrow"]["v0_individual"]
    rows = {dec: {str(b): [got[dec][str(b)]["recall"], want[dec][str(b)]] for b in V.B} for dec in V.DECODERS}
    same = all(g == w for d in rows.values() for g, w in d.values()) and all(
        got[d]["supply"] == want[d]["supply"] for d in V.DECODERS)
    return {"identical_to_bench_v0": same, "values": rows, "proposals_sha256": sha(ROOT / OLD_ADAF)}


def inventory(region: str, st: V.Setup, manifest: dict) -> dict:
    recs = records(region)
    targets = {int(p["record_id"]) for p in manifest["parts"]}
    roles = Counter(V.record_role(r["properties"], int(r["id"]), targets) for r in recs)
    cx = V.build_complexes(recs, manifest, st)
    return {"records": len(recs), "roles": dict(sorted(roles.items())), "targets_in_register": len(
        targets & {int(r["id"]) for r in recs}), "complexes": cx.diag}


def main() -> None:
    out = {"schema": "evaluator_v0_1_checks_v1", "evaluator": V.VERSION, "eval_v01_sha256": sha(HERE / "eval_v01.py"),
           "inputs_sha256": {}, "regions": {}}
    for region, spec in REGIONS.items():
        mp = ROOT / spec["manifest"]
        manifest = V.load_json(mp)
        st = V.setup(manifest, [])
        out["inputs_sha256"] |= {spec[k]: sha(ROOT / spec[k]) for k in ("manifest", "register", "supplement")}
        r = {"N": st.n, "scored_km2": st.area_km2, "K": {str(b): V.budget(b, st.area_km2) for b in V.B},
             "voids": st.diag, "inventory": inventory(region, st, manifest)}
        if region == "verden":
            r["stage1_reproduction"] = stage1_reproduction(st)
            r["readings_adaf_barrow"] = readings(st)
        else:
            r["adaf_barrow_bench_v0_reproduction"] = oldenburg_adaf(st)
        out["regions"][region] = r
        print(region, json.dumps({k: v for k, v in r.items() if k != "inventory"}, indent=1)[:3000], flush=True)
        print(json.dumps(r["inventory"], indent=1), flush=True)
    out["inputs_sha256"] |= {str(p.relative_to(ROOT)): sha(p) for p in (STAGE1, IGNORE, WEEK1)}
    (HERE / "CHECKS.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
