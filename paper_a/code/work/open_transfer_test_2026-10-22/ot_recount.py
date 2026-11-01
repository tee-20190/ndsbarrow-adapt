"""OPEN-TRANSFER-1 independent recount (does not import scot_score): per arm and budget, matched records via
scipy maximum_bipartite_matching; compares with RESULT_SCORE.json point fields. Run after ot_score.py (WSL)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
PRIV = ROOT / "outputs/open_transfer_test_2026-10-22_PRIVATE"
REGIONS, ARMS, BUDGETS = ("GE", "LI", "NB"), ("D1", "CONT", "ADAF_barrow", "ADAF_AO"), (1.0, 2.0, 5.0, 10.0)


def matched(props, sites, n, reach=15.0):
    if n == 0 or not sites:
        return 0
    p = np.array([[o["easting"], o["northing"]] for o in props[:n]])
    s = np.array([[x["e"], x["n"]] for x in sites])
    pairs = cKDTree(p).query_ball_tree(cKDTree(s), reach)
    rows = [i for i, js in enumerate(pairs) for _ in js]
    cols = [j for js in pairs for j in js]
    if not rows:
        return 0
    g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(p), len(s)))
    return int((maximum_bipartite_matching(g, perm_type="column") >= 0).sum())


def main() -> None:
    res = json.loads((PRIV / "RESULT_SCORE.json").read_text(encoding="utf-8"))
    bad, rows = [], {}
    for a in ARMS:
        tot = {b: 0 for b in BUDGETS}
        for r in REGIONS:
            sites = json.loads((PRIV / f"regions/{r}/SITES.json").read_text(encoding="utf-8"))
            area = json.loads((PRIV / f"regions/{r}/PREP.json").read_text(encoding="utf-8"))["eval_area_km2"]
            props = json.loads((PRIV / f"infer/{r}/PROPOSALS_{a}_PRIVATE.json").read_text(encoding="utf-8"))["objects"]
            props = sorted(props, key=lambda o: (-o["score"], -o["mean_probability"], -o["area_m2"], o["id"]))
            for b in BUDGETS:
                m = matched(props, sites, min(int(round(b * area)), len(props)))
                tot[b] += m
                if m != res["arms"][r][a]["budgets"][str(b)]["matched"]:
                    bad.append((a, r, b, m, res["arms"][r][a]["budgets"][str(b)]["matched"]))
        rows[a] = tot
    out = {"recount_matched": rows, "mismatches": bad, "verdict": "AGREE" if not bad else "DISAGREE"}
    (PRIV / "RECOUNT.json").write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
