"""Frozen, CPU-only ADAF_MEAN/E0099 point-list comparison.

This module deliberately has no dependency on the post-lock integration scorer.
``prepare`` and ``smoke`` never score the real fused stream. ``run`` requires an
externally pinned manifest and writes into a fresh, exclusive result directory.
Builder ``verify`` is a reproducibility check, not the independent Opus audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


ARMS = ("ADAF_MEAN", "E0099")
COHORTS = {
    "pilot_excluded_25": (("0.5", 311), ("1.0", 622), ("headline", 1229)),
    "full_26": (("0.5", 323), ("1.0", 647), ("headline", 1294)),
}
PILOT_BLOCK = "SCREEN_113_53"
INPUT_KEYS = ("full_ranked", "pilot_ranked", "catalogue_manifest", "panel_join", "observation_join", "summary")
FROZEN_HASHES = {
    "full_ranked": "aff0b69c069c4d9b5b8385bf992419b91fa7f8beeb0e32162c8f6fdee1d91250",
    "pilot_ranked": "25355df9a7ce33e32422293c3b127d8ea4a4c02902472d794be17dd28424bc7e",
    "catalogue_manifest": "174571e8c4825b1cbf0cfd95ef60e35537f94820ddf2ba12da04d3e24b882378",
    "panel_join": "fcb70dabc76bd83e15591e7fb5113fdb430df32253a5763c4a61f7e54a4d8f6e",
    "observation_join": "728ff6ba0d6dcbd9d5fe52c6deb8f4b8374dc1e6ffbbb8909fddfb291ad35f09",
    "summary": "56f54b37c61bb14d3f6eff29edd84c7c7698950a5738a7ed5fd0a17df6c65660",
}
MAX_WALL = 900
MAX_RSS = 2 * 1024**3
MAX_OUTPUT = 64 * 1024**2


class InvalidPacket(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidPacket(f"non-finite or nonnumeric {name}")
    return float(value)


def _inside_half_open(easting: float, northing: float, bounds: list[float]) -> bool:
    x0, y0, x1, y1 = bounds
    return x0 <= easting < x1 and y0 <= northing < y1


def _rss() -> int:
    # ru_maxrss is KiB in WSL/Linux. psutil supplies current RSS on Windows.
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class Meter:
    def __init__(self, limits: dict[str, int], prior_wall: float = 0.0, prior_output: int = 0):
        self.start = time.monotonic()
        self.limits = limits
        self.peak_rss = _rss()
        self.prior_wall = prior_wall
        self.output_bytes = prior_output

    def check(self) -> None:
        self.peak_rss = max(self.peak_rss, _rss())
        if self.prior_wall + time.monotonic() - self.start > self.limits["wall_seconds"]:
            raise InvalidPacket("wall ceiling exceeded")
        if self.peak_rss > self.limits["rss_bytes"]:
            raise InvalidPacket("RSS ceiling exceeded")
        if self.output_bytes > self.limits["output_bytes"]:
            raise InvalidPacket("output ceiling exceeded")

    def stats(self) -> dict[str, Any]:
        self.check()
        return {"wall_seconds": time.monotonic() - self.start,
                "cumulative_smoke_plus_run_wall_seconds": self.prior_wall + time.monotonic() - self.start,
                "peak_rss_bytes": self.peak_rss,
                "output_bytes_before_current_receipt_including_prior_smoke": self.output_bytes}


def _write_new(path: Path, object_: Any, meter: Meter) -> str:
    data = (json.dumps(object_, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if meter.output_bytes + len(data) > meter.limits["output_bytes"] - 4096:
        raise InvalidPacket("output ceiling would leave no STOPPED receipt reserve")
    meter.check()
    with path.open("xb") as stream:
        stream.write(data)
    meter.output_bytes += len(data)
    return hashlib.sha256(data).hexdigest()


def _write_stopped(path: Path, cause: str, meter: Meter) -> None:
    data = (json.dumps({"status": "STOPPED", "cause": cause,
                        "elapsed_wall_seconds": time.monotonic() - meter.start,
                        "peak_rss_bytes": max(meter.peak_rss, _rss())}, sort_keys=True) + "\n").encode()
    if meter.output_bytes + len(data) <= meter.limits["output_bytes"]:
        with path.open("xb") as stream:
            stream.write(data)


def _reserve(path: Path) -> None:
    # No preexisting result directory, including an empty one, can be reused.
    if path.exists():
        raise InvalidPacket(f"occupied output path: {path}")
    path.mkdir(parents=True, exist_ok=False)


def _manifest(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    m = _read(path)
    if m.get("schema") != 1 or set(m.get("inputs", {})) != set(INPUT_KEYS):
        raise InvalidPacket("manifest schema/input keys mismatch")
    root = Path(m["workspace_root"]).resolve(strict=True)
    paths: dict[str, Path] = {}
    for key in INPUT_KEYS:
        pin = m["inputs"][key]
        if pin["sha256"] != FROZEN_HASHES[key]:
            raise InvalidPacket(f"manifest differs from frozen source identity: {key}")
        p = (root / pin["path"]).resolve(strict=True)
        if not p.is_relative_to(root) or _sha(p) != pin["sha256"]:
            raise InvalidPacket(f"source pin mismatch: {key}")
        paths[key] = p
    pin = m["spec"]
    spec = (root / pin["path"]).resolve(strict=True)
    if not spec.is_relative_to(root) or _sha(spec) != pin["sha256"]:
        raise InvalidPacket("spec pin mismatch")
    paths["spec"] = spec
    limits = m.get("limits", {})
    if any(limits.get(k) != v for k, v in (("wall_seconds", MAX_WALL),
                                           ("rss_bytes", MAX_RSS),
                                           ("output_bytes", MAX_OUTPUT))):
        raise InvalidPacket("resource ceilings must equal frozen contract")
    if not m.get("owner") or not isinstance(m.get("owner"), str):
        raise InvalidPacket("missing writer identity")
    return m, paths


def _prior_smoke(manifest: dict[str, Any]) -> tuple[float, int]:
    pin = manifest.get("prior_smoke")
    if not isinstance(pin, dict):
        raise InvalidPacket("run requires externally pinned prior_smoke receipt")
    root = Path(manifest["workspace_root"]).resolve(strict=True)
    path = (root / pin["path"]).resolve(strict=True)
    if not path.is_relative_to(root) or _sha(path) != pin["sha256"]:
        raise InvalidPacket("prior smoke pin mismatch")
    receipt = _read(path)
    if (receipt.get("status") != "COMPLETION" or receipt.get("command") != "smoke" or
        receipt.get("synthetic_only") is not True or
        receipt.get("real_fused_scoring_performed") is not False):
        raise InvalidPacket("prior smoke is not a valid synthetic completion")
    measured = receipt["resource_measurement"]
    wall = _finite(measured["wall_seconds"], "smoke wall")
    output = path.stat().st_size
    if wall < 0 or wall >= MAX_WALL or output >= MAX_OUTPUT:
        raise InvalidPacket("prior smoke exceeded or exhausted budget")
    return wall, output


def _run_output_matches(manifest: dict[str, Any], output: Path) -> None:
    root = Path(manifest["workspace_root"]).resolve(strict=True)
    pinned = (root / manifest["output_dir"]).resolve()
    if not pinned.is_relative_to(root) or output.resolve() != pinned:
        raise InvalidPacket("run output differs from pinned output_dir")


def _labels(raw: dict[str, Any], summary: dict[str, Any], panel: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    blocks = raw["blocks"]
    full = summary["cohorts"]["full_26"]["baseline"]["blocks"]
    pilot = summary["cohorts"]["pilot_excluded_25"]["baseline"]["blocks"]
    if len(full) != 26 or len(set(full)) != 26 or set(full) != {b["id"] for b in blocks}:
        raise InvalidPacket("full block roster mismatch")
    if len(pilot) != 25 or set(pilot) != set(full) - {PILOT_BLOCK}:
        raise InvalidPacket("pilot block roster mismatch")
    catalogue: list[dict[str, Any]] = []
    for block in blocks:
        for entry in block["labels"]:
            if not (entry["inside_frozen_core"] and set(entry["classes"]) & {"barrow", "enclosure"}):
                continue
            e, n = _finite(entry["easting"], "label easting"), _finite(entry["northing"], "label northing")
            if not _inside_half_open(e, n, block["interior"]):
                raise InvalidPacket(f"eligible label outside half-open interior: {entry['id']}")
            catalogue.append({"id": entry["id"], "easting": e, "northing": n,
                              "owner_block": block["id"]})
    if len(catalogue) != 101 or len({x["id"] for x in catalogue}) != 101:
        raise InvalidPacket("eligible catalogue size/IDs mismatch")
    mapped = {row["mapping"]["part_id"]: row["mapping"] for row in panel["rows"]
              if row["mapping"]["kind"] == "catalogue"}
    if set(mapped) != {x["id"] for x in catalogue}:
        raise InvalidPacket("audited panel/catalogue identity mismatch")
    for label in catalogue:
        entry = mapped[label["id"]]
        if (entry["owner_block"] != label["owner_block"] or
            entry["easting"] != label["easting"] or entry["northing"] != label["northing"]):
            raise InvalidPacket(f"audited label geometry mismatch: {label['id']}")
    out = {"full_26": catalogue, "pilot_excluded_25": [x for x in catalogue if x["owner_block"] != PILOT_BLOCK]}
    if len(out["pilot_excluded_25"]) != 94:
        raise InvalidPacket("pilot eligible catalogue size mismatch")
    return out


def _pools(raw: dict[str, Any], blocks: list[str]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for arm in ARMS:
        seen = set()
        result = []
        for i, entry in enumerate(raw[arm]):
            ident = entry["id"]
            if not isinstance(ident, str) or not ident or ident in seen:
                raise InvalidPacket(f"duplicate/invalid {arm} ID")
            seen.add(ident)
            block = entry["source_block"]
            if block not in blocks:
                raise InvalidPacket(f"candidate source block not in cohort: {arm}:{ident}")
            result.append({"arm": arm, "original_id": ident,
                           "easting": _finite(entry["easting"], "candidate easting"),
                           "northing": _finite(entry["northing"], "candidate northing"),
                           "source_block": block, "original_rank": i + 1})
        out[arm] = result
    return out


def interleave(pools: dict[str, list[dict[str, Any]]], budget: int) -> list[dict[str, Any]]:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise InvalidPacket("invalid budget")
    need_a, need_e = (budget + 1) // 2, budget // 2
    if len(pools[ARMS[0]]) < need_a or len(pools[ARMS[1]]) < need_e:
        raise InvalidPacket(f"insufficient arm supply at B={budget}: need {need_a}/{need_e}")
    rows = []
    for i in range(budget):
        arm = ARMS[i % 2]
        row = dict(pools[arm][i // 2])
        row["fused_rank"] = i + 1
        rows.append(row)
    return rows


def match_greedy(rows: list[dict[str, Any]], labels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unused = {x["id"]: x for x in labels}
    matches = []
    for row in rows:
        best = None
        for label in unused.values():
            d = math.hypot(row["easting"] - label["easting"], row["northing"] - label["northing"])
            if d <= 60.0 and (best is None or (d, label["id"]) < (best[0], best[1]["id"])):
                best = (d, label)
        if best is not None:
            distance, label = best
            matches.append({"label_id": label["id"], "label_owner_block": label["owner_block"],
                            "distance_m": distance, "proposal_arm": row["arm"],
                            "proposal_id": row["original_id"], "proposal_rank": row.get("fused_rank", row["original_rank"]),
                            "original_rank": row["original_rank"], "proposal_source_block": row["source_block"]})
            del unused[label["id"]]
    return matches


def _marks(raw: dict[str, Any], cohort: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"catalogue": [], "random_control": []}
    for entry in raw["rows"]:
        state = entry["cohorts"][cohort]["state"]
        if state != "supported_in_view":
            continue
        if not entry["recorded_in_view_proximity_eligible"] or entry["world_estimate_E_N"] is None:
            raise InvalidPacket("supported mark lacks exact eligible centre")
        kind = entry["panel_kind"]
        e, n = entry["world_estimate_E_N"]
        out[kind].append({"observation_id": entry["observation_id"],
                          "panel_owner_block": entry["panel_owner_block"],
                          "catalogue_part_id": entry.get("catalogue_part_id"),
                          "confidence": entry["instance"]["visibility"],
                          "form": entry["instance"]["form"],
                          "easting": _finite(e, "mark easting"),
                          "northing": _finite(n, "mark northing")})
    expected = (118, 5) if cohort == "pilot_excluded_25" else (126, 5)
    if (len(out["catalogue"]), len(out["random_control"])) != expected:
        raise InvalidPacket("supported mark denominator mismatch")
    return out


def proximity(rows: list[dict[str, Any]], marks: dict[str, list[dict[str, Any]]],
              shift: tuple[float, float] = (0.0, 0.0)) -> dict[str, Any]:
    report = {}
    for kind, values in marks.items():
        detail = []
        for mark in values:
            e, n = mark["easting"] + shift[0], mark["northing"] + shift[1]
            nearest = min(((math.hypot(e - r["easting"], n - r["northing"]),
                            r["fused_rank"], r) for r in rows), key=lambda x: (x[0], x[1])) if rows else None
            dist = nearest[0] if nearest else None
            detail.append({"observation_id": mark["observation_id"],
                           "panel_owner_block": mark["panel_owner_block"],
                           "catalogue_part_id": mark["catalogue_part_id"],
                           "confidence": mark["confidence"], "form": mark["form"],
                           "distance_m": dist, "nearest_arm": nearest[2]["arm"] if nearest else None,
                           "nearest_id": nearest[2]["original_id"] if nearest else None,
                           "nearest_fused_rank": nearest[1] if nearest else None,
                           "within_15m": dist is not None and dist <= 15,
                           "within_30m": dist is not None and dist <= 30,
                           "within_60m": dist is not None and dist <= 60})
        report[kind] = {"denominator": len(detail), "within_15m": sum(x["within_15m"] for x in detail),
                        "within_30m": sum(x["within_30m"] for x in detail),
                        "within_60m": sum(x["within_60m"] for x in detail), "rows": detail}
    return report


def _score(pools: dict[str, list[dict[str, Any]]], labels: list[dict[str, Any]],
           marks: dict[str, list[dict[str, Any]]], cohort: str,
           budget_id: str, budget: int, meter: Meter) -> dict[str, Any]:
    fused = interleave(pools, budget)
    selected = {"INTERLEAVE_50_50": fused}
    for arm in ARMS:
        if len(pools[arm]) < budget:
            raise InvalidPacket(f"control supply short: {cohort} {arm} B={budget}")
        selected[arm] = [dict(row, fused_rank=i + 1) for i, row in enumerate(pools[arm][:budget])]
    arm_reports = {}
    for arm, rows in selected.items():
        meter.check()
        assignments = match_greedy(rows, labels)
        p = proximity(rows, marks)
        arm_reports[arm] = {"requested_B": budget, "actual_slots": len(rows),
                            "available_pool_sizes": {a: len(pools[a]) for a in ARMS},
                            "cohort": cohort, "density_label": budget_id, "budget_id": budget_id,
                            "rows": rows, "assignments": assignments, "H": len(assignments),
                            "proximity": p}
        if arm == "INTERLEAVE_50_50":
            arm_reports[arm]["alternate_halfpixel_proximity"] = proximity(rows, marks, (0.5, -0.5))
    return {"cohort": cohort, "budget_id": budget_id, "requested_B": budget,
            "arms": arm_reports}


def decision(reports: dict[str, Any], blocks: list[str]) -> dict[str, Any]:
    primary = reports["pilot_excluded_25"]["0.5"]["arms"]
    full = reports["full_26"]["0.5"]["arms"]
    by_arm = {a: {m["label_id"]: m for m in primary[a]["assignments"]} for a in primary}
    if (len(by_arm["ADAF_MEAN"]), len(by_arm["E0099"])) != (30, 31):
        raise InvalidPacket("primary control count mismatch")
    if (full["ADAF_MEAN"]["H"], full["E0099"]["H"]) != (32, 34):
        raise InvalidPacket("full control count mismatch")
    fused = by_arm["INTERLEAVE_50_50"]
    e = by_arm["E0099"]
    delta = len(fused) - 31
    contributions = {b: sum(m["label_owner_block"] == b for m in fused.values()) -
                     sum(m["label_owner_block"] == b for m in e.values()) for b in blocks}
    deletion = {b: delta - d for b, d in contributions.items()}
    guard = primary["INTERLEAVE_50_50"]["proximity"]["catalogue"]["within_30m"]
    alternate = primary["INTERLEAVE_50_50"]["alternate_halfpixel_proximity"]["catalogue"]["within_30m"]
    if primary["ADAF_MEAN"]["proximity"]["catalogue"]["within_30m"] != 47:
        raise InvalidPacket("ADAF mark guard control mismatch")
    reversal = (guard >= 47) != (alternate >= 47)
    if delta <= 0 or (guard < 47 and alternate < 47):
        status = "FAILURE"
    elif delta >= 4 and all(v > 0 for v in deletion.values()) and full["INTERLEAVE_50_50"]["H"] >= 34 and guard >= 47 and not reversal:
        status = "SUCCESS"
    else:
        status = "INCONCLUSIVE"
    return {"status": status, "pilot_delta_H_vs_stronger_control": delta,
            "pilot_delta_H_vs_ADAF_MEAN": len(fused) - 30,
            "pilot_delta_H_vs_E0099": delta,
            "pilot_H": len(fused), "full_H": full["INTERLEAVE_50_50"]["H"],
            "pilot_catalogue_mark_30m": guard, "pilot_catalogue_mark_denominator": 118,
            "alternate_halfpixel_30m": alternate, "halfpixel_guard_reversal": reversal,
            "block_contributions_vs_E0099": contributions,
            "fixed_assignment_block_deletion_deltas": deletion,
            "gained_vs_E0099": [fused[k] for k in sorted(set(fused) - set(e))],
            "lost_vs_E0099": [e[k] for k in sorted(set(e) - set(fused))],
            "gained_vs_ADAF_MEAN": [fused[k] for k in sorted(set(fused) - set(by_arm["ADAF_MEAN"]))],
            "lost_vs_ADAF_MEAN": [by_arm["ADAF_MEAN"][k] for k in sorted(set(by_arm["ADAF_MEAN"]) - set(fused))]}


def _load_checked(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary = _read(paths["summary"])
    catalogue_manifest = _read(paths["catalogue_manifest"])
    labels = _labels(catalogue_manifest, summary, _read(paths["panel_join"]))
    interiors = {block["id"]: block["interior"] for block in catalogue_manifest["blocks"]}
    raw_marks = _read(paths["observation_join"])
    marks = {c: _marks(raw_marks, c) for c in COHORTS}
    support_exclusions = {c: [{"observation_id": row["observation_id"],
                               "panel_kind": row["panel_kind"],
                               "centre_state": row["instance"]["centre_state"],
                               "cohort_state": row["cohorts"][c]["state"]}
                              for row in raw_marks["rows"]
                              if row["cohorts"][c]["state"] != "supported_in_view"] for c in COHORTS}
    if sum(x["cohort_state"] == "non_exact_centre" for x in support_exclusions["full_26"]) != 7:
        raise InvalidPacket("non-exact mark exclusion mismatch")
    pools = {}
    for cohort, filename in (("full_26", "full_ranked"), ("pilot_excluded_25", "pilot_ranked")):
        pools[cohort] = _pools(_read(paths[filename]), summary["cohorts"][cohort]["baseline"]["blocks"])
        for arm in ARMS:
            for candidate in pools[cohort][arm]:
                if not _inside_half_open(candidate["easting"], candidate["northing"],
                                         interiors[candidate["source_block"]]):
                    raise InvalidPacket(f"candidate violates half-open source interior: {arm}:{candidate['original_id']}")
        expected = (1294, 15401) if cohort == "full_26" else (1229, 14823)
        if tuple(len(pools[cohort][a]) for a in ARMS) != expected:
            raise InvalidPacket(f"ranked supply mismatch: {cohort}")
    review = summary["review"]
    if (review["reviewer_kind"], review["reviewer_role"], review["human_confirmed"]) != ("model", "primary", False):
        raise InvalidPacket("review provenance mismatch")
    return {"labels": labels, "marks": marks, "pools": pools,
            "support_exclusions": support_exclusions}, summary, review


def _receipt(status: str, command: str, manifest: Path, meter: Meter,
             extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"schema": 1, "status": status, "command": command,
            "manifest_sha256": _sha(manifest), "limits": meter.limits,
            "resource_measurement": meter.stats(), **(extra or {})}


def prepare(manifest: Path, output: Path) -> None:
    m, paths = _manifest(manifest)
    meter = Meter(m["limits"])
    data, summary, review = _load_checked(paths)
    _reserve(output)
    try:
        receipt = _receipt("COMPLETION", "prepare", manifest, meter,
                           {"writer": m["owner"], "input_hashes": {k: m["inputs"][k]["sha256"] for k in INPUT_KEYS},
                            "spec_sha256": m["spec"]["sha256"],
                            "cohorts": {c: {"blocks": summary["cohorts"][c]["baseline"]["blocks"],
                                             "labels": len(data["labels"][c]),
                                             "supported_catalogue_marks": len(data["marks"][c]["catalogue"]),
                                             "supported_control_marks": len(data["marks"][c]["random_control"]),
                                             "pool_sizes": {a: len(data["pools"][c][a]) for a in ARMS}} for c in COHORTS},
                            "review": review, "real_fused_scoring_performed": False})
        _write_new(output / "PREPARE.json", receipt, meter)
    except Exception as exc:
        _write_stopped(output / "STOPPED.json", str(exc), meter)
        raise


def smoke(manifest: Path, output: Path) -> None:
    m, paths = _manifest(manifest)
    meter = Meter(m["limits"])
    _load_checked(paths)  # raw-record schema/load smoke only
    _reserve(output)
    try:
        a = [{"arm": "ADAF_MEAN", "original_id": f"same-{i}", "easting": float(i * 100),
              "northing": 0.0, "source_block": "B", "original_rank": i + 1} for i in range(3)]
        e = [{"arm": "E0099", "original_id": f"same-{i}", "easting": float(i * 100),
              "northing": 0.0, "source_block": "B", "original_rank": i + 1} for i in range(3)]
        rows = interleave({"ADAF_MEAN": a, "E0099": e}, 5)
        if [(r["arm"], r["original_rank"], r["fused_rank"]) for r in rows] != [
            ("ADAF_MEAN", 1, 1), ("E0099", 1, 2), ("ADAF_MEAN", 2, 3),
            ("E0099", 2, 4), ("ADAF_MEAN", 3, 5)]:
            raise InvalidPacket("synthetic allocator smoke failed")
        labels = [{"id": "a", "easting": 0.0, "northing": 60.0, "owner_block": "B"},
                  {"id": "b", "easting": 0.0, "northing": -60.0, "owner_block": "B"}]
        matches = match_greedy(rows[:2], labels)
        if [x["label_id"] for x in matches] != ["a", "b"]:
            raise InvalidPacket("synthetic greedy smoke failed")
        receipt = _receipt("COMPLETION", "smoke", manifest, meter,
                           {"synthetic_only": True, "real_schema_load_only": True,
                            "real_fused_scoring_performed": False,
                            "synthetic_interleave": [(r["arm"], r["original_rank"], r["fused_rank"]) for r in rows],
                            "synthetic_match_ids": [x["label_id"] for x in matches]})
        _write_new(output / "SMOKE.json", receipt, meter)
    except Exception as exc:
        _write_stopped(output / "STOPPED.json", str(exc), meter)
        raise


def _calculate(data: dict[str, Any], summary: dict[str, Any], meter: Meter) -> tuple[dict[str, Any], dict[str, Any]]:
    reports = {}
    for cohort, budgets in COHORTS.items():
        reports[cohort] = {}
        for budget_id, budget in budgets:
            reports[cohort][budget_id] = _score(data["pools"][cohort], data["labels"][cohort],
                                                  data["marks"][cohort], cohort, budget_id, budget, meter)
            reports[cohort][budget_id]["support_exclusions"] = data["support_exclusions"][cohort]
    frozen_controls = {
        "pilot_excluded_25": {"0.5": (30, 31), "1.0": (39, 36), "headline": (44, 41)},
        "full_26": {"0.5": (32, 34), "1.0": (40, 40), "headline": (46, 46)},
    }
    for cohort, values in frozen_controls.items():
        for budget_id, expected in values.items():
            actual = tuple(reports[cohort][budget_id]["arms"][a]["H"] for a in ARMS)
            if actual != expected:
                raise InvalidPacket(f"frozen controls mismatch: {cohort}/{budget_id} {actual} != {expected}")
    return reports, decision(reports, summary["cohorts"]["pilot_excluded_25"]["baseline"]["blocks"])


def run(manifest: Path, output: Path) -> None:
    m, paths = _manifest(manifest)
    _run_output_matches(m, output)
    prior_wall, prior_output = _prior_smoke(m)
    meter = Meter(m["limits"], prior_wall, prior_output)
    _reserve(output)
    try:
        _write_new(output / "STARTED.json", _receipt("STARTED", "run", manifest, meter,
                  {"writer": m["owner"], "real_fused_scoring_performed": False}), meter)
        data, summary, review = _load_checked(paths)
        reports, verdict = _calculate(data, summary, meter)
        payload = {"schema": 1, "manifest_sha256": _sha(manifest), "reports": reports,
                   "decision": verdict, "review_provenance": review,
                   "interpretation_limit": "Repeated-use SCREEN development diagnostics; proximity is not instance correctness or false-alarm rate."}
        result_hash = _write_new(output / "RESULT.json", payload, meter)
        _write_new(output / "COMPLETION.json", _receipt("COMPLETION", "run", manifest, meter,
                  {"result_sha256": result_hash, "decision_status": verdict["status"],
                   "independent_check_required": True}), meter)
    except Exception as exc:
        _write_stopped(output / "STOPPED.json", str(exc), meter)
        raise


def verify(manifest: Path, output: Path) -> None:
    m, paths = _manifest(manifest)
    _run_output_matches(m, output)
    prior_wall, prior_output = _prior_smoke(m)
    completion = _read(output / "COMPLETION.json")
    if completion.get("status") != "COMPLETION" or completion.get("manifest_sha256") != _sha(manifest):
        raise InvalidPacket("completion/manifest mismatch")
    if (output / "STOPPED.json").exists() or _sha(output / "RESULT.json") != completion["result_sha256"]:
        raise InvalidPacket("stopped or result hash mismatch")
    meter = Meter(m["limits"], prior_wall, prior_output)
    data, summary, review = _load_checked(paths)
    reports, verdict = _calculate(data, summary, meter)
    recorded = _read(output / "RESULT.json")
    if recorded != {"schema": 1, "manifest_sha256": _sha(manifest), "reports": reports,
                    "decision": verdict, "review_provenance": review,
                    "interpretation_limit": "Repeated-use SCREEN development diagnostics; proximity is not instance correctness or false-alarm rate."}:
        raise InvalidPacket("result differs from pinned-source recomputation")
    print(json.dumps({"status": "BUILDER_VERIFIED", "decision_status": verdict["status"],
                      "independent_check_required": True, "resource_measurement": meter.stats()}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "smoke", "run", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        globals()[args.command](args.manifest, args.output)
    except (InvalidPacket, KeyError, ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"INVALID / STOPPED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
