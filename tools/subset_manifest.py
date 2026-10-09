"""Write a subset copy of an inference manifest, for running infer_irl.py on a few tiles.

`infer_irl.py run` and `assemble` process every window of the manifest they are given and fail closed when a window is
missing, so a subset run needs a manifest that holds only the wanted tiles. This tool copies the input manifest and
keeps only the tiles whose tile id (for example VER-P-000) or window id (for example VER-P-000-w0) is listed; every
other key is unchanged. The output contains manifest geometry only (tile cores and window bounds), no proposals.

  python -B tools/subset_manifest.py INFER_MANIFEST OUT.json --ids VER-P-000-w0 VER-P-007-w0
  python -B tools/subset_manifest.py INFER_MANIFEST OUT.json --ids-file ids.txt      # one id per line

Stage the same windows with `lgln_reader.py stage ... --only <window ids>` (and `verify ... --only`), then run
`infer_irl.py run OUT.json WIN_DIR OUT_DIR --model ...` and `infer_irl.py assemble OUT.json OUT_DIR NAME DEST_DIR`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def read_ids(ids: list[str], ids_file: str | None) -> list[str]:
    out = list(ids or [])
    if ids_file:
        out += [ln.strip() for ln in Path(ids_file).read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
    if not out:
        raise SystemExit("no ids given (use --ids and/or --ids-file)")
    return out


def subset(manifest: dict, ids: list[str]) -> tuple[dict, list[str]]:
    want = set(ids)
    keep, found = [], set()
    for t in manifest["tiles"]:
        names = {t["id"]} | {w["id"] for w in t.get("windows", [])}
        hit = names & want
        if hit:
            keep.append(t)
            found |= hit
    missing = sorted(want - found)
    out = dict(manifest)
    out["tiles"] = keep
    return out, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", help="input inference manifest (for example INFER_MANIFEST_PRIVATE.json)")
    ap.add_argument("out", help="output path of the subset manifest (must differ from the input)")
    ap.add_argument("--ids", nargs="*", default=[], help="tile ids or window ids to keep")
    ap.add_argument("--ids-file", help="text file with one tile or window id per line")
    a = ap.parse_args()
    src, dst = Path(a.manifest), Path(a.out)
    if dst.resolve() == src.resolve():
        raise SystemExit("output must not overwrite the input manifest")
    ids = read_ids(a.ids, a.ids_file)
    m = json.loads(src.read_text(encoding="utf-8"))
    out, missing = subset(m, ids)
    if missing:
        raise SystemExit(f"{len(missing)} id(s) not in the manifest: {missing[:5]}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({"out": str(dst), "tiles": len(out["tiles"]),
                      "windows": sum(len(t.get("windows", [])) for t in out["tiles"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
