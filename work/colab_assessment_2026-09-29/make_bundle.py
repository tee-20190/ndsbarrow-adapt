"""Pack repo-relative files into a tar (zstd level 9 if DEST ends in .zst) with a sha256 manifest, for a Colab job directory.

  python work/colab_assessment_2026-09-29/make_bundle.py DEST.tar LISTFILE

LISTFILE holds one repo-relative path per line (# comments allowed). Refuses reserved-terrain and VALIDATION paths.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("outputs/validation_cont_2026-09-26/mosaics", "reserved_bertranges", "bertranges_evaluation_cores",
             "outputs/amazon_sensor_data_2026-09-29", "eba_split_assignment", "OPUS_CURRENT_WORK")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(dest: str, listfile: str) -> None:
    rels = [l.strip() for l in Path(listfile).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    bad = [r for r in rels if any(f in r for f in FORBIDDEN)]
    if bad:
        raise SystemExit(f"forbidden paths: {bad}")
    rows = []
    zst = dest.endswith(".zst")
    kw = {"level": 9} if zst else {}
    with tarfile.open(dest, "w:zst" if zst else "w", format=tarfile.PAX_FORMAT, **kw) as tar:
        for r in rels:
            p = ROOT / r
            rows.append({"path": r, "bytes": p.stat().st_size, "sha256": sha(p)})
            tar.add(p, arcname=r, recursive=False)
        text = "".join(f"{x['sha256']}  {x['path']}\n" for x in rows).encode()
        info = tarfile.TarInfo("MANIFEST.sha256")
        info.size = len(text)
        tar.addfile(info, io.BytesIO(text))
    summary = {"dest": dest, "files": len(rows), "bytes": sum(x["bytes"] for x in rows), "tar_sha256": sha(Path(dest))}
    Path(listfile).with_suffix(".manifest.json").write_text(json.dumps({"summary": summary, "files": rows}, indent=1),
                                                            encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(*sys.argv[1:])
