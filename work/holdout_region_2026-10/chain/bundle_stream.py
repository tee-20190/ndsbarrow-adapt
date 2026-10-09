"""Stream a make_bundle.py-format tar to stdout without writing a local tar file (C: keeps <= 1 GB).

Format as work/colab_assessment_2026-09-29/make_bundle.py (imported, unchanged: FORBIDDEN, sha): PAX, arcname = the
repo-relative path, recursive=False, a MANIFEST.sha256 member last. Refuses make_bundle.FORBIDDEN paths, the Welsh
TEST role's staged terrain and anything outside the repo. Writes LIST.manifest.json (per-file sha256) and SHA_OUT
("tar_sha256 bytes") after the last byte, so the VM can verify the reassembled stream.

  python -B bundle_stream.py LISTFILE MANIFEST_OUT SHA_OUT | split -b 48M -d -a 4 --filter=... - PREFIX_part_
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "work/colab_assessment_2026-09-29"))
import make_bundle as mb  # noqa: E402  frozen-in-practice bundler: FORBIDDEN list and sha

EXTRA_FORBIDDEN = ("holdout_region_2026-10/test", "/TEST_", "step2b/")


class HashingWriter(io.RawIOBase):
    def __init__(self, raw) -> None:
        self.raw, self.h, self.n = raw, hashlib.sha256(), 0

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:
        self.h.update(b)
        self.n += len(b)
        self.raw.write(b)
        return len(b)


def read_list(listfile: Path) -> list[str]:
    return [ln.strip() for ln in listfile.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def check_list(rels: list[str], allow: tuple[str, ...] = ()) -> None:
    bad = [r for r in rels if any(f in r for f in mb.FORBIDDEN + EXTRA_FORBIDDEN) and r not in allow]
    outside = [r for r in rels if not (C.ROOT / r).resolve().is_relative_to(C.ROOT) or not (C.ROOT / r).is_file()]
    if bad or outside or len(set(rels)) != len(rels):
        raise SystemExit(f"bundle list refused: forbidden {bad[:3]}, missing/outside {outside[:3]}, "
                         f"duplicates {len(rels) - len(set(rels))}")


def stream(listfile: Path, manifest_out: Path, sha_out: Path, allow: tuple[str, ...] = ()) -> dict:
    rels = read_list(listfile)
    check_list(rels, allow)
    w = HashingWriter(sys.stdout.buffer)
    rows = []
    compression = os.environ.get("BUNDLE_COMPRESSION", "none")
    if compression not in ("none", "gzip"):
        raise SystemExit("BUNDLE_COMPRESSION must be none or gzip")
    opts = {"compresslevel": 1} if compression == "gzip" else {}
    with tarfile.open(fileobj=w, mode="w|gz" if compression == "gzip" else "w|",
                      format=tarfile.PAX_FORMAT, **opts) as tar:
        for r in rels:
            p = C.ROOT / r
            rows.append({"path": r, "bytes": p.stat().st_size, "sha256": mb.sha(p)})
            tar.add(p, arcname=r, recursive=False)
        text = "".join(f"{x['sha256']}  {x['path']}\n" for x in rows).encode()
        info = tarfile.TarInfo("MANIFEST.sha256")
        info.size = len(text)
        tar.addfile(info, io.BytesIO(text))
    sys.stdout.buffer.flush()
    summary = {"files": len(rows), "bytes": sum(x["bytes"] for x in rows), "tar_sha256": w.h.hexdigest(),
               "tar_bytes": w.n, "compression": compression}
    C.write_json(manifest_out, {"summary": summary, "files": rows})
    sha_out.write_bytes(f"{w.h.hexdigest()} {w.n}\n".encode())
    return summary


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    stream(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), tuple(sys.argv[4:]))
