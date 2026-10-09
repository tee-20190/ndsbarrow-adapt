"""Make a runnable working copy of this release (decision D5).

The released code is byte-identical to the files that produced the paper's results, so it still contains the
hard-coded roots of the original machine and of the remote GPU VM. This script copies the verified release tree to
OUT_DIR and applies PATH_SHIM.patch there, replacing those roots with environment variables:

  NDSBARROW_ROOT     repository root (the OUT_DIR working copy, or wherever you keep inputs/outputs)
  NDSBARROW_PY       Python interpreter of the environment pinned in work/colab_assessment_2026-09-29/env/freeze.txt
  NDSBARROW_VM_BASE  base directory on a remote GPU VM (only for the remote training/inference drivers)

The files in this repository are never modified. Usage:

  python release_paths.py OUT_DIR            # verify ALLOWLIST.txt hashes, copy, patch
  python release_paths.py OUT_DIR --check    # dry run: verify hashes and that every hunk applies, write nothing

Most scorers run from the repository root with relative paths and need no patch; the shim covers the run-instruction
docstrings, the remote-VM drivers and a few module-level VM path constants, and makes the WIN_DIR argument of
infer_irl.py absolute (a relative WIN_DIR otherwise fails in tta_run.py's path.relative_to(ROOT)). Two withheld modules that released code
imports at module level are replaced in OUT_DIR by the fail-closed versions in release_shims/ (WITHHELD.md).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUNK = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def allowlist() -> dict[str, str]:
    rows = {}
    for line in (HERE / 'ALLOWLIST.txt').read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):
            continue
        digest, path = line.split('  ', 1)
        rows[path] = digest
    return rows


def parse_patch(text: str) -> dict[str, list[tuple[int, list[str]]]]:
    """{path: [(old_start, hunk_lines)]}; hunk lines keep their prefix (' ', '-', '+') and original line endings."""
    files: dict[str, list[tuple[int, list[str]]]] = {}
    lines = text.splitlines(keepends=True)
    i, cur = 0, None
    while i < len(lines):
        ln = lines[i]
        if ln.startswith('--- a/'):
            cur = lines[i + 1][len('+++ b/'):].rstrip('\r\n')
            files[cur] = []
            i += 2
            continue
        m = HUNK.match(ln)
        if m and cur is not None:
            old_start = int(m.group(1))
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith(('@@', '--- a/')):
                if lines[i].startswith('\\ No newline at end of file'):
                    body[-1] = body[-1][:-1]  # drop the newline the diff writer added after a final line
                else:
                    body.append(lines[i])
                i += 1
            files[cur].append((old_start, body))
            continue
        i += 1
    return files


def apply(original: str, hunks: list[tuple[int, list[str]]], name: str) -> str:
    src = original.splitlines(keepends=True)
    out: list[str] = []
    pos = 0
    for old_start, body in hunks:
        start = old_start - 1 if old_start > 0 else 0
        out.extend(src[pos:start])
        pos = start
        for b in body:
            tag, content = b[0], b[1:]
            if tag in (' ', '-'):
                if pos >= len(src) or src[pos] != content:
                    raise SystemExit(f'{name}: hunk at line {old_start} does not match the released file')
                if tag == ' ':
                    out.append(src[pos])
                pos += 1
            elif tag == '+':
                out.append(content)
    out.extend(src[pos:])
    return ''.join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out_dir')
    ap.add_argument('--check', action='store_true', help='verify and test-apply only; write nothing')
    a = ap.parse_args()
    rows = allowlist()
    bad = [p for p, d in rows.items() if not (HERE / p).is_file() or sha256(HERE / p) != d]
    if bad:
        raise SystemExit(f'{len(bad)} released file(s) differ from ALLOWLIST.txt, e.g. {bad[:3]}')
    patches = parse_patch((HERE / 'PATH_SHIM.patch').read_text(encoding='utf-8'))
    patched = {p: apply((HERE / p).read_bytes().decode('utf-8'), h, p) for p, h in patches.items()}
    print(f'verified {len(rows)} files against ALLOWLIST.txt; {len(patched)} files patch cleanly')
    if a.check:
        return
    out = Path(a.out_dir).resolve()
    if out == HERE or HERE in out.parents:
        raise SystemExit('OUT_DIR must be outside the release repository (the released files stay byte-identical)')
    for p in rows:
        dst = out / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / p, dst)
    for p, text in patched.items():
        (out / p).write_bytes(text.encode('utf-8'))
    shims = [p for p in rows if p.startswith('release_shims/')]
    for p in shims:  # fail-closed replacements for withheld modules that released code imports
        dst = out / p[len('release_shims/'):]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / p, dst)
    print(f'wrote {len(rows)} files to {out} ({len(patched)} patched, {len(shims)} withheld-module replacements); '
          f'set NDSBARROW_ROOT={out} and NDSBARROW_PY=<python> before running')


if __name__ == '__main__':
    sys.exit(main())
