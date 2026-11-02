#!/bin/bash
# Build the S_D + PROX decode environment (tools/clean_start/README.md, step 2): CPython 3.12.14 from uv, the exact pins
# of requirements-sd-prox.txt installed with --no-deps from PyPI, a freeze comparison, then check_env.py.
# The same recipe as work/colab_assessment_2026-09-29/env/setup_env.sh (the reference decodes' /content/venv), without
# Colab paths and without needing pip: uv >= 0.12 on PATH or in $UV (tested with 0.12.13).
# Usage, from the repository root: bash tools/clean_start/setup_env.sh ENV_DIR
# uv honours UV_CACHE_DIR and UV_PYTHON_INSTALL_DIR; keep the cache on ENV_DIR's volume so installs are hardlinks.
set -euo pipefail
ENV_DIR=${1:?usage: setup_env.sh ENV_DIR}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
UV=${UV:-uv}
command -v "$UV" >/dev/null || { echo "uv not found: install uv 0.12.13 (pip install uv==0.12.13, or the standalone installer)"; exit 2; }
echo "uv: $("$UV" --version)"
DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1 || true)
[ "${DRIVER_CUDA:-0}" -ge 13 ] || { echo "NVIDIA driver must report CUDA >= 13 (found: ${DRIVER_CUDA:-none})"; exit 5; }
[ -e "$ENV_DIR" ] && { echo "$ENV_DIR exists; use a new, empty path"; exit 6; }
"$UV" venv -q "$ENV_DIR" --python 3.12.14
PY=$ENV_DIR/bin/python
"$UV" pip install -q --no-deps --default-index https://pypi.org/simple --python "$PY" -r "$HERE/requirements-sd-prox.txt"
"$UV" pip freeze --python "$PY" > "$ENV_DIR/clean_start_freeze.txt"
diff <(grep -v '^#' "$HERE/requirements-sd-prox.txt" | tr 'A-Z_' 'a-z-' | sort) \
     <(tr 'A-Z_' 'a-z-' < "$ENV_DIR/clean_start_freeze.txt" | sort) \
  || { echo "installed packages differ from requirements-sd-prox.txt"; exit 4; }
"$PY" -I -B "$HERE/check_env.py" "$REPO" "$ENV_DIR/clean_start_env.json"
