#!/bin/bash
# Build the local adaf-h2h-venv replica once per Colab VM: Python 3.12.14 + exact freeze (--no-deps; torch 2.14.0 cu130).
# Usage: bash setup_env.sh ENVDIR RESDIR   (idempotent; writes RESDIR/colab_freeze.txt)
set -uo pipefail
ENVDIR=$1
RES=$2
if [ -f /content/venv/.complete ] && cmp -s "$ENVDIR/freeze.txt" /content/venv/.freeze; then
  echo "env: reusing /content/venv"
else
  rm -rf /content/venv
  pip install -q uv
  uv venv -q /content/venv --python 3.12.14 || { echo "UV PYTHON FAILED"; exit 3; }
  DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
  [ "${DRIVER_CUDA:-0}" -ge 13 ] || { echo "DRIVER LACKS CUDA 13 ($DRIVER_CUDA)"; exit 5; }
  uv pip install -q --no-deps --python /content/venv/bin/python -r "$ENVDIR/freeze.txt" || { echo "PIP FAILED"; exit 4; }
  cp "$ENVDIR/freeze.txt" /content/venv/.freeze
  touch /content/venv/.complete
  echo "env: built exact local freeze (cu130)"
fi
uv pip freeze --python /content/venv/bin/python > "$RES/colab_freeze.txt" 2>/dev/null || true
/content/venv/bin/python -c "import torch,rasterio,timm,rvt;print('env check', torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.get_device_name(0), rasterio.__gdal_version__)"
