#!/bin/bash
# D1-COLLAPSE-1 remote job (Colab VM; started by remote_run.py through `colab exec`): a checked copy of ADAPT-MATCH-0's
# vm_job_match.sh (frozen 351b5961; only the D1 paths differ: one bundle, no English bundle). Reassembles and verifies each
# uploaded bundle (stream sha256 + MANIFEST.sha256), rebuilds the exact environment (COLAB-ASSESS-1 setup_env.sh,
# unchanged), runs the chain through HR1's frozen launcher step2/run_chain.sh inside storage_guard, then packages
# results and every final.pt for download in 48 MiB parts with ADAPT-NDS-1's package_results.sh (index with sha256).
# Settings: /content/cli/job.env (written by driver_d1.sh write_env; no secrets).
set -uo pipefail
unset MPLBACKEND PYTHONPATH PYTHONSTARTUP        # the Colab kernel exports an inline matplotlib backend
export NVIDIA_TF32_OVERRIDE=0                     # TF32 off (COLAB-ASSESS-1 rule)
export PYTHONDONTWRITEBYTECODE=1
J=/content/cli
DB=/content/DB
PARTS_OUT=/content/res_parts
set -a
# shellcheck source=/dev/null
source "$J/job.env"
set +a
: "${MODE:?}" "${BUNDLES:?}"
stamp() { echo "== $(date -u +%FT%TZ) $*"; }
stamp host
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
lscpu | grep -E '^Model name|^CPU\(s\)'
free -g | awk 'NR == 2 { print "ram_total_gb=" $2 }'
df -B1 /content | awk 'NR == 2 { printf "disk_start total_gb=%.1f free_gb=%.1f\n", $2 / 1e9, $4 / 1e9 }'
mkdir -p "$DB"
for B in $BUNDLES; do
  stamp "unpack $B"
  want_var="SHA_$B"
  want=${!want_var:?no expected sha256 for bundle $B}
  TAR_MODE=-xf
  if [[ ${BUNDLE_COMPRESSION:-none} == gzip ]]; then TAR_MODE=-xzf; fi
  python3 "$J/cat_parts.py" "$J/parts" "$B" "$J/$B.sha" | tar "$TAR_MODE" - -C "$DB" || { echo "UNPACK FAILED $B"; exit 2; }
  got=$(cut -d' ' -f1 "$J/$B.sha")
  [ "$got" = "$want" ] || { echo "TAR SHA FAILED $B: $got != $want"; exit 2; }
  (cd "$DB" && sha256sum -c --quiet MANIFEST.sha256) || { echo "MANIFEST CHECK FAILED $B"; exit 2; }
  mv "$DB/MANIFEST.sha256" "$DB/MANIFEST_$B.sha256"
  echo "$B ok $(cat "$J/$B.sha")"
done
ENVDIR=$DB/work/colab_assessment_2026-09-29/env
VM=$DB/outputs/d1_collapse_2026-10-06/vm
RESULTS=$VM/results
mkdir -p "$RESULTS" "$VM/state"
stamp env
bash "$ENVDIR/setup_env.sh" "$ENVDIR" "$RESULTS" || { echo "ENV FAILED"; exit 3; }
cd "$DB" || exit 2
D1=$DB/work/d1_collapse_2026-10-06
export PY=/content/venv/bin/python CHAIN_DIR=$D1/chain D1_DIR=$D1 NDS_DIR=$DB/work/adapt_nds_2026-10-05 VM RESULTS
export HR1_CHAIN=$DB/work/holdout_region_2026-10/chain HR3_CHAIN=$DB/work/hr3_ireland_2026-10-02/step2/chain
export FRESH_DIR=$DB/work/fresh_region_2026-10-03/step2
export CHAIN_MODE=$MODE CHAIN_FILE=$CHAIN_DIR/chain_d1_$MODE.sh OUT_DIR=$VM CHAIN_LOG=$RESULTS/chain.log
export FREEZE_TXT=$ENVDIR/freeze.txt
export LEDGER_CMD='echo "utc=$(date -u +%FT%TZ) uptime_s=$(cut -d. -f1 /proc/uptime) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"'
python3 - "$VM/state/BUDGET.json" "${WALL_CAP_S:?}" <<'PY'
import json, sys, time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"t0_epoch": time.time(), "cap_s": float(sys.argv[2])}))
PY
stamp "chain $MODE"
"$PY" "$DB/verify/v3/storage_guard.py" run --path /content --peak-gb 60 -- \
  bash "$DB/work/holdout_region_2026-10/step2/run_chain.sh" "$CHAIN_FILE"
rc=$?
stamp "chain exit $rc"
cp "$RESULTS/colab_freeze.txt" "$RESULTS/env_freeze.txt" 2>/dev/null || true
cp "$VM/state/"*.json* "$RESULTS/" 2>/dev/null || true
df -B1 /content | awk 'NR == 2 { printf "disk_chain_end total_gb=%.1f free_gb=%.1f\n", $2 / 1e9, $4 / 1e9 }'
# ---------------------------------------------------------------- package for download (partial results included)
bash "$DB/work/adapt_nds_2026-10-05/chain/package_results.sh" "$VM" "$PARTS_OUT" "${WEIGHTS_SINK:-keep}"
df -B1 /content | awk 'NR == 2 { printf "disk_end total_gb=%.1f free_gb=%.1f\n", $2 / 1e9, $4 / 1e9 }'
stamp "packaged $(wc -l < "$PARTS_OUT/INDEX.sha256") parts"
exit $rc
