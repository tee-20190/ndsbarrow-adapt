#!/bin/bash
# COLAB-CLI-1 helpers, sourced by the stage drivers (WSL). Caller sets RUN_DIR (local output dir) and SESSION.
COLAB_BIN=${COLAB_BIN:-/home/tee-/.local/bin/colab}
CLI_STATE=${CLI_STATE:-/home/tee-/.config/colab-cli/db_colab_cli1.json}
LEDGER=${LEDGER:-/mnt/c/Users/15516/DISCOVERYBREAKTHROUGH/work/colab_cli_2026-09-30/CU_LEDGER.jsonl}

ts() { date -u +%FT%TZ; }
log() { echo "== $(ts) $*" | tee -a "$RUN_DIR/driver.log" >&2; }
cb() { "$COLAB_BIN" --config "$CLI_STATE" "$@"; }
# tcb SECS ARGS...: cb under an external timeout. A dropped kernel websocket can leave the CLI process hung after it
# prints "Connection was lost" (observed 2026-09-30, CLI 0.7.4).
tcb() { local s=$1; shift; timeout -k 10 "$s" "$COLAB_BIN" --config "$CLI_STATE" "$@"; }

# wait_ready SESSION: the VM reports READY before its kernel websocket accepts connections (the first exec 3 s after
# `new` lost its connection; after a 20 s settle it worked). Poll a trivial exec until it answers.
wait_ready() {
  local i
  for i in 1 2 3 4 5 6 7 8; do
    sleep 15
    if echo 'print("READY")' | tcb 90 exec -s "$1" --timeout 60 2>>"$RUN_DIR/driver.log" | grep -aq '^READY'; then
      log "kernel ready after attempt $i"
      return 0
    fi
    log "kernel not ready (attempt $i)"
  done
  return 1
}

# timed CMD...: run, log exit code and wall seconds.
timed() {
  local t0 t1 rc
  t0=$(date +%s.%N)
  "$@"
  rc=$?
  t1=$(date +%s.%N)
  log "rc=$rc secs=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}') :: $*"
  return $rc
}

# ledger NOTE: append the raw `colab usage` reading to the CU ledger (append-only JSONL).
ledger() {
  local raw
  raw=$(cb usage 2>&1)
  python3 - "$LEDGER" "$1" "$raw" <<'PY'
import json, re, sys, datetime
path, note, raw = sys.argv[1], sys.argv[2], sys.argv[3]
num = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, raw))
row = {"utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "available_cu": num(r"balance:\s*([0-9.]+)"), "usage_rate_cu_per_h": num(r"rate:\s*([0-9.]+)"),
       "active_assignments": num(r"assignments:\s*([0-9]+)"), "note": note, "raw": raw.strip()}
with open(path, "a", newline="\n") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps(row))
PY
}
