#!/bin/bash
# D1-COLLAPSE-1 local driver (WSL) on the official Colab CLI transport: a checked copy of ADAPT-MATCH-0's
# driver_match.sh (frozen 351b5961; itself derived from ADAPT-NDS-1's driver_adapt.sh). Unchanged: every artifact rule
# (K = 1, CU ledger and cumulative CU plan, harvest worker, per-model proposal harvest against VM-side sha256 sidecars,
# verified export, the hold with no time limit, VM-gone and RELEASE ends, abort handling, watch and rescue,
# export-check). Changed: paths and names (sed transformation, tests/test_chain.py), the session plan (sessions_d1.py:
# a list of runs with a recipe, not one arm x three seeds), the per-recipe CU ledger and ceiling, and the local disk
# rule (d1_gates.py local: C: free >= 20 GB reserve + the session's declared peak + 2 GB page-file headroom).
#   bash work/d1_collapse_2026-10-06/chain/driver_d1.sh run INDEX [PREV_TOP]   # session INDEX (PACKET "Sequence")
#   bash work/d1_collapse_2026-10-06/chain/driver_d1.sh rescue TOP             # export, then stop, unstopped VMs
#   bash work/d1_collapse_2026-10-06/chain/driver_d1.sh export-check TOP       # local smoke of the export route
# (`watch TOP PID` is internal: the detached orphan watchdog.)
# Owner rule: no VM is stopped before every final weight and result has a verified local copy (sha256 and size); a
# weight download waits while C: free minus its bytes is below 22.0 GB (the VM keeps it). The whole driver runs under
# verify/v3/archive_gate.py; every download under storage_guard's run_guarded; the launch under storage_guard check at
# the session's declared peak. CU: chain/CU_SPEND_<phase>.jsonl; monitor limit min(phase ceiling - phase spend so
# far, 1.25 x the session projection); a launch is refused below 1.10 x projection. The monitor alerts and stops nothing.
set -uo pipefail
MODE=${1:-}
case $MODE in
  run) SESSION_INDEX=${2:?usage: driver_d1.sh run INDEX [PREV_TOP]}; PREV_TOP=${3:-} ;;
  rescue | watch | export-check) RUN_TOP=${2:?usage: driver_d1.sh rescue TOP} ;;
  *) echo "usage: driver_d1.sh run INDEX [PREV_TOP] | rescue TOP | export-check TOP" >&2; exit 2 ;;
esac
ROOT=${D1_ROOT:-/mnt/c/Users/15516/DISCOVERYBREAKTHROUGH}
REL=work/d1_collapse_2026-10-06/chain
CHAIN=$ROOT/$REL
D1=$ROOT/work/d1_collapse_2026-10-06
HR1_CHAIN=$ROOT/work/holdout_region_2026-10/chain
CLI=$ROOT/work/colab_cli_2026-09-30
PY=${LOCAL_PY:-/home/tee-/adaf-h2h-venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1
OUTV=${D1_OUTV:-$ROOT/outputs}
D1_OUT=$OUTV/d1_collapse_2026-10-06
if [[ ${D1_LIB:-} != 1 && ${D1_GUARDED:-} != 1 && $MODE == run ]]; then
  cd "$ROOT" || exit 2
  exec "$PY" -B verify/v3/archive_gate.py run -- env D1_GUARDED=1 bash "$CHAIN/driver_d1.sh" "$@"
fi
cd "$ROOT" || exit 2
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [[ $MODE == run ]]; then TOP=${TOP_OVERRIDE:-$D1_OUT/chain_runs/run_s${SESSION_INDEX}_$STAMP}
else TOP=$RUN_TOP
fi
RUN_DIR=$TOP/v1
mkdir -p "$RUN_DIR"
export LEDGER=${LEDGER:-$CHAIN/CU_LEDGER.jsonl}
# shellcheck source=/dev/null
source "$CLI/cli_common.sh"                        # ts, log, cb, tcb, wait_ready, timed, ledger
CLI_STATE=${CLI_STATE:-/home/tee-/.config/colab-cli/d1_collapse_v1.json}
SESSION_BASE=${SESSION_BASE:-d1-collapse-$MODE${SESSION_INDEX:+-s$SESSION_INDEX}-$(date -u +%m%d%H%M)}
SESSION=${SESSION:-$SESSION_BASE-v1}
VM_TAG=v1
export COLAB_BIN CLI_STATE SESSION RUN_DIR
export PYTHONUNBUFFERED=1
CHAIN_CU_CEILING=${CHAIN_CU_CEILING:-}        # run: sessions_d1.py plan (recipe ceiling)
CU_SPEND=${CU_SPEND:-}                        # run: chain/CU_SPEND_<phase>.jsonl (preflight)
CU_KIND=$MODE
SESSION_CU_CAP=${SESSION_CU_CAP:-}            # run: sessions_d1.py plan (1.25 x projection)
CU_REQUIRE=${CU_REQUIRE:-}                    # run: sessions_d1.py plan (1.10 x projection)
WEIGHTS_DIR=${WEIGHTS_DIR:-$D1_OUT/weights}
LOCAL_PEAK_GB=${LOCAL_PEAK_GB:-}              # run: sessions_d1.py plan (session weights + 0.5 GB)
FREEZE=${FREEZE:-$D1/freeze/FREEZE.json}
VM_ROOT=${VM_ROOT:-/content/DB/outputs/d1_collapse_2026-10-06/vm}
VM_RESULTS=$VM_ROOT/results
WEIGHT_BYTES_HINT=${WEIGHT_BYTES_HINT:-330000000}
WALL_CAP_S=${WALL_CAP_S:-}
MON_PID=""
HARVEST_PID=""

vm_exec() { local t=${1:-60}; tcb $((t + 60)) exec -s "$SESSION" --timeout "$t"; }
die() { log "STOP: $1"; exit "${2:-1}"; }
sha_of() { sha256sum "$1" | cut -d' ' -f1; }
json_get() { "$PY" -c 'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else v)' "$1" "$2"; }
stop_session() { timeout -k 10 120 "$COLAB_BIN" --config "$2" stop -s "$1" >>"$TOP/driver.log" 2>&1; }
alert() { echo "$(ts) $*" >>"$TOP/ALERT.txt"; log "ALERT: $*"; }

fault() {  # fault KIND: 0 (inject) for the first N calls of KIND named in FAULT_INJECT="KIND:N ..." (export-check only)
  local kind=$1 spec n c f
  [[ -n ${FAULT_INJECT:-} ]] || return 1
  for spec in $FAULT_INJECT; do
    [[ ${spec%%:*} == "$kind" ]] || continue
    n=${spec#*:}
    mkdir -p "$TOP/faults"
    f=$TOP/faults/$kind
    exec {c}>>"$f.lock"
    flock "$c"
    touch "$f"
    if (($(wc -l <"$f") < n)); then
      echo "$(ts) ${RUN_DIR##*/}" >>"$f"
      exec {c}>&-
      return 0
    fi
    exec {c}>&-
  done
  return 1
}

unexported() { [[ -f $RUN_DIR/EXEC_STARTED && ! -f $RUN_DIR/EXPORT_DONE && ! -f $RUN_DIR/LOSS.json ]]; }

cleanup() {  # parent EXIT (owner rule): a session with unexported artifacts is never stopped here. After an abort it is
  # kept (KEPT.json) and DRIVER_DONE is not written, so the watchdog's rescue exports and then stops it; a session the
  # operator released stays up; any other session the VM subshell left running is rescued now
  local name state hand=0
  trap '' INT TERM HUP
  [[ -n $MON_PID ]] && kill "$MON_PID" 2>/dev/null
  if [[ -f $TOP/ABORT ]]; then
    # shellcheck disable=SC2046
    kill $(jobs -p) 2>/dev/null
  fi
  if [[ -s $TOP/LAUNCHED ]]; then
    while read -r name state; do
      grep -qxF "$name" "$TOP/STOPPED" 2>/dev/null && continue
      if grep -qxF "$name" "$TOP/KEPT" 2>/dev/null; then
        log "$name kept running (KEPT.json): not stopped"
        [[ -f $TOP/ABORT ]] && hand=1
      elif [[ -f $TOP/ABORT ]]; then
        if unexported; then
          SESSION=$name record_pending kept ABORT
          alert "ABORT: $name kept (export not verified); the watchdog's rescue exports and then stops it"
          hand=1
        else
          echo "== $(ts) abort: stop $name (nothing unexported)" >>"$TOP/driver.log"
          stop_session "$name" "$state"
          echo "$name" >>"$TOP/STOPPED"
        fi
      else
        (rescue_session "$name" "$state")
      fi
    done <"$TOP/LAUNCHED"
    tcb 120 sessions >"$TOP/sessions_after.txt" 2>&1
    ledger "d1-collapse $MODE after stop ($SESSION_BASE)" >/dev/null
    cu_record
  fi
  if ((hand == 1)); then
    alert "driver ends with a kept VM: DRIVER_DONE not written; the watchdog's rescue takes over"
  else
    : >"$TOP/DRIVER_DONE"
  fi
}

on_signal() { : >"$TOP/ABORT"; alert "driver aborted by signal $1: a VM with unexported artifacts is kept (owner rule), then exported and stopped by the watchdog's rescue"; exit "$2"; }

# ---------------------------------------------------------------- CU: cumulative budget (JOINT-2 cu_plan / cu_record)
cu_plan() {  # cu_plan START_BALANCE -> TOP/CU_PLAN.json; prints this attempt's monitor limit
  "$PY" - "$CU_SPEND" "${TOP##*/}" "$CU_KIND" "$1" "$SESSION_CU_CAP" "$CHAIN_CU_CEILING" "$TOP/CU_PLAN.json" <<'PY'
import datetime, json, os, sys
path, attempt, kind, start, cap, ceiling, out = sys.argv[1:8]
start, cap, ceiling = float(start), float(cap), float(ceiling)
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
rows = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()] if os.path.exists(path) else []
if rows and rows[-1].get("end_balance") is not None and start < rows[-1]["end_balance"] - 0.005:
    last = rows[-1]
    lag = {"attempt": "lag_before_" + attempt, "kind": last["kind"], "start_balance": last["end_balance"],
           "end_balance": start, "spent": round(last["end_balance"] - start, 2), "utc": now,
           "note": "balance at the next launch is below the last recorded end of " + last["attempt"] +
                   " (accounting lag or other account use); charged to the chain conservatively"}
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(lag) + "\n")
    rows.append(lag)
per = {}
for r in rows:
    per[r["attempt"]] = r
all_spent = round(sum(r["spent"] for r in per.values()), 2)
limit = min(cap, ceiling - all_spent)
plan = {"attempt": attempt, "kind": kind, "start_balance": start, "session_cap": cap, "chain_ceiling": ceiling,
        "earlier_chain_spent": all_spent, "limit": round(limit, 2), "utc": now, "earlier_attempts": sorted(per)}
json.dump(plan, open(out, "w"), indent=1)
print(round(limit, 2))
PY
}

cu_record() {  # append this attempt's measured spend (start balance - latest ledger reading) to CU_SPEND.jsonl
  [[ -f $TOP/CU_PLAN.json ]] || return 0
  "$PY" - "$CU_SPEND" "$TOP/CU_PLAN.json" "$LEDGER" "$TOP" <<'PY'
import datetime, json, sys
path, plan, ledger, top = sys.argv[1:5]
p = json.load(open(plan))
end = None
for ln in open(ledger, encoding="utf-8"):
    r = json.loads(ln)
    if r.get("available_cu") is not None and r.get("utc", "") >= p["utc"]:
        end = r["available_cu"]
if end is None:
    raise SystemExit(0)
spent = round(p["start_balance"] - end, 2)
row = {"attempt": p["attempt"], "kind": p["kind"], "start_balance": p["start_balance"], "end_balance": end,
       "spent": spent, "limit": p["limit"], "overrun": round(max(0.0, spent - p["limit"]), 2), "source": top,
       "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "note": "last ledger reading after the stop; a later lag row may add to it; overrun > 0 = the VM was kept "
               "past the CU limit until its artifacts were verified (ADAPT-NDS-1 BUILD_DECISIONS.md decision 2)"}
with open(path, "a", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(row) + "\n")
PY
}

cu_monitor() {  # cu_monitor START LIMIT: account-wide spend since START. Owner rule (ADAPT-NDS-1 BUILD_DECISIONS.md decision 2): at
  # LIMIT it alerts and records the overrun (TOP/CU_OVERRUN.jsonl at the first crossing and at every further
  # CU_OVERRUN_STEP, 2 CU); it stops nothing - the VM stays up and the downloads continue
  local start=$1 limit=$2 now spent over next=""
  while sleep "${CU_POLL_S:-120}"; do
    now=$(tcb 120 usage 2>/dev/null | grep -oP 'balance:\s*\K[0-9.]+' | head -1)
    [[ -n $now ]] || continue
    spent=$(awk -v a="$start" -v b="$now" 'BEGIN { printf "%.2f", a - b }')
    echo "== $(ts) cu_monitor spent=$spent limit=$limit" >>"$TOP/driver.log"
    awk -v s="$spent" -v b="$limit" 'BEGIN { exit !(s >= b) }' || continue
    over=$(awk -v s="$spent" -v b="$limit" 'BEGIN { printf "%.2f", s - b }')
    if [[ -z $next ]] || awk -v o="$over" -v n="$next" 'BEGIN { exit !(o >= n) }'; then
      printf '{"attempt": "%s", "session": "%s", "spent": %s, "limit": %s, "overrun": %s, "utc": "%s", "vm_kept": true, "rule": "ADAPT-NDS-1 BUILD_DECISIONS.md decision 2"}\n' \
        "${TOP##*/}" "$SESSION" "$spent" "$limit" "$over" "$(ts)" >>"$TOP/CU_OVERRUN.jsonl"
      alert "CU OVERRUN: spent $spent >= limit $limit (+$over CU); the VM is kept until every weight and result has a verified local copy; downloads continue"
      next=$(awk -v o="$over" -v d="${CU_OVERRUN_STEP:-2}" 'BEGIN { printf "%.2f", o + d }')
    fi
  done
}

# ---------------------------------------------------------------- preflight
storage_check() {
  "$PY" -B "$ROOT/verify/v3/storage_guard.py" check --path "$OUTV" --peak-gb "$LOCAL_PEAK_GB" >"$TOP/STORAGE_CHECK.json" \
    || die "storage check refused (free space below LOCAL_PEAK_GB $LOCAL_PEAK_GB + 20 GB): $(cat "$TOP/STORAGE_CHECK.json")" 73
  "$PY" -B "$CHAIN/d1_gates.py" local "$TOP/LOCAL_CHECK.json" "$LOCAL_PEAK_GB" >/dev/null \
    || die "local disk rule refused (C: free < 20 GB reserve + declared peak + 2 GB page-file headroom: archive verified weights first): $(cat "$TOP/LOCAL_CHECK.json")" 73
  log "storage check PASS: $(cat "$TOP/STORAGE_CHECK.json") $(cat "$TOP/LOCAL_CHECK.json" | tr -d '\n ')"
}

head_check() {  # every pinned tile URL this session reads answers with its pinned size (FIT: and ETag = md5)
  "$PY" -B "$CHAIN/d1_gates.py" head "$TOP/HEAD_CHECK.json" fit="$ROOT/$LS_PLAN_REL" dev="$ROOT/$DEV_PLAN_REL" \
    >>"$TOP/driver.log" 2>&1 || die "HEAD check of the pinned tiles failed (HEAD_CHECK.json)" 4
  log "HEAD check PASS: every pinned DGM1 tile (Lower Saxony FIT, Verden DEV) answers with its pinned size"
}

preflight() {
  local o=outputs/adapt_nds_2026-10-05 f=outputs/fresh_region_2026-10-03 plan proj
  LS_PLAN_REL=$o/fit/FIT_STAGING_PINNED_PRIVATE.json LS_FIT_REL=$o/fit/FIT_TABLE_PRIVATE.json
  LS_MASKS_REL=$o/fit/FIT_INSIDE_MASKS_PRIVATE.npz
  DEV_PLAN_REL=$f/step2/STAGING_PLAN_PRIVATE.json DEV_MANIFEST_REL=$f/step2/INFER_MANIFEST_PRIVATE.json
  DEV_ENC_MANIFEST_REL=$f/step2/ENCLOSURE_MANIFEST_PRIVATE.json DEV_MASKS_REL=$f/step2/INSIDE_MASKS_PRIVATE.npz
  DEV_TILE_INDEX_REL=$f/step0/TILE_INDEX_PRIVATE.json DEV_TILE_LEDGER_REL=$f/step0/TILE_LEDGER_PRIVATE.jsonl
  [[ -z ${FAULT_INJECT:-} ]] || die "FAULT_INJECT is for export-check; run refuses it" 2
  "$PY" -B "$CHAIN/freeze_d1.py" check-run "$FREEZE" "$SESSION_INDEX" "${PREV_TOP:-}" "$TOP/BINDING.json" \
    >"$TOP/BINDING_CHECK.txt" 2>&1 || die "run refused: freeze binding check failed (BINDING_CHECK.txt)" 20
  plan=$("$PY" -B "$CHAIN/sessions_d1.py" plan "$FREEZE" "$SESSION_INDEX" ${PREV_TOP:+--previous "$PREV_TOP"} \
    2>>"$TOP/BINDING_CHECK.txt") || die "no session plan (BINDING_CHECK.txt)" 20
  eval "$plan"
  CU_REQUIRE=${CU_REQUIRE:-$PLAN_CU_REQUIRE} SESSION_CU_CAP=${SESSION_CU_CAP:-$PLAN_SESSION_CU_CAP}
  WALL_CAP_S=${WALL_CAP_S:-$PLAN_WALL_CAP_S}
  CHAIN_CU_CEILING=${CHAIN_CU_CEILING:-$PLAN_CHAIN_CU_CEILING} LOCAL_PEAK_GB=${LOCAL_PEAK_GB:-$PLAN_LOCAL_PEAK_GB}
  CU_SPEND=${CU_SPEND:-$CHAIN/CU_SPEND_$PHASE.jsonl}
  PINS_REL=$(realpath --relative-to="$ROOT" "$TOP/PINS_SESSION.json")
  "$PY" -B "$CHAIN/freeze_d1.py" session-pins "$FREEZE" "$SESSION_INDEX" "$ROOT/$PINS_REL" \
    >>"$TOP/BINDING_CHECK.txt" 2>&1 || die "session pins failed (BINDING_CHECK.txt)" 20
  PINS_KIND=d1_run
  proj=$D1/CU_PROJECTION.json
  BASE_S=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["projection"]["per_window_base_s"])' "$proj")
  PER_S=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["projection"]["per_model_s"])' "$proj")
  head_check
  storage_check
  log "preflight: session=$SESSION_INDEX recipe=$RECIPE runs='$RUN_NAMES' infer='$INFER_NAMES'" \
    "wall_cap_s=$WALL_CAP_S session_cu_cap=$SESSION_CU_CAP cu_require=$CU_REQUIRE chain_cu_ceiling=$CHAIN_CU_CEILING" \
    "local_peak_gb=$LOCAL_PEAK_GB"
}

make_lists() {
  "$PY" -B "$CHAIN/bundle_d1.py" lists "$SESSION_INDEX" "$TOP/lists" --extra "$PINS_REL" >"$TOP/lists.json" \
    || die "bundle list refused"
  "$PY" -B "$CHAIN/freeze_d1.py" check-lists "$FREEZE" "$TOP/lists" "$PINS_REL" >>"$TOP/BINDING_CHECK.txt" 2>&1 \
    || die "bundle list differs from the frozen pins (BINDING_CHECK.txt)" 20
  log "bundles: $(cat "$TOP/lists.json")"
}

stream_bundle() {  # stream_bundle NAME: repo files -> gzip tar stream -> 48 MiB parts -> VM /content/cli/parts/
  local name=$1 t0 rcs
  t0=$(date +%s)
  BUNDLE_COMPRESSION=gzip "$PY" -B "$HR1_CHAIN/bundle_stream.py" "$TOP/lists/$name.txt" "$RUN_DIR/$name.manifest.json" \
    "$RUN_DIR/$name.sha" | split -b 48M -d -a 4 --filter="bash $CLI/upload_part.sh" - "${name}_part_"
  rcs="${PIPESTATUS[*]}"
  [[ $rcs == "0 0" ]] || die "bundle $name upload failed (rc $rcs)" 13
  log "bundle $name uploaded: $(cat "$RUN_DIR/$name.sha") secs=$(($(date +%s) - t0))"
}

write_env() {
  local d=/content/DB
  cat >"$RUN_DIR/job.env" <<EOF
MODE=run
VM_TAG=$VM_TAG
BUNDLES="code"
BUNDLE_COMPRESSION=gzip
SHA_code=$(cut -d' ' -f1 "$RUN_DIR/code.sha")
WALL_CAP_S=$WALL_CAP_S
LS_PLAN=$d/$LS_PLAN_REL
LS_FIT=$d/$LS_FIT_REL
LS_MASKS=$d/$LS_MASKS_REL
DEV_PLAN=$d/$DEV_PLAN_REL
DEV_MANIFEST=$d/$DEV_MANIFEST_REL
DEV_ENC_MANIFEST=$d/$DEV_ENC_MANIFEST_REL
DEV_MASKS=$d/$DEV_MASKS_REL
DEV_TILE_INDEX=$d/$DEV_TILE_INDEX_REL
DEV_TILE_LEDGER=$d/$DEV_TILE_LEDGER_REL
PINS=$d/$PINS_REL
PINS_KIND=$PINS_KIND
EQ1_BLOCK=SCREEN_117_54
EXPECT_GPU="${EXPECT_GPU:-RTX PRO 6000}"
SESSION_INDEX=$SESSION_INDEX
RECIPE=$RECIPE
SEEDS="$SEEDS"
RUN_NAMES="$RUN_NAMES"
INFER_NAMES="$INFER_NAMES"
BASE_S=$BASE_S
PER_S=$PER_S
WEIGHTS_SINK=keep
INFER_PROCS=1
CUDNN_ALLOW_TF32=0
EOF
}

# ---------------------------------------------------------------- export: proposals and receipts first, then weights
dl() {  # dl KIND REMOTE LOCAL PEAK_BYTES [TIMEOUT_S]: one `colab download` under the storage guard
  local kind=$1 remote=$2 dst=$3 peak=$4 t=${5:-900} gb rc t0
  gb=$(awk -v b="$peak" 'BEGIN { printf "%.3f", b / 1e9 + 0.05 }')
  if fault "$kind"; then log "FAULT_INJECT $kind: download of $remote fails"; return 75; fi
  t0=$(date +%s)
  "$PY" -B -c 'import sys; sys.path.insert(0, sys.argv[1]); import storage_guard as g
raise SystemExit(g.run_guarded(sys.argv[2], float(sys.argv[3]), sys.argv[5:], interval=float(sys.argv[4])))' \
    "$ROOT/verify/v3" "$OUTV" "$gb" "${GUARD_POLL_S:-0.5}" \
    timeout -k 10 "$t" "$COLAB_BIN" --config "$CLI_STATE" download -s "$SESSION" "$remote" "$dst" \
    </dev/null >>"$RUN_DIR/driver.log" 2>&1
  rc=$?
  log "download rc=$rc secs=$(($(date +%s) - t0)) $kind $remote"
  return "$rc"
}

fetch_verified() {  # fetch_verified KIND REMOTE LOCAL SHA256 PEAK_BYTES: up to FETCH_ATTEMPTS guarded downloads
  local kind=$1 remote=$2 dst=$3 want=$4 peak=$5 a
  [[ -f $dst && $(sha_of "$dst") == "$want" ]] && return 0
  for ((a = 1; a <= ${FETCH_ATTEMPTS:-3}; a++)); do
    rm -f "$dst.tmp"
    if dl "$kind" "$remote" "$dst.tmp" "$peak"; then
      if fault "corrupt_$kind"; then printf 'X' >>"$dst.tmp"; log "FAULT_INJECT corrupt_$kind: one byte appended"; fi
      if [[ $(sha_of "$dst.tmp") == "$want" ]]; then mv -f "$dst.tmp" "$dst"; return 0; fi
      log "sha256 mismatch for $remote (attempt $a)"
    fi
    rm -f "$dst.tmp"
  done
  log "fetch of $remote failed after ${FETCH_ATTEMPTS:-3} attempts"
  return 16
}

download_results() {  # results parts (sha256 per part and for the tar) -> RUN_DIR/results; 15 = no index on the VM
  local sha name
  [[ -f $RUN_DIR/RESULTS_EXPORTED ]] && return 0
  mkdir -p "$RUN_DIR/parts"
  for name in INDEX.sha256 RES_TAR.sha256; do
    rm -f "$RUN_DIR/parts/$name"
    if ! dl index "/content/res_parts/$name" "$RUN_DIR/parts/$name" 1000000 300; then
      log "no result index on the VM ($name)"
      return 15
    fi
  done
  while read -r sha name; do
    [[ $name == res.tar.part_* ]] || continue
    fetch_verified res_part "/content/res_parts/$name" "$RUN_DIR/parts/$name" "$sha" 52000000 || return 16
  done <"$RUN_DIR/parts/INDEX.sha256"
  cat "$RUN_DIR"/parts/res.tar.part_* >"$RUN_DIR/res.tar"
  if [[ $(sha_of "$RUN_DIR/res.tar") != "$(cat "$RUN_DIR/parts/RES_TAR.sha256")" ]]; then
    log "res.tar sha mismatch"
    rm -f "$RUN_DIR/res.tar"
    return 16
  fi
  tar -xf "$RUN_DIR/res.tar" -C "$RUN_DIR" || { log "res.tar unpack failed"; return 16; }
  sha_of "$RUN_DIR/res.tar" >"$RUN_DIR/RESULTS_EXPORTED"
  rm -f "$RUN_DIR/res.tar" "$RUN_DIR"/parts/res.tar.part_*
  log "results in $RUN_DIR/results (res.tar sha256 $(cat "$RUN_DIR/RESULTS_EXPORTED"))"
}

weights_room() {  # 0 when C: free minus one weight's bytes stays >= 22.0 GB (20 GB reserve + 2 GB; else deferred)
  "$PY" -B "$CHAIN/d1_gates.py" weights-room "${1:-$WEIGHT_BYTES_HINT}" >>"$RUN_DIR/driver.log" 2>&1
}

train_result() {  # train_result RUN: path of the run's TRAIN_RESULT.json (downloaded results, else fetched from the VM)
  local r=$1
  if [[ -f $RUN_DIR/results/runs/$r/TRAIN_RESULT.json ]]; then echo "$RUN_DIR/results/runs/$r/TRAIN_RESULT.json"; return 0; fi
  mkdir -p "$WEIGHTS_DIR/$r"
  if [[ ! -s $WEIGHTS_DIR/$r/TRAIN_RESULT.json ]]; then
    dl harvest_meta "$VM_ROOT/runs/$r/TRAIN_RESULT.json" "$WEIGHTS_DIR/$r/TRAIN_RESULT.json" 1000000 300 || return 1
  fi
  echo "$WEIGHTS_DIR/$r/TRAIN_RESULT.json"
}

weight_verified() {
  local d=$WEIGHTS_DIR/$1
  [[ -f $d/VERIFIED.json && -f $d/final.pt ]] || return 1
  [[ $(stat -c %s "$d/final.pt") == "$(json_get "$d/VERIFIED.json" bytes)" ]]
}

mark_verified() {  # mark_verified RUN TRAIN_RESULT SOURCE: final.pt against final_sha256 and final_bytes -> VERIFIED.json
  "$PY" - "$WEIGHTS_DIR/$1" "$2" "$3" "$VM_TAG" "$SESSION" <<'PY'
import datetime, hashlib, json, sys
from pathlib import Path
d, tr, source, vm, session = Path(sys.argv[1]), json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4], sys.argv[5]
f = d / "final.pt"
h = hashlib.sha256()
with open(f, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 22), b""):
        h.update(b)
size = f.stat().st_size
ok = h.hexdigest() == tr["final_sha256"] and tr.get("final_bytes") in (None, size)
if not ok:
    print(f"{f}: sha256 {h.hexdigest()} / {size} B differ from TRAIN_RESULT.json", file=sys.stderr)
    raise SystemExit(1)
rec = {"run": d.name, "sha256": tr["final_sha256"], "bytes": size, "bytes_in_train_result": tr.get("final_bytes"),
       "source": source, "vm": vm, "session": session,
       "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
json.dump(rec, open(d / "VERIFIED.json", "w"), indent=1)
print(json.dumps(rec))
PY
}

index_weight_runs() { sed -n 's/^[0-9a-f]\{64\}  w_\(.*\)\.pt\.part_[0-9]*$/\1/p' "$RUN_DIR/parts/INDEX.sha256" 2>/dev/null | sort -u; }

export_weights() {  # every packaged run's final.pt -> WEIGHTS_DIR/<run>/final.pt, verified; 19 = deferred (no room)
  local r tr sha name lock
  while read -r r; do
    [[ -n $r ]] || continue
    weight_verified "$r" && continue
    weights_room || { log "WEIGHTS DEFERRED: C: free would fall below 22.0 GB with $r; it stays on the VM"; return 19; }
    tr=$(train_result "$r") || { log "no TRAIN_RESULT.json for $r"; return 16; }
    mkdir -p "$WEIGHTS_DIR/$r"
    exec {lock}>>"$TOP/.weights.lock"
    flock "$lock"
    while read -r sha name; do
      [[ $name == w_"$r".pt.part_* ]] || continue
      fetch_verified w_part "/content/res_parts/$name" "$RUN_DIR/parts/$name" "$sha" 52000000 \
        || { exec {lock}>&-; return 16; }
    done <"$RUN_DIR/parts/INDEX.sha256"
    cat "$RUN_DIR"/parts/w_"$r".pt.part_* >"$WEIGHTS_DIR/$r/final.pt.partial"
    mv -f "$WEIGHTS_DIR/$r/final.pt.partial" "$WEIGHTS_DIR/$r/final.pt"
    if mark_verified "$r" "$tr" export >>"$RUN_DIR/driver.log" 2>&1; then
      rm -f "$RUN_DIR"/parts/w_"$r".pt.part_*
      log "weights $r verified -> $WEIGHTS_DIR/$r/final.pt"
    else
      rm -f "$WEIGHTS_DIR/$r/final.pt" "$RUN_DIR"/parts/w_"$r".pt.part_*
      exec {lock}>&-
      log "weights $r: reassembled file differs from TRAIN_RESULT.json"
      return 16
    fi
    exec {lock}>&-
  done < <(index_weight_runs)
  for tr in "$RUN_DIR"/results/runs/*/TRAIN_RESULT.json; do   # trained runs whose weights were not packaged
    [[ -f $tr ]] || continue
    r=$(basename "$(dirname "$tr")")
    index_weight_runs | grep -qxF "$r" && continue
    weight_verified "$r" && continue
    weights_room || { log "WEIGHTS DEFERRED: C: free would fall below 22.0 GB with $r"; return 19; }
    harvest_run "$r" direct || { log "weights $r: not packaged and not retrievable from the VM's runs dir"; return 16; }
  done
  return 0
}

export_record() {  # RUN_DIR/EXPORT.json: results exported, harvested proposals listed, every trained final.pt verified
  "$PY" - "$RUN_DIR" "$WEIGHTS_DIR" "$VM_TAG" "$SESSION" <<'PY'
import datetime, json, re, sys
from pathlib import Path
run_dir, wdir, vm, session = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
runs = {}
res = run_dir / "results" / "runs"
if res.is_dir():
    for d in sorted(res.iterdir()):
        if (d / "TRAIN_RESULT.json").is_file():
            runs[d.name] = json.load(open(d / "TRAIN_RESULT.json"))
idx = run_dir / "parts" / "INDEX.sha256"
if idx.is_file():
    for m in re.finditer(r"  w_(.+)\.pt\.part_\d+$", idx.read_text(), re.M):
        t = wdir / m.group(1) / "TRAIN_RESULT.json"
        runs.setdefault(m.group(1), json.load(open(t)) if t.is_file() else {})
rows = {}
for name, tr in sorted(runs.items()):
    v = wdir / name / "VERIFIED.json"
    if v.is_file() and (wdir / name / "final.pt").is_file() and \
            (wdir / name / "final.pt").stat().st_size == json.load(open(v))["bytes"] \
            and json.load(open(v))["sha256"] == tr.get("final_sha256"):
        rows[name] = {"status": "VERIFIED", **{k: json.load(open(v))[k] for k in ("sha256", "bytes", "source")},
                      "path": str(wdir / name / "final.pt")}
    else:
        rows[name] = {"status": "MISSING", "sha256": tr.get("final_sha256")}
harvested = sorted(p.relative_to(run_dir / "harvest").as_posix() for p in (run_dir / "harvest").glob("*/*.json")) \
    if (run_dir / "harvest").is_dir() else []
results = (run_dir / "RESULTS_EXPORTED").is_file()
ok = results and all(r["status"] != "MISSING" for r in rows.values())
rec = {"vm": vm, "session": session, "status": "VERIFIED" if ok else "INCOMPLETE", "weights_sink": "local",
       "results_tar_sha256": (run_dir / "RESULTS_EXPORTED").read_text().strip() if results else None,
       "harvested_verified": harvested, "runs": rows,
       "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
json.dump(rec, open(run_dir / "EXPORT.json", "w"), indent=1)
raise SystemExit(0 if ok else 1)
PY
}

export_vm() {
  local rc=0
  download_results || return $?
  export_weights || return $?
  export_record || rc=16
  return "$rc"
}

job_ended() { grep -aq '^REMOTE_EXIT=' "$RUN_DIR/job.log" 2>/dev/null; }
chain_started() { grep -aqE '^== [0-9TZ:-]+ chain run$' "$RUN_DIR/job.log" 2>/dev/null; }
job_running() {
  echo 'print("KERNEL_IDLE")' | tcb 100 exec -s "$SESSION" --timeout 90 2>>"$RUN_DIR/driver.log" \
    | grep -aq '^KERNEL_IDLE' && return 1
  return 0
}

# ---------------------------------------------------------------- harvest: proposals as they are ready, then weights
ready_models() { grep -aoE 'PROPOSALS_READY [a-z]+/[A-Za-z0-9_]+' "$RUN_DIR/job.log" 2>/dev/null | sed -E 's/^PROPOSALS_READY //' | sort -u; }
ready_runs() { grep -aoE 'WEIGHTS_READY [A-Za-z0-9_]+' "$RUN_DIR/job.log" 2>/dev/null | sed -E 's/^WEIGHTS_READY //' | sort -u; }

harvest_model() {  # harvest_model TAG/NAME: proposals (and the consume record for LGLN passes) against their sidecars
  local tn=$1 tag=${1%%/*} n=${1#*/} f want files
  files=("$tag/PROPOSALS_${n}_PRIVATE.json")
  [[ $tag == confirm || $tag == dev ]] && files+=("$tag/CONSUME_${n}.json")
  for f in "${files[@]}"; do
    mkdir -p "$RUN_DIR/harvest/$tag"
    [[ -f $RUN_DIR/harvest/$f && -f $RUN_DIR/harvest/$f.sha256 && \
      $(sha_of "$RUN_DIR/harvest/$f") == "$(cat "$RUN_DIR/harvest/$f.sha256")" ]] && continue
    rm -f "$RUN_DIR/harvest/$f.sha256"
    dl harvest "$VM_RESULTS/$f.sha256" "$RUN_DIR/harvest/$f.sha256" 1000000 300 || return 1
    want=$(cut -c1-64 "$RUN_DIR/harvest/$f.sha256")
    [[ $want =~ ^[0-9a-f]{64}$ ]] || { rm -f "$RUN_DIR/harvest/$f.sha256"; return 1; }
    echo "$want" >"$RUN_DIR/harvest/$f.sha256"
    FETCH_ATTEMPTS=1 fetch_verified harvest "$VM_RESULTS/$f" "$RUN_DIR/harvest/$f" "$want" 200000000 || return 1
  done
  log "HARVEST: $tn proposals verified -> $RUN_DIR/harvest"
}

harvest_run() {  # harvest_run RUN [SOURCE]: final.pt straight from the VM's runs dir, verified
  local r=$1 source=${2:-harvest} tr want lock rc=0
  mkdir -p "$WEIGHTS_DIR/$r"
  tr=$(train_result "$r") || return 1
  want=$(json_get "$tr" final_sha256)
  [[ $want =~ ^[0-9a-f]{64}$ ]] || return 1
  exec {lock}>>"$TOP/.weights.lock"
  flock "$lock"
  FETCH_ATTEMPTS=1 fetch_verified harvest "$VM_ROOT/runs/$r/final.pt" "$WEIGHTS_DIR/$r/final.pt" "$want" \
    "$WEIGHT_BYTES_HINT" || rc=1
  exec {lock}>&-
  ((rc == 0)) || return 1
  mark_verified "$r" "$tr" "$source" >>"$RUN_DIR/driver.log" 2>&1 || { rm -f "$WEIGHTS_DIR/$r/final.pt"; return 1; }
  log "HARVEST ($source): $r verified -> $WEIGHTS_DIR/$r/final.pt"
}

harvest_pending() {
  local n r
  for n in $(ready_models); do harvest_model "$n" || log "HARVEST: $n not retrieved in this round"; done
  for r in $(ready_runs); do
    weight_verified "$r" && continue
    weights_room || { log "HARVEST: $r deferred (C: free would fall below 22.0 GB)"; continue; }
    harvest_run "$r" || log "HARVEST: $r not retrieved in this round"
  done
}

harvest_worker() {  # background: proposals first each round, then ready weights when C: has room; a failed proposal
  # harvest ends the worker (the end-of-job export carries every file); deferred weights wait for the export
  local n r seen=" " wseen=" " last=0
  while :; do
    [[ -f $RUN_DIR/EXEC_ENDED ]] && last=1
    for n in $(ready_models); do
      [[ $seen == *" $n "* ]] && continue
      sleep "${HARVEST_SETTLE_S:-5}"
      if ! harvest_model "$n"; then
        log "HARVEST: direct download of $n failed; the end-of-job export will carry it"
        : >"$RUN_DIR/HARVEST_OFF"
        return 0
      fi
      seen+="$n "
    done
    for r in $(ready_runs); do
      [[ $wseen == *" $r "* ]] && continue
      weight_verified "$r" && { wseen+="$r "; continue; }
      weights_room || { log "HARVEST: weights $r deferred (C: free would fall below 22.0 GB)"; continue; }
      sleep "${HARVEST_SETTLE_S:-5}"
      if harvest_run "$r"; then wseen+="$r "; else log "HARVEST: $r failed; the end-of-job export will fetch it"; wseen+="$r "; fi
    done
    ((last == 1)) && return 0
    sleep "${HARVEST_POLL_S:-30}"
  done
}

start_harvest() { HARVEST_PID=""; [[ ${HARVEST:-1} == 1 ]] || return 0; harvest_worker & HARVEST_PID=$!; }

stop_harvest() {
  : >"$RUN_DIR/EXEC_ENDED"
  if [[ -n ${HARVEST_PID:-} ]]; then
    [[ ${1:-} == kill ]] && kill "$HARVEST_PID" 2>/dev/null
    wait "$HARVEST_PID" 2>/dev/null
    HARVEST_PID=""
  fi
  return 0
}

vm_gone() {  # 0 when a successful `colab sessions` listing does not show SESSION (a failed listing proves nothing)
  local out
  out=$(tcb 120 sessions 2>&1) || return 1
  grep -qF -- "$SESSION" <<<"$out" && return 1
  echo "== $(ts) sessions listing without $SESSION: $(head -c 300 <<<"$out" | tr '\n' ' ')" >>"$RUN_DIR/driver.log"
  return 0
}

export_with_hold() {  # owner rule (ADAPT-NDS-1 BUILD_DECISIONS.md decision 2): the VM stays up until EXPORT.json is VERIFIED (0),
  # with no time limit. Other ends: 18 the VM is gone (LOSS.json); 20 the operator released the hold (TOP/RELEASE,
  # renamed when used; KEPT.json: the VM is left running)
  local now rc deadline tw="" last=0 lastw=0 gone=0 retry=${RESCUE_RETRY_S:-120} every=${RESCUE_ALERT_S:-1800}
  deadline=$(cat "$RUN_DIR/DEADLINE" 2>/dev/null || echo 0)
  if job_ended && ! chain_started && ! grep -aq '^REMOTE_EXIT=0' "$RUN_DIR/job.log"; then
    log "EXPORT: the job ended before its chain started; nothing to export"
    printf '{"vm": "%s", "session": "%s", "status": "NO_ARTIFACTS"}\n' "$VM_TAG" "$SESSION" >"$RUN_DIR/EXPORT.json"
    return 0
  fi
  while :; do
    if [[ -f $TOP/RELEASE ]]; then
      mv -f "$TOP/RELEASE" "$TOP/RELEASE.used_$(date -u +%Y%m%dT%H%M%SZ)"
      record_pending kept RELEASED
      alert "HOLD RELEASED $VM_TAG ($SESSION) by TOP/RELEASE: export not verified; the VM is left running (KEPT.json); 'driver_d1.sh rescue $TOP' exports, then stops it"
      return 20
    fi
    rc=0
    export_vm || rc=$?
    if ((rc == 0)); then
      log "EXPORT VERIFIED ($VM_TAG): $RUN_DIR/EXPORT.json"
      ((last > 0 || lastw > 0)) && alert "$VM_TAG: export verified after the hold"
      return 0
    fi
    now=$(date +%s)
    if ((rc == 15)) && ! job_ended && ((now < deadline)) && job_running; then
      log "EXPORT: the remote job is still running (no result index); waiting ($((deadline - now)) s to its deadline)"
      sleep "${RUNNING_POLL_S:-600}"
      continue
    fi
    if ((rc == 15)); then harvest_pending; fi
    if vm_gone; then gone=$((gone + 1)); else gone=0; fi
    if ((gone >= ${VM_GONE_CHECKS:-3})); then
      record_pending loss VM_GONE
      alert "VM GONE $VM_TAG ($SESSION): $gone successful session listings without it; export not verified; LOSS.json records what was lost"
      return 18
    fi
    if ((rc == 19)); then          # weights deferred for C: space: results and proposals are already verified locally
      [[ -n $tw ]] || tw=$now
      if ((now - lastw >= every)); then
        alert "WEIGHT HOLD $VM_TAG ($SESSION) for $((now - tw)) s: C: free < 22.0 GB + weight; the VM keeps the weights until C: has room (no time limit)"
        lastw=$now
      fi
      sleep "${WEIGHT_POLL_S:-120}"
      continue
    fi
    if ((now - last >= every)); then
      alert "RESCUE HOLD $VM_TAG ($SESSION): export failed rc=$rc; the VM stays up, retrying every $retry s with no time limit (touch $TOP/RELEASE to end the hold and leave the VM running)"
      last=$now
    fi
    sleep "$retry"
  done
}

record_pending() {  # record_pending loss|kept REASON: what is not verified locally -> LOSS.json + TOP/LOSSES.jsonl (the VM
  # is gone) or KEPT.json + TOP/KEPT.jsonl + TOP/KEPT (the VM is left running); neither is a normal outcome
  "$PY" - "$RUN_DIR" "$WEIGHTS_DIR" "$1" "$2" "$VM_TAG" "${SESSION:-}" "$TOP" <<'PY'
import datetime, json, re, sys
from pathlib import Path
run_dir, wdir, kind, reason, vm, session, top = Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:7], Path(sys.argv[7])
log = (run_dir / "job.log").read_text(errors="replace") if (run_dir / "job.log").is_file() else ""
finished = sorted(set(re.findall(r"WEIGHTS_READY ([A-Za-z0-9_]+)", log)))
ready = sorted(set(re.findall(r"PROPOSALS_READY ([a-z]+/[A-Za-z0-9_]+)", log)))
exp = json.load(open(run_dir / "EXPORT.json")) if (run_dir / "EXPORT.json").is_file() else {}
verified = sorted(n for n in finished if (wdir / n / "VERIFIED.json").is_file())
harvested = sorted({p.parent.name + "/" + p.name.split("PROPOSALS_")[1].rsplit("_PRIVATE", 1)[0]
                    for p in (run_dir / "harvest").glob("*/PROPOSALS_*_PRIVATE.json")}) \
    if (run_dir / "harvest").is_dir() else []
results = (run_dir / "RESULTS_EXPORTED").is_file()
rec = {"vm": vm, "session": session, "kind": kind, "reason": reason, "normal_outcome": False,
       "vm_left_running": kind == "kept", "results_exported": results,
       "runs_finished_on_vm": finished, "runs_verified_locally": verified,
       "runs_not_verified": [n for n in finished if n not in verified], "export_status": exp.get("status"),
       "proposals_ready": ready, "proposals_harvested_verified": harvested,
       "proposals_not_harvested": [n for n in ready if n not in harvested and not results],
       "hashes_known": {n: r.get("sha256") for n, r in exp.get("runs", {}).items() if r.get("status") == "MISSING"},
       "chain_started": bool(re.search(r"^== \S+ chain run$", log, re.M)),
       "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
name, book = ("LOSS.json", "LOSSES.jsonl") if kind == "loss" else ("KEPT.json", "KEPT.jsonl")
json.dump(rec, open(run_dir / name, "w"), indent=1)
with open(top / book, "a", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(rec) + "\n")
PY
  [[ $1 == kept ]] && echo "$SESSION" >>"$TOP/KEPT"
  log "$1 ($2) recorded in $RUN_DIR"
}

vm_stop() {  # EXIT trap of the VM subshell (owner rule): no stop call while anything is unexported. A run-mode abort
  # keeps an unexported VM for the watchdog's rescue and a released hold leaves it running (both KEPT.json); a gone VM
  # (LOSS.json) gets no stop call
  local erc=0
  trap '' INT TERM HUP
  stop_harvest kill
  if [[ $MODE == run ]] && grep -qxF "$SESSION" "$TOP/KEPT" 2>/dev/null; then
    return 0
  fi
  if unexported; then
    if [[ $MODE == run && -f $TOP/ABORT ]]; then
      record_pending kept ABORT
      alert "ABORT: $SESSION kept (export not verified); the watchdog's rescue exports and then stops it"
      return 0
    fi
    export_with_hold || erc=$?
    ((erc == 0)) && : >"$RUN_DIR/EXPORT_DONE"
    ((erc == 20)) && return 0
  fi
  if [[ -f $RUN_DIR/LOSS.json ]]; then
    log "$SESSION is gone (LOSS.json): no stop call"
  else
    timed tcb 120 stop -s "$SESSION" >>"$RUN_DIR/driver.log" 2>&1
  fi
  echo "$SESSION" >>"$TOP/STOPPED"
  ledger "d1-collapse $MODE after stop ($SESSION)" >/dev/null
}

vm_run() {  # allocate, upload, remote chain (+ harvest), verified export (hold with no time limit), stop
  local rc=0 f erc=0
  log "VM session=$SESSION recipe=$RECIPE runs='$RUN_NAMES' wall_cap_s=$WALL_CAP_S"
  echo "$SESSION $CLI_STATE" >>"$TOP/LAUNCHED"
  trap vm_stop EXIT
  timed tcb 600 new -s "$SESSION" --gpu G4 >"$RUN_DIR/new.txt" 2>&1 || die "G4 not allocated (see new.txt)" 10
  ledger "d1-collapse $MODE allocated ($SESSION)" >/dev/null
  wait_ready "$SESSION" || die "kernel not ready" 11
  echo 'import os; os.makedirs("/content/cli/parts", exist_ok=True); print("dirs ok")' \
    | vm_exec >>"$RUN_DIR/driver.log" 2>&1 || die "exec failed" 11
  stream_bundle code
  write_env
  for f in "$CHAIN/vm_job_d1.sh:/content/cli/vm_job.sh" "$HR1_CHAIN/cat_parts.py:/content/cli/cat_parts.py" \
    "$RUN_DIR/job.env:/content/cli/job.env"; do
    timed tcb 300 upload -s "$SESSION" "${f%%:*}" "${f#*:}" >>"$RUN_DIR/driver.log" 2>&1 || die "upload ${f%%:*} failed" 12
  done
  echo $(($(date +%s) + WALL_CAP_S + ${DEADLINE_MARGIN_S:-2400})) >"$RUN_DIR/DEADLINE"
  : >"$RUN_DIR/EXEC_STARTED"
  start_harvest
  timed tcb $((WALL_CAP_S + 2400)) exec -s "$SESSION" -f "$CLI/remote_run.py" --timeout $((WALL_CAP_S + 2300)) \
    --env "DB_CMD=bash /content/cli/vm_job.sh" >"$RUN_DIR/job.log" 2>&1
  stop_harvest
  grep -aq '^REMOTE_EXIT=0' "$RUN_DIR/job.log" || { rc=14; log "remote chain did not complete (see job.log)"; }
  export_with_hold || erc=$?
  if ((erc == 0)); then
    : >"$RUN_DIR/EXPORT_DONE"
  else
    rc=18                      # the VM is gone (LOSS.json) or the operator released the hold (KEPT.json)
  fi
  timed tcb 180 log -s "$SESSION" -o "$RUN_DIR/session_log.ipynb" >>"$RUN_DIR/driver.log" 2>&1
  exit "$rc"
}

# ---------------------------------------------------------------- rescue (a driver that ended without stopping)
rescue_session() {
  SESSION=$1 CLI_STATE=$2
  export SESSION CLI_STATE
  HARVEST_PID=""
  log "RESCUE $SESSION: exec started=$([[ -f $RUN_DIR/EXEC_STARTED ]] && echo yes || echo no)"
  trap vm_stop EXIT
  exit 0
}

rescue_mode() {
  local name state start limit
  [[ -s $TOP/LAUNCHED ]] || die "rescue: nothing was launched in $TOP" 2
  if [[ -f $TOP/CU_PLAN.json ]]; then
    start=$(json_get "$TOP/CU_PLAN.json" start_balance)
    limit=$(json_get "$TOP/CU_PLAN.json" limit)
    (cu_monitor "$start" "$limit") &
    MON_PID=$!
  fi
  while read -r name state; do
    grep -qxF "$name" "$TOP/STOPPED" 2>/dev/null && continue
    (rescue_session "$name" "$state")
  done <"$TOP/LAUNCHED"
  [[ -n $MON_PID ]] && kill "$MON_PID" 2>/dev/null
  MON_PID=""
  tcb 120 sessions >"$TOP/sessions_after_rescue.txt" 2>&1
  ledger "d1-collapse rescue after stop (${TOP##*/})" >/dev/null
  cu_record
  : >"$TOP/DRIVER_DONE"
}

export_check_mode() {  # local smoke of the artifact route: COLAB_BIN = colab_local.sh over an emulated /content;
  # each round harvests every ready model's proposals (sidecar-verified), then runs the export (results tar, weights
  # verified against TRAIN_RESULT.json), retried as the rescue hold does, with FAULT_INJECT transfer faults; done when
  # the export is verified and every ready model's proposals are harvested
  local n rc=1 a missing=1
  [[ $(basename "$COLAB_BIN") == colab_local.sh ]] || die "export-check needs COLAB_BIN=colab_local.sh" 2
  HARVEST_PID=""
  cp "${EXPORT_CHECK_JOB_LOG:?}" "$RUN_DIR/job.log"
  for ((a = 1; a <= ${EXPORT_CHECK_ATTEMPTS:-3}; a++)); do
    missing=0
    for n in $(ready_models); do harvest_model "$n" || { missing=1; log "HARVEST: $n not retrieved in round $a"; }; done
    rc=0
    export_vm || rc=$?
    log "export-check round $a: export rc=$rc proposals_missing=$missing"
    ((rc == 0 && missing == 0)) && break
    sleep "${EXPORT_CHECK_RETRY_S:-2}"
  done
  ((missing == 0)) || rc=16
  exit "$rc"
}

watch_mode() {
  local pid=${3:?watch needs the driver PID}
  while kill -0 "$pid" 2>/dev/null; do
    [[ -f $TOP/DRIVER_DONE ]] && exit 0
    sleep "${WATCH_POLL_S:-30}"
  done
  sleep 2
  [[ -f $TOP/DRIVER_DONE ]] && exit 0
  alert "WATCH: driver $pid ended without DRIVER_DONE; rescue of $TOP starts"
  exec bash "$CHAIN/driver_d1.sh" rescue "$TOP"
}

start_watch() {
  [[ ${D1_WATCH:-1} == 1 ]] || return 0
  setsid bash "$CHAIN/driver_d1.sh" watch "$TOP" "$$" </dev/null >>"$TOP/watch.log" 2>&1 &
  disown
}

run_session() {
  local start limit r=0
  start=$(ledger "d1-collapse $MODE before ($SESSION_BASE)" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["available_cu"])') \
    || die "usage failed (auth?)" 9
  [[ $start =~ ^[0-9.]+$ ]] || die "no balance read" 9
  limit=$(cu_plan "$start") || die "cumulative CU plan failed" 9
  log "CU: start balance $start; monitor limit $limit (CU_PLAN.json); this session needs $CU_REQUIRE with margin"
  if awk -v l="$limit" -v n="$CU_REQUIRE" 'BEGIN { exit !(l <= 0 || l < n) }'; then
    die "CU limit $limit is below what this session needs ($CU_REQUIRE); nothing is allocated" 21
  fi
  (cu_monitor "$start" "$limit") &
  MON_PID=$!
  (vm_run) || r=$?
  log "v1 exit $r"
  return "$r"
}

[[ ${D1_LIB:-} == 1 ]] && return 0
case $MODE in
  watch) watch_mode "$@" ;;
  rescue) rescue_mode; exit $? ;;
  export-check) export_check_mode ;;
esac
trap cleanup EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP
log "D1-COLLAPSE-1 driver: mode=$MODE session=$SESSION_INDEX top=$TOP"
preflight
make_lists
start_watch
rc=0
run_session || rc=$?
"$PY" -B "$CHAIN/sessions_d1.py" verify "$TOP" "$FREEZE" "$SESSION_INDEX" "$WEIGHTS_DIR" \
  ${PREV_TOP:+--previous "$PREV_TOP"} >>"$TOP/driver.log" 2>&1 || { log "session verification failed"; ((rc == 0)) && rc=6; }
log "driver exit $rc"
exit "$rc"
