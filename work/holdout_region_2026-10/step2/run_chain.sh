#!/usr/bin/env bash
# HOLDOUT-REGION-1 Step 2a (d): chain launcher (packet "Steps" 2).
#   OUT_DIR=/content/out bash run_chain.sh CHAIN_FILE
# CHAIN_FILE is sourced after the helpers are defined and calls, in order:
#   gate NAME CMD...            non-zero exit stops the chain with that exit code
#   stage NAME PEAK_GB CMD...   refuses unless free >= PEAK_GB + RESERVE_GB; runs CMD in its own process group,
#                               a watchdog kills the group when free < RESERVE_GB (exit 97); waits on CMD's PID only
#   step0_check REF OBS REPLAY  |OBS - REF| <= 1e-6 (declared, not overridable) and OBS bit-equal to REPLAY (exit 96)
# Group termination (repair 2026-09-30): whenever the launcher exits during a stage (watchdog 97, stage failure,
# TERM/INT/HUP to the launcher -> 128+n, any other error), it sends TERM to the stage's process group, allows GRACE_S
# (2 s), then sends KILL until /proc shows no live member or KILL_WAIT_S passes (survivors are logged), all before it
# returns; a leader that exits early no longer cancels the KILL. Processes that leave the group (setsid/setpgid) are
# not covered. After a successful stage, remaining members of its group are left running (unchanged). CHAIN_FILE
# must not replace the EXIT/TERM/INT/HUP traps.
# Environment: OUT_DIR (guarded output volume, required); RESERVE_GB (20); WATCH_S (15); FREE_PROBE (command printing
# free GB; default df on OUT_DIR); LEDGER_CMD (command printing the compute ledger; read at start and end);
# KILL_WAIT_S (10).
set -euo pipefail

OUT_DIR=${OUT_DIR:?set OUT_DIR to the guarded output volume}
RESERVE_GB=${RESERVE_GB:-20}
WATCH_S=${WATCH_S:-15}
FREE_PROBE=${FREE_PROBE:-}
LEDGER_CMD=${LEDGER_CMD:-}
KILL_WAIT_S=${KILL_WAIT_S:-10}
LOG=${CHAIN_LOG:-$OUT_DIR/chain.log}
readonly STEP0_TOL=1e-6 DISK_EXIT=95 STEP0_EXIT=96 WATCHDOG_EXIT=97 GRACE_S=2
STAGE_NAME="" STAGE_PGID="" STAGE_WPID=""   # the running stage: terminated by on_exit if the launcher exits

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG" >&2; }

numeric() { [[ ${1:-} =~ ^[0-9]+([.][0-9]+)?$ ]]; }

ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 >= b + 0) }'; }

nap() { sleep "$1" </dev/null >/dev/null 2>&1 || true; }   # an orphaned nap holds no launcher pipe

free_gb() {
  if [[ -n $FREE_PROBE ]]; then
    bash -c "$FREE_PROBE"
  else
    df -P -B1 "$OUT_DIR" | awk 'NR == 2 { printf "%.3f\n", $4 / 1e9 }'
  fi
}

ledger() {
  if [[ -n $LEDGER_CMD ]]; then
    log "LEDGER $1: $(bash -c "$LEDGER_CMD" 2>&1 | tr '\n' ' ')"
  else
    log "LEDGER $1: LEDGER_CMD not set"
  fi
}

gate() {
  local name=$1
  shift
  log "GATE $name: $*"
  local rc=0
  "$@" || rc=$?
  if ((rc != 0)); then
    log "GATE $name FAIL rc=$rc; chain stops"
    exit "$rc"
  fi
  log "GATE $name PASS"
}

# Live (non-zombie) PIDs in process group $1, plus PID $1 itself (e.g. before its setsid); /proc read with builtins
# only, so it works on a full disk.
group_members() {
  local d s st grp
  for d in /proc/[0-9]*; do
    { read -r s <"$d/stat"; } 2>/dev/null || continue
    s=${s##*) }                          # after "pid (comm) ": state ppid pgrp ...
    st=${s%% *} s=${s#* } s=${s#* } grp=${s%% *}
    if [[ ($grp == "$1" || ${d#/proc/} == "$1") && $st != Z ]]; then printf '%s ' "${d#/proc/}"; fi
  done
}

terminate_group() {  # TERM, up to GRACE_S for a clean exit, then KILL until no live member (bounded by KILL_WAIT_S)
  local pg=$1 name=$2 live i
  kill -TERM -- "-$pg" 2>/dev/null || kill -TERM "$pg" 2>/dev/null || true
  live=$(group_members "$pg")
  for ((i = 0; i < GRACE_S * 10; i++)); do
    [[ -n $live ]] || break
    nap 0.1
    live=$(group_members "$pg")
  done
  for ((i = 0; i < KILL_WAIT_S * 10; i++)); do
    [[ -n $live ]] || break
    kill -KILL -- "-$pg" 2>/dev/null || kill -KILL "$pg" 2>/dev/null || true
    nap 0.1
    live=$(group_members "$pg")
  done
  if [[ -n $live ]]; then
    log "STAGE $name: process group $pg still has live members after KILL: $live" || true
    return 1
  fi
  log "STAGE $name: process group $pg has no live members" || true
}

watchdog() {  # background subshell; the stage stops it once the leader has exited and then owns termination
  local pid=$1 name=$2 flag=$3 f
  while kill -0 "$pid" 2>/dev/null; do
    f=$(free_gb 2>/dev/null) || f=""
    if ! numeric "$f" || ! ge "$f" "$RESERVE_GB"; then
      # flag first (the stage reads it); on a full disk the flag write or the log may fail, never the kill
      { printf '%s\n' "${f:-unreadable}" >"$flag"; } 2>/dev/null || true
      log "WATCHDOG $name: free '${f:-unreadable}' GB < reserve $RESERVE_GB GB; killing process group $pid" || true
      terminate_group "$pid" "$name" || true
      return 0
    fi
    nap "$WATCH_S"
  done
}

stop_watchdog() {  # KILL and reap the stage's watchdog; a watchdog stopped mid-escalation is completed by on_exit
  local wpid=$STAGE_WPID
  STAGE_WPID=""
  if [[ -n $wpid ]]; then
    kill -KILL "$wpid" 2>/dev/null || true
    wait "$wpid" 2>/dev/null || true
  fi
}

stage() {
  local name=$1 peak=$2 f need rc=0
  shift 2
  f=$(free_gb) || f=""
  need=$(awk -v p="$peak" -v r="$RESERVE_GB" 'BEGIN { print p + r }')
  if ! numeric "$f" || ! ge "$f" "$need"; then
    log "STAGE $name REFUSED: free '${f:-unreadable}' GB < peak $peak + reserve $RESERVE_GB GB"
    exit "$DISK_EXIT"
  fi
  local flag="$OUT_DIR/.watchdog_$name"
  rm -f "$flag"
  log "STAGE $name START free=$f GB peak=$peak GB: $*"
  STAGE_NAME=$name
  setsid "$@" &
  local pid=$!
  STAGE_PGID=$pid
  watchdog "$pid" "$name" "$flag" &
  STAGE_WPID=$!
  wait "$pid" || rc=$?
  stop_watchdog
  if [[ -e $flag ]]; then   # exit runs on_exit, which completes TERM -> KILL on the whole group before returning
    log "STAGE $name KILLED by watchdog (rc=$rc); chain stops" || true
    exit "$WATCHDOG_EXIT"
  fi
  if ((rc != 0)); then
    log "STAGE $name FAIL rc=$rc; chain stops" || true
    exit "$rc"
  fi
  STAGE_NAME="" STAGE_PGID=""
  log "STAGE $name DONE"
}

step0_check() {
  local rc=0
  python3 - "$STEP0_TOL" "${1:-}" "${2:-}" "${3:-}" <<'PY' || rc=$?
import math, struct, sys
tol, ref, obs, rep = sys.argv[1:5]
def val(x, what):
    s = x.strip()                      # decimal or float.hex(); raises on empty or garbage input
    v = float.fromhex(s) if s.lower().lstrip("+-").startswith("0x") else float(s)
    if not math.isfinite(v):
        raise SystemExit(f"step0: {what} is not finite: {x!r}")
    return v
t, r, o, p = val(tol, "tolerance"), val(ref, "reference"), val(obs, "observed"), val(rep, "replay")
if not abs(o - r) <= t:
    raise SystemExit(f"step0: |dloss| = {abs(o - r):.3e} > {t:g} (ref {r!r}, observed {o!r})")
if struct.pack("<d", o) != struct.pack("<d", p):
    raise SystemExit(f"step0: replay not bit-equal ({o.hex()} vs {p.hex()})")
print(f"step0 PASS |dloss|={abs(o - r):.3e} replay={o.hex()}")
PY
  if ((rc != 0)); then
    log "STEP0 FAIL (ref='${1:-}' observed='${2:-}' replay='${3:-}'); chain stops"
    exit "$STEP0_EXIT"
  fi
  log "STEP0 PASS ref=$1 observed=$2 replay=$3 tol=$STEP0_TOL"
}

on_signal() {  # TERM/INT/HUP to the launcher: exit 128+n; on_exit then terminates the running stage's group
  trap '' TERM INT HUP
  if [[ -n $STAGE_NAME && -z $STAGE_PGID ]]; then STAGE_PGID=${!:-}; fi   # signal between launch and pid record
  log "INTERRUPTED by SIG$1${STAGE_NAME:+ during stage $STAGE_NAME}" || true
  exit "$2"
}

on_exit() {  # every exit path: finish terminating a running stage's group, then read the ledger
  local rc=$1
  trap '' TERM INT HUP
  if [[ -n $STAGE_PGID ]]; then
    stop_watchdog
    terminate_group "$STAGE_PGID" "$STAGE_NAME" || true
  fi
  trap - TERM INT HUP
  ledger end || true
  log "CHAIN EXIT rc=$rc" || true
}

main() {
  if (($# != 1)) || [[ ! -f $1 ]]; then
    echo "usage: OUT_DIR=... bash run_chain.sh CHAIN_FILE" >&2
    exit 2
  fi
  if ! [[ $KILL_WAIT_S =~ ^[1-9][0-9]*$ ]]; then
    echo "KILL_WAIT_S must be a positive integer (seconds)" >&2
    exit 2
  fi
  mkdir -p "$OUT_DIR"
  trap 'on_exit "$?"' EXIT
  trap 'on_signal TERM 143' TERM
  trap 'on_signal INT 130' INT
  trap 'on_signal HUP 129' HUP
  ledger start
  # shellcheck source=/dev/null
  source "$1"
  log "CHAIN COMPLETE"
}

main "$@"
