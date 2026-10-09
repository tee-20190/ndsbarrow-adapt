# D1-COLLAPSE-1 chain library (Colab VM): a checked copy of ADAPT-MATCH-0's chain_lib_match.sh (frozen 351b5961).
# Sourced by chain_d1_run.sh, which HR1's frozen launcher work/holdout_region_2026-10/step2/run_chain.sh sources after
# defining log, gate, stage and step0_check. Unchanged: Lower Saxony FIT staging (nds_reader.py + HR1 masks.py),
# FRESH-1's Verden DEV staging (lgln_reader.py), void, step 0, parity, the keep sink, one shared inference pass with
# HR3 infer_irl.py, consume checks with fresh_gates.py and sha256 sidecars before PROPOSALS_READY. Changed: the trainer
# (d1_train.py), the session's runs (RUN_NAMES, trained in order, each announced WEIGHTS_READY; curves.jsonl copied
# into results), the DEV pass over INFER_NAMES only (the probe P0 trains without inference), no local smoke mode.
# Environment (vm_job_d1.sh sets it from job.env): PY CHAIN_DIR D1_DIR NDS_DIR HR1_CHAIN HR3_CHAIN FRESH_DIR
#   CHAIN_FILE CHAIN_MODE VM RESULTS PINS PINS_KIND FREEZE_TXT EQ1_BLOCK LS_PLAN LS_FIT LS_MASKS DEV_PLAN DEV_MANIFEST
#   DEV_ENC_MANIFEST DEV_MASKS DEV_TILE_INDEX DEV_TILE_LEDGER SESSION_INDEX; run: RECIPE SEEDS RUN_NAMES INFER_NAMES
#   BASE_S PER_S.
: "${PY:?}" "${CHAIN_DIR:?}" "${D1_DIR:?}" "${NDS_DIR:?}" "${HR1_CHAIN:?}" "${HR3_CHAIN:?}" "${FRESH_DIR:?}"
: "${CHAIN_FILE:?}" "${CHAIN_MODE:?}" "${VM:?}" "${RESULTS:?}"
declare -gA PASSED=()
readonly REFUSE_EXIT=98
readonly STEP0_REF=1.0099719719146378          # declared reference: C, seed 2026092800, step 0 (JOINT-1; not overridable)
PEAK_TRAIN=${PEAK_TRAIN:-3} PEAK_INFER=${PEAK_INFER:-2} PEAK_LS=${PEAK_LS:-4} PEAK_DEV=${PEAK_DEV:-16}
PEAK_SMOKE=${PEAK_SMOKE:-1}
# Stage budgets (wall seconds; gates.py budget refuses a mandatory stage past the wall cap). LGLN staging:
# CU_PROJECTION.json x 1.25 (ADAPT-NDS-1's values); training: one run of 5,116 updates (two J epochs measured at 165 s
# each on G4, plus loading; ADAPT-NDS-1's 900 s per arm with its retention pass kept as the cap); the DEV pass:
# windows x (BASE_S + models x PER_S) x 1.25 with BASE_S / PER_S from CU_PROJECTION.json (driver).
T_LS=${T_LS:-2400} T_DEV_STAGE=${T_DEV_STAGE:-1800} T_TRAIN=${T_TRAIN:-900}
BASE_S=${BASE_S:-2.38} PER_S=${PER_S:-1.25}
STAGE_WORKERS=${STAGE_WORKERS:-8}

hg() { "$PY" -B "$HR1_CHAIN/gates.py" "$@"; }
passed() { PASSED[$1]=1; log "PASSED $1"; }
need() {
  local g
  for g in "$@"; do
    if [[ ${PASSED[$g]:-} != 1 ]]; then
      log "REFUSED: prerequisite '$g' has not passed; chain stops"
      exit "$REFUSE_EXIT"
    fi
  done
}
mgate() { local n=$1; shift; gate "$n" "$@"; passed "$n"; }
mstage() { local n=$1 p=$2; shift 2; stage "$n" "$p" "$@"; passed "$n"; }
json_field() { "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
smoke() { [[ $CHAIN_MODE == smoke ]]; }
ls_only() { if smoke; then printf -- '--only %s' "${SMOKE_LS_WINDOWS:?}"; fi; }

budget() {  # budget NAME SECONDS: exit 94 if the stage would pass the wall cap (every stage here is mandatory)
  local rc=0
  hg budget "$VM/state/BUDGET.json" "$1" "$2" || rc=$?
  if ((rc != 0)); then log "BUDGET: stage $1 does not fit the wall cap (rc=$rc); chain stops"; exit "$rc"; fi
}
pass_budget() {  # pass_budget MANIFEST N_MODELS: projected seconds of one shared LGLN pass (x 1.25)
  local n
  n=$("$PY" -B "$CHAIN_DIR/d1_gates.py" windows "$1")
  awk -v n="$n" -v m="$2" -v b="$BASE_S" -v p="$PER_S" 'BEGIN { printf "%d", n * (b + m * p) * 1.25 + 60 }'
}

# ---------------------------------------------------------------- mandatory gates, fixed order (match_gates MANDATORY)
do_lint() { mgate lint "$PY" -B "$CHAIN_DIR/d1_gates.py" lint "$CHAIN_FILE" "$CHAIN_MODE"; }
do_env() { need lint; mgate env "$PY" -B "$HR1_CHAIN/gates.py" env "$FREEZE_TXT"; }
do_pins() {
  if smoke; then need lint; else need env; fi
  mgate pins "$PY" -B "$HR1_CHAIN/gates.py" pins "$PINS" --require "$PINS_KIND"
}
do_roles() {  # every planned tile / window inside its role's terrain before any request (nothing fetched on failure)
  need pins
  if smoke; then
    mgate roles_smoke "$PY" -B "$NDS_DIR/nds_reader.py" check-roles confirm "${SMOKE_PLAN:?}" \
      --manifest main="${SMOKE_MANIFEST:?}"
  else
    mgate roles_dev "$PY" -B "$FRESH_DIR/lgln_reader.py" check-roles "$DEV_PLAN" "$DEV_TILE_INDEX" "$DEV_TILE_LEDGER" \
      --manifest main="$DEV_MANIFEST" --manifest enclosure="$DEV_ENC_MANIFEST"
  fi
  # shellcheck disable=SC2046
  mgate roles_ls "$PY" -B "$NDS_DIR/nds_reader.py" check-roles fit "$LS_PLAN" $(ls_only)
  passed roles
}
do_stage_ls() {  # Lower Saxony FIT windows from DGM1 tiles fetched on the VM (no local tile store); masks from the table
  need roles
  budget stage_ls "$T_LS"
  # shellcheck disable=SC2046
  mstage stage_ls "$PEAK_LS" "$PY" -B "$NDS_DIR/nds_reader.py" stage fit "$LS_PLAN" "$LS_MASKS" "$VM/ls_windows" \
    "$VM/LS_FETCH_LOG.jsonl" --workers "$STAGE_WORKERS" $(ls_only)
  if smoke; then
    mstage masks_ls 1 "$PY" -B "$NDS_DIR/smoke_local.py" masks "$LS_FIT" "$VM/ls_masks" "${SMOKE_LS_WINDOWS:?}"
  else
    mstage masks_ls 1 "$PY" -B "$HR1_CHAIN/masks.py" build "$LS_FIT" "$VM/ls_masks"
  fi
  # shellcheck disable=SC2046
  mgate staging_ls "$PY" -B "$NDS_DIR/nds_reader.py" verify fit "$LS_PLAN" "$VM/ls_windows" "$VM/LS_FETCH_LOG.jsonl" \
    "$RESULTS/LS_STAGING_RECEIPT.json" $(ls_only)
  cp "$VM/LS_FETCH_LOG.jsonl" "$RESULTS/"
  cp "$VM/ls_windows/RENDER_LOG.json" "$RESULTS/LS_RENDER_LOG.json"
}
do_stage_dev() {  # FRESH-1's Verden DEV windows (lgln_reader.py stage + verify, unchanged)
  need staging_ls
  budget stage_dev "$T_DEV_STAGE"
  mstage stage_dev "$PEAK_DEV" "$PY" -B "$FRESH_DIR/lgln_reader.py" stage "$DEV_PLAN" "$DEV_TILE_INDEX" "$DEV_TILE_LEDGER" \
    "$DEV_MASKS" "$VM/dev_windows" "$VM/DEV_FETCH_LOG.jsonl" --manifest main="$DEV_MANIFEST" \
    --manifest enclosure="$DEV_ENC_MANIFEST" --workers "$STAGE_WORKERS"
  mgate staging_dev "$PY" -B "$FRESH_DIR/lgln_reader.py" verify "$DEV_PLAN" "$DEV_TILE_INDEX" "$DEV_TILE_LEDGER" \
    "$VM/dev_windows" "$VM/DEV_FETCH_LOG.jsonl" "$RESULTS/DEV_STAGING_RECEIPT.json" --manifest main="$DEV_MANIFEST" \
    --manifest enclosure="$DEV_ENC_MANIFEST"
  cp "$VM/DEV_FETCH_LOG.jsonl" "$RESULTS/"
  cp "$VM/dev_windows/RENDER_LOG.json" "$RESULTS/DEV_RENDER_LOG.json"
}
do_stage_smoke() {  # smoke only: the inference path on a two-tile smoke manifest of FIT-district tiles (ADAPT-NDS-1's
  # CONFIRM-path smoke inputs, nds_reader confirm kind; no DEV, CONFIRM or other unopened terrain)
  need staging_ls
  mstage stage_smoke "$PEAK_SMOKE" "$PY" -B "$NDS_DIR/nds_reader.py" stage confirm "${SMOKE_PLAN:?}" "${SMOKE_MASKS:?}" \
    "$VM/s_windows" "$VM/SMOKE_FETCH_LOG.jsonl" --manifest main="${SMOKE_MANIFEST:?}" --workers "$STAGE_WORKERS"
  mgate staging_smoke "$PY" -B "$NDS_DIR/nds_reader.py" verify confirm "$SMOKE_PLAN" "$VM/s_windows" \
    "$VM/SMOKE_FETCH_LOG.jsonl" "$RESULTS/SMOKE_STAGING_RECEIPT.json" --manifest main="$SMOKE_MANIFEST"
  cp "$VM/SMOKE_FETCH_LOG.jsonl" "$RESULTS/"
}
do_void() {  # descriptive void shares; background rectangles must be wholly valid
  if smoke; then
    need staging_smoke
    mgate void_smoke "$PY" -B "$NDS_DIR/nds_reader.py" void "$SMOKE_MANIFEST" "$VM/s_windows" "$RESULTS/SMOKE_VOID.json"
  else
    need staging_dev
    mgate void_dev "$PY" -B "$FRESH_DIR/lgln_reader.py" void "$DEV_MANIFEST" "$VM/dev_windows" "$RESULTS/DEV_VOID.json"
  fi
  passed void
}
do_step0() {  # the environment's step-0 reproduction (JOINT-1/HR1 gate on its declared English fixture; no training data)
  need void
  mstage step0_a 1 "$PY" -B "$D1_DIR/d1_train.py" step0 "$RESULTS/STEP0_A.json"
  mstage step0_b 1 "$PY" -B "$D1_DIR/d1_train.py" step0 "$RESULTS/STEP0_B.json"
  step0_check "$STEP0_REF" "$(json_field "$RESULTS/STEP0_A.json" repr || true)" \
    "$(json_field "$RESULTS/STEP0_B.json" repr || true)"
  passed step0
}
do_parity() {
  need step0
  mgate parity_ao "$PY" -B "$HR1_CHAIN/parity.py" "$EQ1_BLOCK" AO "$RESULTS/parity"
  mgate parity_cont "$PY" -B "$HR1_CHAIN/parity.py" "$EQ1_BLOCK" CONT "$RESULTS/parity"
  passed parity
}
tables() { printf -- '--ls-table %s --ls-win %s --ls-mask %s' "$LS_FIT" "$VM/ls_windows" "$VM/ls_masks"; }
do_plan_check() {  # every seed's all-local plan on the staged table (CPU, no forward); digests checked after export
  local s
  need parity
  for s in $SEEDS; do
    # shellcheck disable=SC2046
    mgate "plan_check_$s" "$PY" -B "$D1_DIR/d1_train.py" plan-check "$s" "$RESULTS/PLAN_CHECK_$s.json" $(tables)
  done
  passed plan_check
}

# ---------------------------------------------------------------- training (one arm, seeds in order), final weights only
model_arg() {  # NAME=PATH@SHA256 of a finished run; a missing TRAIN_RESULT.json stops the chain
  local sha
  sha=$(json_field "$VM/runs/$1/TRAIN_RESULT.json" final_sha256) || { log "no TRAIN_RESULT.json for $1; chain stops"; exit 1; }
  printf -- '%s=%s@%s' "$1" "$VM/runs/$1/final.pt" "$sha"
}
finish_trained() {  # run records into results and the keep sink (final.pt stays on the VM for the verified download)
  local name=$1
  mkdir -p "$RESULTS/runs/$name"
  cp "$VM/runs/$name/TRAIN_RESULT.json" "$VM/runs/$name/epochs.jsonl" "$VM/runs/$name/curves.jsonl" \
    "$RESULTS/runs/$name/"
  mgate "sink_$name" "$PY" -B "$HR1_CHAIN/gates.py" sink "$VM/runs/$name" keep
  cp "$VM/runs/$name/WEIGHTS_SINK.json" "$RESULTS/runs/$name/"
}
run_names() { printf '%s ' ${INFER_NAMES:?}; }                 # the runs of the DEV pass (probe excluded)
do_train() {
  local name
  need plan_check
  for name in ${RUN_NAMES:?}; do
    budget "train_$name" "$T_TRAIN"
    # shellcheck disable=SC2046
    mstage "train_$name" "$PEAK_TRAIN" "$PY" -B "$D1_DIR/d1_train.py" train "$name" $(tables) \
      --run-root "$VM/runs"
    finish_trained "$name"
    log "WEIGHTS_READY $name"
  done
  passed train
}

# ---------------------------------------------------------------- inference (HR3 infer_irl.py, unchanged)
sidecars() {  # FILE.sha256 beside each file, written before PROPOSALS_READY (the driver's harvest verifies against it)
  local f
  for f in "$@"; do sha256sum "$f" | cut -d' ' -f1 >"$f.sha256.part" && mv -f "$f.sha256.part" "$f.sha256"; done
}
lgln_pass() {  # lgln_pass TAG MANIFEST WIN_DIR NAMES SPECS...: one shared pass, assembled, consume-checked, ready
  local tag=$1 manifest=$2 win=$3 names=$4 n specs=()
  shift 4
  for n in "$@"; do specs+=(--model "$n"); done
  mstage "infer_$tag" "$PEAK_INFER" "$PY" -B "$HR3_CHAIN/infer_irl.py" run "$manifest" "$win" "$VM/${tag}_out" "${specs[@]}"
  mkdir -p "$RESULTS/$tag"
  for n in $names; do
    mgate "assemble_${tag}_$n" "$PY" -B "$HR3_CHAIN/infer_irl.py" assemble "$manifest" "$VM/${tag}_out" "$n" "$RESULTS/$tag"
    mgate "consume_${tag}_$n" "$PY" -B "$FRESH_DIR/fresh_gates.py" consume "$manifest" \
      "$RESULTS/$tag/PROPOSALS_${n}_PRIVATE.json" "$RESULTS/$tag/CONSUME_${n}.json"
    sidecars "$RESULTS/$tag/PROPOSALS_${n}_PRIVATE.json" "$RESULTS/$tag/CONSUME_${n}.json"
    log "PROPOSALS_READY $tag/$n"
  done
}
trained_specs() {  # a run without TRAIN_RESULT.json fails the caller's assignment (set -e stops the chain)
  local n s
  for n in $(run_names); do s=$(model_arg "$n") || return 1; printf '%s ' "$s"; done
}
do_dev() {  # DEV (Verden, FRESH-1's frozen label-free inference manifest): the three final weights in one shared pass
  local specs
  need train
  # shellcheck disable=SC2206
  specs=($(trained_specs))
  budget dev "$(pass_budget "$DEV_MANIFEST" "${#specs[@]}")"
  lgln_pass dev "$DEV_MANIFEST" "$VM/dev_windows" "$(run_names)" "${specs[@]}"
  passed dev
}
do_complete() {  # every expected proposals file of the session
  need dev
  # shellcheck disable=SC2046
  mgate complete_dev "$PY" -B "$HR1_CHAIN/gates.py" complete "$RESULTS/dev" $(run_names)
  : >"$RESULTS/session_${SESSION_INDEX:?}.COMPLETE"
}
