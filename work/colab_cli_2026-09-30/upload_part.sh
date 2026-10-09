#!/bin/bash
# split --filter target: stage stdin to a temp file, upload it as /content/cli/parts/$FILE (one retry), log wall time.
set -uo pipefail
tmp=$(mktemp)
cat >"$tmp"
rc=1
for attempt in 1 2; do
  t0=$(date +%s.%N)
  timeout -k 10 900 "$COLAB_BIN" --config "$CLI_STATE" upload -s "$SESSION" "$tmp" "/content/cli/parts/$FILE" >/dev/null 2>>"$RUN_DIR/driver.log"
  rc=$?
  t1=$(date +%s.%N)
  echo "== $(date -u +%FT%TZ) part $FILE attempt $attempt rc=$rc bytes=$(stat -c %s "$tmp") secs=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')" >>"$RUN_DIR/driver.log"
  [ $rc -eq 0 ] && break
done
rm -f "$tmp"
exit $rc
