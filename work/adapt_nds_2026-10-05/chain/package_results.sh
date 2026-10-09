#!/bin/bash
# ADAPT-NDS-1 result packaging (the VM's last step; the local smoke runs the same file): results/ as a tar in 48 MiB
# parts and every run's final.pt in 48 MiB parts, one INDEX.sha256 over all parts, RES_TAR.sha256 for the reassembled
# tar (JOINT-2 vm_job_joint2.sh's packaging block, unchanged in content; partial results included).
#   bash package_results.sh VM_DIR PARTS_OUT [keep]
set -uo pipefail
VM=${1:?VM dir} PARTS_OUT=${2:?parts dir} SINK=${3:-keep}
rm -rf "$PARTS_OUT" && mkdir -p "$PARTS_OUT"
TMP_TAR=$(dirname "$PARTS_OUT")/res.tar
tar -cf "$TMP_TAR" -C "$VM" results
split -b 48M -d -a 3 "$TMP_TAR" "$PARTS_OUT/res.tar.part_"
( cd "$PARTS_OUT" && sha256sum res.tar.part_* > INDEX.sha256 )
sha256sum "$TMP_TAR" | cut -d' ' -f1 > "$PARTS_OUT/RES_TAR.sha256"
rm -f "$TMP_TAR"
if [ "$SINK" = keep ]; then         # every final.pt for the driver's verified download
  for f in "$VM"/runs/*/final.pt; do
    [ -f "$f" ] || continue
    split -b 48M -d -a 3 "$f" "$PARTS_OUT/w_$(basename "$(dirname "$f")").pt.part_"
  done
  if compgen -G "$PARTS_OUT/w_*" >/dev/null; then ( cd "$PARTS_OUT" && sha256sum w_* >> INDEX.sha256 ); fi
fi
echo "packaged $(wc -l < "$PARTS_OUT/INDEX.sha256") parts"
