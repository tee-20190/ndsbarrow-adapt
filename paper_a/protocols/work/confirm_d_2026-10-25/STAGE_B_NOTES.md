# CONFIRM-D stage B launch notes (DRAFT_FREEZE §6 B, §9; the "stage-B launch notes" pin)

Stage B reuses DT1's training code **unchanged**. Only the seed list and the output directory differ.

| File (work/dense_terrain1_2026-10-25/) | sha256 prefix |
|---|---|
| `dt1_ssl.py` | `c801d2ffa923e221` |
| `dt1_train.py` | `f241d7151d6a2e61` |
| `dt1_model.py` | `9e3fdbcee88b762c` |
| `dt1_channels.py` | `0783d1e97f2111c8` |
| `dt1_decode.py` | `af88392ab734fc49` |
| VM driver `train_vm/dt1_tv.sh`, `dt1_tv_bundle.py`, `dt1_tv_job.py` | as committed in DT1 |

**Runs.**
- New D_k: four-channel input, dense SSL (4,096 updates) → F1 BARROW_L. Same recipe as DT1 arm D.
- New A_k: legacy input, ADAF barrow start → F1 BARROW_L. Same recipe as DT1 arm A (the D1 recipe).
- k ∈ {404, 505, 606, 707, 808, 909}. For k ∈ {101, 202, 303}, the registered DT1 checkpoints are reused: D_k from
  DETECTOR_REGISTRY DT1D-F1_BARROW_L_k; A_k from the DT1 A finals (archived; restore via fastarch receipts in
  `work/dt1_archive_2026-10-25/RECLAIM_LOG.jsonl`).
- Training data: DT1's FIT/halo bundle (Lower Saxony TRAIN only), exactly as DT1. No Danish terrain is ever a training
  input.

**Gates (DRAFT_FREEZE §10.4).**
- D1's step-0 rule on each training VM.
- SSL point and view manifests hashed per seed.
- The F1 plan digest `match_train.schedule(k, …)` must be identical for D_k and A_k.
- A pair whose digests differ is rerun. It is never scored.

**Order and budget.**
- Stage A completes and is scored first. Stage B outputs are scored only after all K pairs exist.
- If a mid-run forecast exceeds the 48 VM-h ceiling, K drops to 6 (k ≤ 606) before any stage-B output is scored.

**Registry.**
- Each new checkpoint is registered in DETECTOR_REGISTRY as `evaluated_candidate_not_adopted` with its sha256, parent,
  seed and this packet as its evidence. Its writer does this.
