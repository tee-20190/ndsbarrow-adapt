# DK-REGSUP-1 stage A — FREEZE addendum 1 (2026-10-29, before any C_k decode placement, replay output or A − C score)

**Source.** Owner instruction in chat, about 03:50Z, recorded by the coordinator in
`decisions/2026-10-29_dk_regsup1_colab_no_cap_two_replays.md` (commit e930da2ec): "keep all GPU work on Colab, lift the
120 CU cap while we're above 1,000 CU, and complete the two control replays before accepting the result." It settles
Codex's STATUS_0343 points DRR-2 and DRR-3. The cross-session message never reached this session; it read the
committed record at about 05:12Z and applies it from then. This addendum amends FREEZE.md (478c1593a) §5, §8 and §10
only. Arms, seeds, folds, labels, scoring, the gate, power and the forecasts are unchanged.

## 1. Placement: every GPU step runs on Colab

- The RTX 2060 option of FREEZE §10 is withdrawn: training, decode and replays all run on Colab G4 machines.
- Local GPU use before this session read the instruction, kept only as descriptive receipts and never used in any
  proposal set:
  - the one-core D_101 parity decode (03:29Z, `local_decode/ltest`): same map sha256 and objects as CONFIRM-D stage A's
    local part, except the restored mosaic's `source` path;
  - a partial decode of A_F1_101–505 on 17 of fold 1's 54 cores (04:15–05:13Z, `local_decode/L1`), stopped at 05:13Z.
    Every A_F1 model is decoded again on Colab. These 85 local parts may be compared with the Colab parts descriptively.

## 2. Ceiling

- The 15 G4-hour / 120 CU limit of FREEZE §10 is lifted while the account balance stays ≥ 1,000 CU (3,999.09 CU at
  04:58:53Z). Nothing is launched that would take the balance below 1,000 CU.
- Machine limits are unchanged: at most 4 machines of this packet, 6 account-wide.
- FREEZE §8: INCOMPLETE no longer triggers on a CU or G4-hour ceiling. The step-0 rule and the failed-twice rule stand.

## 3. Control qualification (ADDENDUM_1 §2, binding; replaces FREEZE §5's optional fold-0 replay)

**3a. Zero-Danish plan identity: met.** `PLAN_CHECK.json` builds each historical seed's C plan through `rs1_train.py`'s
own code (`rs1_train.context` for the mixed loader, `rs1_train.schedules`, `match_train.plan_summary`). For all 9
historical seeds it equals, as a dict, the `plan_summary` in that D_k's own TRAIN_RESULT.json. Each epoch's draw digest
is step → [window uid, dy, dx, dihedral k] in step order, so equality means the same window IDs, crops, transforms and
order. `zero_danish_equals_c` (A's plan with the Danish steps returned to C's draws equals C's plan) is true in all 45
plans, locally (`PLAN_CHECK.json`) and on the F VM (`STAGE_F_CHECK.json`). Each trained A_k^f's TRAIN_RESULT carries
the same C-plan digest (`rs1_traincheck.py`). Nine historical seeds ≥ the two required.

**3b. Runtime replays.**
- C_101 and C_404 are retrained on the F VM (Colab G4, the same runtime as the six new C_k) in the same `ctrain` job,
  with the same command (`rs1_seeds.py train train k --arm D`, the same Lower Saxony inputs and flags) and the same
  code bytes as the six new C_k.
- `rs1_seeds.py`'s train-mode seed tuple is extended to (1010, 1111, 1212, 1313, 1414, 1515, 101, 404) before any C_k
  F1 run starts, so all eight C runs share one wrapper. `dt1_train.py` (unchanged, `f241d715…`) reads that tuple only
  for its membership gates and `SEEDS[0]` in smoke (still 1010). The six SSL runs used the pinned bytes; SSL mode keeps
  the six new seeds.
- Encoders: the historical encoders of D_101 (DT1) and D_404 (stage B), sha256 checked by `rs1_traincheck.py` against
  each D_k's TRAIN_RESULT.
- Each replay (decode names R_101, R_404) is decoded on all 171 DK-JW cores on Colab with the pinned decoder. It is
  scored by `rs1_replay.py`: all records, 2/km² + PROX, reach 15 m, through `rs1_score.inputs` and `rs1_score.hits`
  exactly as RESULT_A scores a control. The comparison is with the historical proposal file the gate uses for that
  seed: D_101 from CONFIRM-D stage A's decode, D_404 from stage B's VM decode.
- **PASS iff |Δ| ≤ 0.76 points for both seeds.** An identity stand-in (historical files as replays) gives Δ = 0.000
  and PASS (scratch smoke, 05:23Z).
- The pair was fixed by the coordinator before any replay output, one seed from each historical decode batch.

**3c. Fallback.** If either replay fails, no historical control is used. C_202, 303, 505, 606, 707, 808 and 909 are
retrained the same way and decoded on all 171 cores on Colab. C becomes the 15 fresh C_k (the two replays are C_101
and C_404). `rs1_score.c_file` is then pointed at the fresh files by a recorded change before any A − C score.

**3d. Order.** `REPLAY_CHECK.json` (the verdict) is committed before any A − C contrast is computed. `rs1_replay.py`
reads no A proposal set.

## 4. Acceptance

No A − C result is accepted, read as a gate result or used for the stage-B decision until item 3 is complete, including
its fallback if triggered.

## 5. Operational changes recorded with this addendum

| File | Change | sha256 |
|---|---|---|
| `rs1_seeds.py` | train-mode SEEDS += (101, 404) (was `6c3a3cfe…`) | `0802efcd878c2cdbe3c80cca1d69130847e4de6676867bbf88321a3ffe19b8a1` |
| `vm/rs1_job.py` | `ctrain` admits seeds 101, 404 (was `7befd004…` after the 01:45Z plan-batching edit) | `0ce63dd1dae0b914bea401b95ea239c2ff1390f7198fa4ddbc0a0dff0ccfd046` |
| `rs1_replay.py` | new: the replay scorer of item 3b | (in this commit) |

Coordinator prediction (decision record, before any replay output): both replays within 0.76 points, P ≈ 0.8.
This session's forecast, also before any replay output: P ≈ 0.8. FL2 found replays not bitwise equal; with the encoder
and draw order shared, the replay difference should be well inside the stage-B seed spread.
