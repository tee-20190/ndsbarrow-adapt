# GOETTINGEN-TRANSFER-1: does D1's advantage over public ADAF barrow replicate on a second fresh district?

**Pre-registration** (Opus, 2026-10-20). It is written and committed before any Göttingen register record detail, tile,
pixel or detector output has been read. Known beforehand are only the frozen district-level counts in
`work/adapt_nds_2026-10-05/CONFIRM_SELECTION.json`: rank 3, 683 eligible Grabhügel under rules 1-4, 240 groups
before NODATA, floor OK. The packet is frozen by `FREEZE.json` (to be written) before the VM job is launched.

**Capability line:** generalisation. Does a regional public-register adaptation recipe beat the public specialist
across districts and landscapes of its region?

## 1. Question and status

OLDENBURG-TRANSFER-1 found that the ten frozen D1 runs (public ADAF barrow + recipe F1, trained only on the Lower
Saxony FIT districts Uelzen, Harburg and Heidekreis) beat unchanged ADAF barrow on Oldenburg. The figures were
+0.0386, 10/10 runs positive, joint 95 % lower bound +0.0137 (`work/oldenburg_transfer_2026-10-07/RESULT.md`).
ENG-ADAPT-1 (A016) showed that this gain needs local register labels.

**This packet asks whether the gain holds on Landkreis Göttingen.** Göttingen lies about 90 km or more from every
FIT district, Verden and Oldenburg. Its terrain is the southern Lower Saxony uplands (Leine valley, wooded hills,
Harz foothills), unlike the northern lowlands where D1 was trained, selected and first confirmed.

**Why Göttingen.** It is the next district in ADAPT-NDS-1's frozen CONFIRM order (`CONFIRM_SELECTION.json`):
- rank 1 Oldenburg is used;
- rank 2 is D4R's reserved confirmation district [name redacted] and stays sealed;
- ranks 4 and below fail the 200-group floor before NODATA.

**Status of the terrain.** Göttingen has never been inferred by any project detector. It was named only as a
runner-up in D4R's `CONFIRM_DISTRICT.json` (counts and boundaries). It did not select D1's recipe, its checkpoints or
this rule.

**Cost of the choice.** Göttingen is the last floor-meeting confirmation district in the
`D1_PLAN_AMENDMENT.md` budget. Using it to replicate D1 itself ends further Verden increments with a fresh
confirmation ("when the districts run out, stop climbing on Verden"). This is accepted: a second-district replication
of the base result is worth more than one more increment check, and the Verden increment line has not produced a
confirmed gain since D1.

## 2. Eligible data (ADAPT-NDS-1's CONFIRM design, unchanged except the district name)

1. **Register.** `fetch_register.py`'s district rule, applied unchanged to "Göttingen": every Denkmalatlas record
   whose `lage.landkreis` is Göttingen, plus the 1 km edge supplement. It also uses the existing boundaries snapshot
   `SNAPSHOT_boundaries_20261005T005530Z`. The new snapshot is written to
   `outputs/goettingen_transfer_2026-10-20_PRIVATE/register/`.
2. **Labels, cores, background order and tile plan.** Built by `plan_confirm.py`'s functions with their frozen
   constants: FRESH-1 rules 1-7; positive cores with jitter `default_rng(20261004)`; background order
   `random.Random(20261003)`; ≥ 300 km² projected; spare factor 1.25; halo 512 m; LGLN DGM1 tiles.
   - No pixel is read at this step.
   - The plan is locked (sha256) before the VM job.
3. **Validity, NODATA, manifests and floors.** Built by `confirm_build.py`'s functions, unchanged, run on the VM:
   - V1 ≥ 0.95, V2 ≥ 0.90; the HR3 ownership partition; 10 m boxes; background squares in the locked order until
     300 km² are scored.
   - **Floors:** ≥ 200 groups and ≥ 150 km² background. **If a floor fails, the packet stops with
     NOT_EVALUATED.** No other district is substituted: none remains.
   - The build stage reads tile validity only. No elevation is stored and no detector runs before the manifests and
     floors are written.
4. **Legitimacy.** These checks follow the Verden logic of `work/oldenburg_transfer_2026-10-07/legit.py`, adapted:
   - FIT windows ∩ Göttingen district = ∅, with the minimum distance reported;
   - no register record is shared between FIT members and Göttingen entities;
   - no 1 km DGM1 square is shared between FIT pins and the Göttingen staging plan.

   **A legitimacy failure stops the packet.**

## 3. Models and inference (one pass, one VM)

- **Arms.** The ten D1 runs, D1F1_BARROW_L seeds 2026092805-14, with the exact hashes in
  `work/oldenburg_transfer_2026-10-07/INFERENCE_PLAN.md`. Local copies are in `work/release_2026-10/weights/batch_01/`,
  each sha256-checked before upload.
- **Comparator.** Unchanged public ADAF barrow `51b679fa…`.
- **Same pass for all eleven.** The models run with the same `infer_irl.py` (`f3855551…`), `nds_reader.py`,
  `fresh_gates.py`, `gates.py` and `parity.py` as OLDENBURG-TRANSFER-1. They use the same Göttingen inference manifest
  and the same staged DGM1 bytes (sha256 logged per file).
- **No earlier ADAF comparator.** No ADAF barrow Göttingen proposals exist, so this pass's ADAF barrow proposals are
  the primary comparator.
- **Technical failures.** A failure before scoring is rerun unchanged, and the erratum is recorded. A missing D1 run
  is reported, not replaced.

## 4. Primary comparison (scored once, after the VM job; CPU)

- **Scorer.** The Stage 1 scorer, imported unchanged: `work/stage1_d1_v0_2026-10-07/score_v0.py` (`v0`,
  `raw_pool`, `match_count`).
  - Inputs: the frozen Göttingen individual manifest. Every in-core proposal is charged.
  - Budget: K = floor(b × scored km²), with b = 0.5, 1 and 2.
  - Decoders: RAW primary, NMS60 secondary.
- **Gate (b = 0.5, RAW, unrounded, float guard 1e-12).** It passes when both hold:
  - the mean over the ten runs of (run recall − ADAF barrow recall) is **≥ +0.030**;
  - **≥ 8 of 10** differences are **> 0**.
- **Clear transfer improvement.** The gate passes **and** the joint 95 % lower bound is **> 0**.
  - The bound comes from the frozen v0 bootstrap: engine `boot_v0.py`; 2 km grid at (0, 0); 5,000 replicates.
  - Training draws come from stream `train_d1`, shared with Verden and Oldenburg because the checkpoints are
    identical.
  - The geography stream is `np.random.default_rng(np.random.SeedSequence(20261020))`. It is new, because
    `boot_v0.STREAM_NAMES` is frozen and has no Göttingen entry.
- **Reading.** The precedence table of OLDENBURG-TRANSFER-1 applies with "Göttingen" substituted.
- **Claim if CLEAR.** "D1 beats public ADAF barrow on two training-disjoint districts with a lower bound above
  zero." That is a replicated external-comparator gain within Lower Saxony.
- **Claim if not.** The Oldenburg gain does not carry to the southern uplands under this rule. The scale-up advice
  then narrows to adapting per landscape, not per state.

## 5. Secondary (descriptive, no gate)

- Every run, b = 1 and 2, NMS60, supply, and C_s / G_ranking / G_pool for each run and for ADAF barrow.
- Geography-only and training-only intervals; 4 km and offset grids.
- The deployable checkpoint's own result (run 05, `d9ece1fb…`).
- **Three-district summary** (Verden, Oldenburg, Göttingen): the equal-weight mean of the D1 − ADAF barrow
  differences, with the shared training draw and per-region geography.
- An evaluator v0.1 sensitivity where the scorer supports it.
- None of these changes the verdict.

## 6. Resources and safeguards

- **CU and VM.** One Colab G4 VM. Ceiling **70 CU**; whole-job cap 9 h. Projection: build plus staging about 1 h,
  inference of 11 models about 3 h, so about 40 CU.
- **Gates at launch.** `verify/v3/archive_gate_v2.py` and `verify/v3/storage_guard.py` on the VM against its own
  disk. Local work is small CPU and network only; nothing bulky is written to C:.
- **Licences and privacy.**
  - LGLN DGM1 and boundaries: CC BY 4.0. Register: CC BY-SA 4.0. ADAF and D1 weights: CC BY-SA 4.0.
  - Record-level data, coordinates and proposals stay in `outputs/goettingen_transfer_2026-10-20_PRIVATE/`.
  - Committed files carry aggregates, sha256 and utc only.
- **Colab etiquette.** The VM is stopped only after the result tar is sha-verified locally. Act only on VMs this
  session starts; at most three run concurrently across the project.
- **Verification.**
  - An independent recount, written by a Haiku 5.5 builder without reading the scorer, must agree on every point
    field.
  - An Opus verifier re-derives the gate and lower bound.

## 7. Forecasts (scored at the result; conditional on floors and legitimacy PASS)

| Event | P |
|---|---:|
| A floor fails (NOT_EVALUATED) | 0.12 |
| Gate PASS | 0.40 |
| Clear transfer improvement | 0.30 |
| Mean D1 − ADAF barrow ≤ 0 | 0.20 |
| ADAF barrow recall at b = 0.5 RAW within ±0.03 of Oldenburg's 0.0991 | 0.45 |

Basis:
- The effect shrank from +6.25 points on Verden to +3.9 on Oldenburg. The upland terrain is further from the training
  landscape.
- The Oldenburg coordinator forecast of 0.50 for a gate PASS came true. A016 shows the gain is local, which cuts both
  ways for a same-state district with a different landscape.

## Addendum 1 (2026-10-20, before the freeze; no Göttingen pixel or detector output read)

These are build decisions found while writing the code. None depends on any outcome.

1. **No SCREEN parity in this pass.** OLDENBURG-TRANSFER-1's pass ran `parity.py SCREEN_117_54 AO/CONT`, an
   environment reproduction check. It needs the CONT and AO weights and SCREEN data on the VM. The primary contrast
   here is within one pass (same code, same environment, same bytes), so parity is dropped.
   - The pass keeps `gates.py env` (exact package freeze, GPU present, TF32 override 0) before any read.
   - It also keeps `gates.py complete` after inference (all eleven proposal sets present). PACKED is COMPLETE only
     when the completeness gate passes.
   - The three-district summary (section 5) compares across passes. It stays descriptive.
2. **Plan-lock key.** `GT_PLAN_LOCK.json` names the selection hash `selection_sha256`, while `plan_confirm.verify_lock`
   reads `confirm_selection_sha256`. `gt_build.bind_vm` substitutes a verify function that applies the same checks to
   either key name. The frozen module is not edited.
3. **Validity tolerance.** The frozen `confirm_build` rule applies: consolidation with at most 20 % of the planned
   tiles failed, after one re-read pass. Failed-tile counts are reported.
4. **Weight names.** The local files `D1-F1_BARROW_L_<seed>.pt` are shipped as `D1F1_BARROW_L_<seed>.pt` on the VM.
   Each one is sha256-checked against the INFERENCE_PLAN pins before and after transfer.
5. **Verification so far.**
   - Builders: four Haiku 5.5 agents, then one Haiku fixer.
   - Two independent Opus 5.5 verification passes.
   - Builder-claim precision in the first pass was 22/25 (88 %, below the 90 % bar). All defects found were fixed and
     re-verified before freezing.
