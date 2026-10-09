# OLDENBURG-TRANSFER-1: inference plan for the one GPU step (frozen with PACKET.md)

Not launched. The coordinator launches. This file fixes what the GPU step must produce, and how.

## What runs

**One shared inference pass** over the frozen Oldenburg CONFIRM inference manifest. Nothing else uses the GPU.

- **Manifest:** `outputs/adapt_nds_2026-10-05/confirm/run/INFER_MANIFEST_PRIVATE.json`, sha256
  `8fce717eea156a9a00ec1bcdea4a936b4b89454c166b1bf4b878f702dc31f411`, 699 windows. Its tiles are identical to the
  individual scoring manifest `3531675c…`.
- **Models: eleven.**

| Model name (exact) | Weights | sha256 |
|---|---|---|
| D1F1_BARROW_L_2026092805 (**deployable checkpoint**) | final.pt | `d9ece1fb023d2daf72ab0e5f789b5b89a27af73a28307e6260e4ee2fb5ac501e` |
| D1F1_BARROW_L_2026092806 | final.pt | `f291694980fe24843e027b8ad846da08c4f407a0d75c1f3600ee0c5ef4f0bade` |
| D1F1_BARROW_L_2026092807 | final.pt | `d705204634fc35751371c37e7fea85600acdd12e2eae15ed22aaf90cf7d73511` |
| D1F1_BARROW_L_2026092808 | final.pt | `61ae1cb053a35a6bc427eefeaf7614d299dd8772e7c0d5eb99a47f35ae0762b5` |
| D1F1_BARROW_L_2026092809 | final.pt | `38d14080dc98962ea79f31c0fbbfedc96f9bfd121cc1ef82003fe0d3b610563d` |
| D1F1_BARROW_L_2026092810 | final.pt | `6b294645c62a074f864b6fe2ab8df4e9796e69b25a9abd6993093b53b78f21a5` |
| D1F1_BARROW_L_2026092811 | final.pt | `ed12f627ec5975fb7cfde6565f9af6eab7b30b7f89423f49da97c1001bf69229` |
| D1F1_BARROW_L_2026092812 | final.pt | `5437f281701203d9295e1f3b1da9a80261e2681cdecfe94b51ee4c7e54a815eb` |
| D1F1_BARROW_L_2026092813 | final.pt | `3a1c2643ef26a336ba8915780328f1af27bf32a52b24905cb3ccb3f19c18a044` |
| D1F1_BARROW_L_2026092814 | final.pt | `24db9082f55fe2f8cbd45e265b6164d73ec01823858c3ebd4630fe6b0b5f0858` |
| ADAF_barrow (same-pass parity only; never the primary comparator) | `work/campaign_2026-09-22/adaf_weights/barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar` | `51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216` |

The D1 hashes are the `trained` map of
`outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z/SESSION_VERIFIED.json` (status PASS) and the
registry.

**Code and environment (all byte-identical to the ADAPT-NDS-1 session that made the frozen ADAF barrow comparator):**

| File | sha256 |
|---|---|
| `work/hr3_ireland_2026-10-02/step2/chain/infer_irl.py` | `f3855551…` |
| `work/adapt_nds_2026-10-05/nds_reader.py` | `27359e11…` |
| `work/fresh_region_2026-10-03/step2/fresh_gates.py` | `f0365d8e…` |
| `work/holdout_region_2026-10/chain/gates.py` | `d04cdd0b…` |
| `work/holdout_region_2026-10/chain/parity.py` | `e9320255…` |
| `work/colab_assessment_2026-09-29/env/setup_env.sh` | `1d17df26…` |
| `work/colab_assessment_2026-09-29/env/freeze.txt` | `74512145…` |

- These hashes are pinned identically in the ADAPT-NDS-1 and D1 freezes (`work/*/freeze/PINS_FREEZE.json`); 95 of 95
  shared pins are equal.
- Both sessions recorded the same VM environment freeze, `9d23a2c7…`.
- **Input bytes are pinned.** The CONFIRM staging plan pins every DGM1 file's sha256 (1,011 files), so re-staging reads
  identical terrain or fails `verify`.
- **Inference determinism.** G4 inference is deterministic under the project environment (EQ-1: 1,217/1,217 proposals
  matched, |Δscore| ≤ 2e-6, maps bit-identical; `work/colab_assessment_2026-09-29/NOTES.md:24-33,84`).

## Before launch (local, coordinator)

1. **Archive and storage gates.**

   ```bash
   python -B verify/v3/archive_gate.py check
   ```

   ```bash
   MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- python3 -B \
     verify/v3/storage_guard.py check --path /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH/outputs --peak-gb 7.5
   ```

   - Declared local peak: **7.5 GB**. That is 3.20 GB of restored weights, plus 2.96 GB of Drive-for-desktop cache from
     the first read of the archive zip, plus ≤ 1 GB of bundle stream, plus ≤ 0.1 GB of results
     (`CU_PROJECTION.json` `local_storage`).
   - storage_guard needs ≥ 27.5 GB free.
   - The D1 driver's local rule adds 2 GB of page-file headroom, so it needs ≥ 29.5 GB.
   - Measured 2026-10-07 ~22:45Z: 27.78 GB free. storage_guard PASS (`STORAGE_CHECK.json`). **The D1-style local rule
     would refuse (−1.7 GB).** Free ≥ 2 GB first: clear the Drive-for-desktop cache, or archive verified completed
     artifacts.
2. **Restore the ten D1 weights** from archive bundle k1. Restore manifest:
   `outputs/archive_completed_2026-10-07k1/restore-manifest.json`. The Drive zip is `completed-2026-10-07k1.zip`, sha256
   `44639b2c…`.
   - Destination: `outputs/oldenburg_transfer_2026-10-07/weights/<name>/final.pt`. This packet is the single writer;
     the D1 directory is not touched.
   - Read G: only through PowerShell exact paths or the connector, never Git Bash.
   - Verify each file against the table above (sha256 and 319,970,247 bytes) before bundling.
3. **Build the driver.**
   - It is a checked copy of D1's `driver_d1.sh` / `vm_job_d1.sh` / `chain_lib_d1.sh`. Delete the training, LS-staging
     and plan-check stages.
   - Add ADAPT-NDS-1's CONFIRM stage functions (`chain_lib_adapt.sh` `do_stage_confirm`, `void_confirm`).
   - Bundle: code, the CONFIRM run directory (`STAGING_PLAN`, `INSIDE_MASKS`, `INFER_MANIFEST`), the ten restored
     weights and the ADAF barrow tar. Size ≈ 4.4 GB, streamed by `bundle_stream.py`.
   - `WEIGHTS_SINK` = none: no weight leaves the VM; the archive already holds them.
   - Smoke it on the production artifact route before the run. Use D1's two-tile FIT-district smoke manifest
     (`SMOKE_PLAN` / `SMOKE_MANIFEST`), not Oldenburg windows.
   - The driver's freeze must pin this packet's `FREEZE.json`.

## VM-side chain (exact stage order and commands)

Paths: `DB=/content/DB`, `PY=/content/venv/bin/python`, `VM=$DB/outputs/oldenburg_transfer_2026-10-07/vm`,
`RESULTS=$VM/results`, `C=$DB/outputs/adapt_nds_2026-10-05/confirm/run`,
`W=$DB/outputs/oldenburg_transfer_2026-10-07/weights`. Each stage runs inside storage_guard (VM: `--path /content
--peak-gb 60`) through HR1's frozen launcher `work/holdout_region_2026-10/step2/run_chain.sh`.

```bash
$PY -B $DB/work/holdout_region_2026-10/chain/gates.py env $DB/work/colab_assessment_2026-09-29/env/freeze.txt
$PY -B $DB/work/holdout_region_2026-10/chain/gates.py pins <run>/PINS_SESSION.json --require oldenburg_transfer_run
$PY -B $DB/work/adapt_nds_2026-10-05/nds_reader.py check-roles confirm $C/STAGING_PLAN_PRIVATE.json \
    --manifest main=$C/INFER_MANIFEST_PRIVATE.json
$PY -B $DB/work/adapt_nds_2026-10-05/nds_reader.py stage confirm $C/STAGING_PLAN_PRIVATE.json \
    $C/INSIDE_MASKS_PRIVATE.npz $VM/c_windows $VM/CONFIRM_FETCH_LOG.jsonl \
    --manifest main=$C/INFER_MANIFEST_PRIVATE.json --workers 8                        # VM peak 18 GB
$PY -B $DB/work/adapt_nds_2026-10-05/nds_reader.py verify confirm $C/STAGING_PLAN_PRIVATE.json $VM/c_windows \
    $VM/CONFIRM_FETCH_LOG.jsonl $RESULTS/CONFIRM_STAGING_RECEIPT.json --manifest main=$C/INFER_MANIFEST_PRIVATE.json
$PY -B $DB/work/adapt_nds_2026-10-05/nds_reader.py void $C/INFER_MANIFEST_PRIVATE.json $VM/c_windows \
    $RESULTS/CONFIRM_VOID.json
$PY -B $DB/work/holdout_region_2026-10/chain/parity.py SCREEN_117_54 AO $RESULTS/parity
$PY -B $DB/work/holdout_region_2026-10/chain/parity.py SCREEN_117_54 CONT $RESULTS/parity
$PY -B $DB/work/hr3_ireland_2026-10-02/step2/chain/infer_irl.py run $C/INFER_MANIFEST_PRIVATE.json $VM/c_windows \
    $VM/confirm_out \
    --model D1F1_BARROW_L_2026092805=$W/D1F1_BARROW_L_2026092805/final.pt@d9ece1fb023d2daf72ab0e5f789b5b89a27af73a28307e6260e4ee2fb5ac501e \
    --model D1F1_BARROW_L_2026092806=$W/D1F1_BARROW_L_2026092806/final.pt@f291694980fe24843e027b8ad846da08c4f407a0d75c1f3600ee0c5ef4f0bade \
    --model D1F1_BARROW_L_2026092807=$W/D1F1_BARROW_L_2026092807/final.pt@d705204634fc35751371c37e7fea85600acdd12e2eae15ed22aaf90cf7d73511 \
    --model D1F1_BARROW_L_2026092808=$W/D1F1_BARROW_L_2026092808/final.pt@61ae1cb053a35a6bc427eefeaf7614d299dd8772e7c0d5eb99a47f35ae0762b5 \
    --model D1F1_BARROW_L_2026092809=$W/D1F1_BARROW_L_2026092809/final.pt@38d14080dc98962ea79f31c0fbbfedc96f9bfd121cc1ef82003fe0d3b610563d \
    --model D1F1_BARROW_L_2026092810=$W/D1F1_BARROW_L_2026092810/final.pt@6b294645c62a074f864b6fe2ab8df4e9796e69b25a9abd6993093b53b78f21a5 \
    --model D1F1_BARROW_L_2026092811=$W/D1F1_BARROW_L_2026092811/final.pt@ed12f627ec5975fb7cfde6565f9af6eab7b30b7f89423f49da97c1001bf69229 \
    --model D1F1_BARROW_L_2026092812=$W/D1F1_BARROW_L_2026092812/final.pt@5437f281701203d9295e1f3b1da9a80261e2681cdecfe94b51ee4c7e54a815eb \
    --model D1F1_BARROW_L_2026092813=$W/D1F1_BARROW_L_2026092813/final.pt@3a1c2643ef26a336ba8915780328f1af27bf32a52b24905cb3ccb3f19c18a044 \
    --model D1F1_BARROW_L_2026092814=$W/D1F1_BARROW_L_2026092814/final.pt@24db9082f55fe2f8cbd45e265b6164d73ec01823858c3ebd4630fe6b0b5f0858 \
    --model ADAF_barrow=$DB/work/campaign_2026-09-22/adaf_weights/barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar@51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216
# for each NAME of the eleven:
$PY -B $DB/work/hr3_ireland_2026-10-02/step2/chain/infer_irl.py assemble $C/INFER_MANIFEST_PRIVATE.json \
    $VM/confirm_out NAME $RESULTS/confirm
$PY -B $DB/work/fresh_region_2026-10-03/step2/fresh_gates.py consume $C/INFER_MANIFEST_PRIVATE.json \
    $RESULTS/confirm/PROPOSALS_NAME_PRIVATE.json $RESULTS/confirm/CONSUME_NAME.json
sha256sum ... > PROPOSALS_NAME_PRIVATE.json.sha256 ; CONSUME_NAME.json.sha256        # sidecars, then PROPOSALS_READY
$PY -B $DB/work/holdout_region_2026-10/chain/gates.py complete $RESULTS/confirm D1F1_BARROW_L_2026092805 ... \
    D1F1_BARROW_L_2026092814 ADAF_barrow
```

Every command except the eleven `--model` arguments and the output directory appears verbatim in ADAPT-NDS-1 session
1's chain log (`outputs/adapt_nds_2026-10-05/chain_runs/run_s1_20261005T043612Z/v1/results/chain.log`).

## Expected outputs and verification

- **Local:** `outputs/oldenburg_transfer_2026-10-07/chain_runs/<run>/`, containing:
  - `v1/results/confirm/PROPOSALS_<name>_PRIVATE.json` for the eleven models, each with a `.sha256` sidecar and
    `CONSUME_<name>.json`;
  - `CONFIRM_FETCH_LOG.jsonl`, `CONFIRM_STAGING_RECEIPT.json`, `CONFIRM_VOID.json` and `parity/`;
  - `SESSION_VERIFIED.json`.
- **SESSION_VERIFIED.json** has status PASS, and a `proposals` map keyed `confirm/PROPOSALS_<name>_PRIVATE.json` with
  sha256. It is written by the driver's verify only after every file has a verified local copy.
- **VM stop:** only after the results are verified locally. No weights are downloaded.
- **Then, once:**

  ```bash
  MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/15516/DISCOVERYBREAKTHROUGH -- \
    /home/tee-/adaf-h2h-venv/bin/python -B work/oldenburg_transfer_2026-10-07/score_oldenburg.py \
    outputs/oldenburg_transfer_2026-10-07/chain_runs/<run>
  ```

  - `score_oldenburg.py` refuses unless:
    - `LEGITIMACY.json` is PASS;
    - every frozen file matches `FREEZE.json`;
    - the session is PASS;
    - each proposals file matches its sidecar and pin, its checkpoint hash and the inference manifest.
  - It writes `RESULT.json`. `RESULT.md` is written from it with the frozen reading table.
- **Technical failure versus scientific result.** A technical failure (staging, verify, consume, session) is not a
  scientific result. Fix the route and rerun the same inference. The scorer has not run, so this is not a re-score.

## Resources (`CU_PROJECTION.json`, from measured actuals)

The projection rests on measured actuals:

- **D1 Verden inference:** 10,443 s for 10 models × 717 windows, giving 1.2185 s per model-window over the 2.38 s
  base.
- **D1 session 3:** 37.09 CU over 15,223 s, an effective 8.77 CU/h.
- **ADAPT-NDS-1 Oldenburg:** staging 228 s. Its 8-model Oldenburg pass took 8,594 s; the model predicts 8,477 s
  (−1.4 %).

| Item | Seconds |
|---|---:|
| Set-up (allocation → chain; D1 measured 257 s + 3.2 GB of weights at 49.2 s/GB) | 414 |
| Gates (env, pins, roles, parity) | 128 |
| CONFIRM staging + verify + void | 268 |
| Inference, 11 models × 699 windows | 11,033 |
| Assemble/consume, package, download, stop | 180 |
| **Wall** | **12,023 (3.34 h)** |
| **CU** | **29.3** (1.25× cap 36.6; **hard ceiling 40**) |

- Dropping the ADAF parity model saves 2.1 CU. It stays in: it checks the comparator route at about 7 % of the cost.
- **VM:** staging peak 18 GB; guard 60 GB on ~180 GB free.

## Sharing a VM session with the Stage 2 controls (coordinator decides)

Stage 2's projection is 40.8 CU for session 1 (`work/stage2_d1_controls_2026-10-07/PACKET.md`, read only).

| Option | For | Against |
|---|---|---|
| Append this pass to Stage 2's session 1 | Saves one allocation, set-up and environment gates: about 470 s, ≈ 1.1 CU. Stage 2 already bundles the ADAF barrow tar. | Stage 2 has a **hard 60 CU ceiling**: 40.8 + 29 ≈ 70 CU would breach it unless the ledgers are split. The parallel agent's Stage 2 freeze would need an amendment (new stages, new hash) after or during its freeze. One ~8 h session raises wall-cap and idle-reclaim exposure. A failure in either chain stops both. |
| **Separate session after Stage 2 (recommended)** | Independent freeze, ledger and ceiling. Stage 2's design is untouched. Sequential-first rule kept (one VM at a time). | ≈ 1.1 CU more, and one more allocation. |

Order relative to Stage 2 does not matter scientifically: both designs are frozen before either outcome exists (R4-3).
