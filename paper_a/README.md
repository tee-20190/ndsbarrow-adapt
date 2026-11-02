# Paper A: code, frozen protocols and aggregate results

Release material for the manuscript:

> Thomas Shanahan. *Cross-border recovery of registered burial mounds by an adapted public LiDAR detector at fixed
> review workloads: a sealed Dutch test, a confirmed Danish system, and seed-level recipe repeatability.* Manuscript
> in preparation, 2026.

The rest of this repository belongs to the earlier preprint (see the top-level `README.md`). This folder is separate
and does not change any earlier file.

**This folder is a pinned record of the implementation, not a standalone package.** The scripts import further
project modules (listed below) and the ADAF code, and they read private inputs by fixed repository paths. They show
exactly what was run. Running them needs those modules, the public data fetched at source and the weights.

## Contents

| Path | Content |
|---|---|
| `code/` | Code files of the detection systems, training recipes, scorers and independent recounts, at their original repository-relative paths (`code/work/<experiment>/...`, `code/tools/clean_start/...`), so imports can be traced. Each file is byte-identical to the file that produced the result, except the four with a recorded redaction (`REDACTIONS.md`). |
| `protocols/` | Frozen decision files (freeze, specification and addenda), at their original paths. Five carry recorded redactions. |
| `results/` | Committed aggregate result files and independent-recount outputs, at their original paths. They hold counts, recalls, contrasts, intervals and SHA-256 values only: no record identifiers, coordinates, core identifiers or per-record hits. |
| `weights/` | `WEIGHTS_MANIFEST.csv` (58 checkpoints with role, seed, parent, bytes, SHA-256, MD5, training data and licence), `SYSTEM.md` (exact system configuration), `LICENCE_AND_ATTRIBUTION.md` and `WITHHELD.md` from the weights package. The checkpoints themselves are not in this repository. |
| `REDACTIONS.md` | Every changed file: original SHA-256, released SHA-256 and the exact text change. |
| `MANIFEST.sha256` | SHA-256 of every other file in this folder. Check with `sha256sum -c MANIFEST.sha256` from inside `paper_a/`. |

The SHA-256 pins in `weights/SYSTEM.md`, in the weights deposit's `CODE_PINS.json` and in the freeze files refer to
the original bytes. For unredacted files they verify on the released bytes; for the nine redacted files they verify
against the "original" column of `REDACTIONS.md`.

## Which file supports which result

| Manuscript item | Protocol (`protocols/work/...`) | Code (`code/...`) | Results (`results/work/...`) |
|---|---|---|---|
| §5.1 Göttingen, D1 vs ADAF barrow (+5.59 points) | `goettingen_transfer_2026-10-20/SPEC.md`, `FREEZE.json`, `FREEZE_LOCK.json` | not included: that packet's inference and scoring code (`gt_*.py`; the Stage 1 scorer `work/stage1_d1_v0_2026-10-07/score_v0.py` is in this repository from the earlier release) | `goettingen_transfer_2026-10-20/RESULT.json` |
| §5.1 English barrow corpus (EB) vs D1 on Oldenburg | `eng_adapt1_2026-10-19/SPEC.md` | `work/eng_adapt1_2026-10-19/ea_train.py` (training only; its scorer is not included) | `eng_adapt1_2026-10-19/RESULT.json` |
| §5.2 Table 2, Dutch sealed TEST (D1, CONT, ADAF barrow, ADAF AO) | `open_transfer_test_2026-10-22/FREEZE_TEST.md` (with `open_transfer_2026-10-22/FREEZE.md`) | `work/open_transfer_test_2026-10-22/pipeline/ot_infer.py` (inference and decoder), `work/open_transfer_test_2026-10-22/ot_score.py`, `work/scot_transfer_2026-10-19/scot_score.py` (matching, budgets, bootstrap, verdict), `work/open_transfer_2026-10-22/ot_labels.py` (record cohort and province roles), `work/open_transfer_test_2026-10-22/ot_recount.py` (independent recount) | `open_transfer_test_2026-10-22/RESULT_TEST.json`, `REGIONS_SUMMARY_TEST.json` |
| §5.2 Dutch EVAL replication (+11.8 points) | `open_transfer_2026-10-22/FREEZE.md` | `work/open_transfer_2026-10-22/ot_score.py`, `ot_labels.py`, `ot_recount.py`, `scot_score.py` (the EVAL copy of `ot_infer.py` is not included; it differs from the TEST copy in paths only, `FREEZE_TEST.md` §5) | `open_transfer_2026-10-22/RESULT.json`, `REGIONS_SUMMARY.json` |
| §5.3 Table 3, West Jutland: A1, A2, guards, sensitivities | `confirm_d_2026-10-25/FREEZE.md`, `ADDENDUM_SEQ.md` | decoders `work/dense_terrain1_2026-10-25/dt1_decode.py`, `dt1_channels.py`, `dt1_model.py` (four-channel) and `work/adaf_head_to_head_2026-09-19/adapter.py`, `work/campaign_2026-09-22/adaf_configs.py`, `work/tta_ensemble_2026-09-24/tta_run.py` (legacy); `work/merge1_2026-10-25/merge1.py` (MERGE-1); `work/context_rescue1_2026-10-23/cr1_score.py` (PROX, budgets, matching); `work/card_base/strict_mask.py` (map mask); `work/confirm_d_2026-10-25/dk_score.py`, `dk_a1.py`, `cd_analysis.py`, `work/context_rescue1_2026-10-23/cr1_analysis.py` (intervals); `work/confirm_d_2026-10-25/dk_a1_check.py` (independent recount) | `confirm_d_2026-10-25/RESULT_A1.json`, `CHECK_A1.json` |
| §5.4 Recipe repeatability over nine seed pairs | `confirm_d_2026-10-25/FREEZE.md` §6–7, `STAGE_B_NOTES.md` | D recipe `work/dense_terrain1_2026-10-25/dt1_ssl.py`, `dt1_train.py`; F1 recipe `work/d1_collapse_2026-10-06/d1_train.py`, `work/amazon_adapt_2026-09-28/adapt_train.py`, `work/adapt_nds_2026-10-05/plan_fit.py`; scorer `work/confirm_d_stage_b_2026-10-26/cdb_score.py`; recount `cdb_check.py` | `confirm_d_stage_b_2026-10-26/RESULT_B_K9.json`, `CHECK_B_K9.json` |
| §5.4 development precursor; §5.5 mechanism contrasts (B − A, D − B, D − G) | `dense_terrain1_2026-10-25/FREEZE.md` | `work/dense_terrain1_2026-10-25/dt1_*.py` | `dense_terrain1_2026-10-25/RESULT_AB.json`, `RESULT_FULL.json` |
| §5.5 Dutch footprint breakdown | as Table 2 | `ot_score.py` (TEST) | `RESULT_TEST.json` (`size_primary_budget`) |
| §5.5 Danish footprint strata | as Table 3 | as Table 3 | `RESULT_A1.json` (`recall_points`), `RESULT_B_K9.json` (`descriptive`) |
| §5.5 Danish relief-proxy (H) table | — | not included in this release | not included in this release |
| §5.6 Danish register supervision (−0.30 points) | `dk_regsup1_2026-10-29/FREEZE.md`, `FREEZE_ADDENDUM_1.md` | `work/dk_regsup1_2026-10-29/rs1_score.py`; recount `rs1_check.py` | `dk_regsup1_2026-10-29/RESULT_A.json`, `CHECK_A.json` |
| §4.1–4.2 systems; end-to-end S_D + PROX decode | `weights/SYSTEM.md` | `code/tools/clean_start/run_sd_prox.py` (needs an example manifest, not released), `setup_env.sh`, `requirements-sd-prox.txt` | — |

## Environment and dependency pins

- **Python packages:** `code/tools/clean_start/requirements-sd-prox.txt`, the exact pins of the reference decode
  environment (CPython 3.12.14; torch 2.14.0 for CUDA 13.0), installed with `--no-deps`
  (`code/tools/clean_start/setup_env.sh`). TF32 was disabled for decoding. The top-level `requirements-lock.txt` of
  this repository belongs to the earlier preprint.
- **ADAF:** https://github.com/EarthObservation/adaf at commit `a44acdd678773666458ea50f544b5d2292bcef51`; AiTLAS wheel
  SHA-256 `be2dbb7e3c39c30a91a79b00f7dc3365320cbc57584a8601e286a8b9c91ab69a`; public ADAF barrow checkpoint SHA-256
  `51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216` (Zenodo 15848663). Neither code nor weights of
  ADAF are redistributed. See the top-level "Upstream ADAF" section for where the code expects them.

## Project modules imported but not included

The scripts import these modules, which are not in `paper_a/`. Ten are in this repository at the same path from the
earlier release, byte-identical to the versions in the author's working repository on 2026-11-01; the others are not
released.

| Module (path in the author's repository) | Imported by | In this repository |
|---|---|---|
| `work/adapt_match_2026-10-05/match_train.py` | `dt1_model.py`, `d1_train.py`, `ea_train.py` | yes, identical |
| `work/adapt_match_2026-10-05/match_schedule.py` | `d1_train.py` | yes, identical |
| `work/d1_collapse_2026-10-06/d1_common.py` | `d1_train.py` | yes, identical |
| `work/counterfactual_2026-09-28/paired_train.py`, `plans.py` | `adapt_train.py` | yes, identical |
| `work/campaign_2026-09-22/r2_train.py` | `adapt_train.py` | yes, identical |
| `work/adapt_nds_2026-10-05/nds_common.py`, `nds_schedule.py`, `nds_train.py` | `plan_fit.py`, `ea_train.py` | yes, identical |
| `work/next_experiment_2026-09-23/r3_curation.py` | `ot_infer.py` (Verden parity gate only) | yes, identical |
| `work/open_transfer_test_2026-10-22/pipeline/ot_common.py`, `ot_fetch.py` | `ot_infer.py`, `dt1_decode.py` | no |
| `work/context_rescue1_2026-10-23/eval_score/cr1_eval_score.py` | `merge1.py`, `cd_analysis.py` | no |
| `work/context_rescue0_2026-10-23/dev_baseline_nl.py` | `cr1_score.py` | no |
| `work/card_base/overpass_batch.py` | `strict_mask.py` | no |
| `work/context_rescue0_2026-10-23/inventory_b_faint.py` | `ls_mosaic.py` | no |
| `work/holdout_region_2026-10/chain/masks.py` | `plan_fit.py` | no |
| `work/fresh_region_2026-10-03/step2/plan_fresh.py`, `prepixel_stats.py`, `step0.py` | `plan_fit.py` | no |
| `work/dense_terrain1_2026-10-25/dt1_coords.py` | `dt1_ssl.py` | no |
| `work/eng_adapt1_2026-10-19/ea_common.py` | `ea_train.py` | no |
| `work/scot_transfer_2026-10-19/scot_train.py` | `scot_score.py` (path reference) | no |

Also not included: the inference and scoring code of the Göttingen and EB packets, the EVAL copy of the Dutch pipeline,
the source of the §5.5 relief table, and the private inputs every scorer reads (`outputs/..._PRIVATE/`: register
snapshots, record and core lists, fold assignments and proposal files).

## Weights

The 58 checkpoints used in the paper (18.53 GB; CC BY-SA 4.0, as derivatives of the ADAF weights) are published on
Zenodo: https://doi.org/10.5281/zenodo.23862470. `weights/WEIGHTS_MANIFEST.csv`
lists every file with its SHA-256 and MD5. It is a CSV, not a checksum file; to check a download, run from the folder
holding the checkpoints and the CSV:

```bash
python3 -c "import csv; [print(r['sha256'] + '  ' + r['release_filename']) for r in csv.DictReader(open('WEIGHTS_MANIFEST.csv', newline=''))]" > weights.sha256
sha256sum -c weights.sha256
```

Checkpoints trained partly on Danish register labels (§5.6, arm A) are
withheld until the agency has given its view on the geometry licence (`weights/WITHHELD.md`).

## Data

All inputs are public at source and are not redistributed here:

- Netherlands: AHN4 DTM (PDOK) and RCE `landschapsatlas:grafheuvels`, CC0 1.0.
- Denmark: SDFI Danmarks Højdemodel (DHM/Terræn), free-data terms with attribution; Slots- og Kulturstyrelsen, Fund og
  Fortidsminder.
- Lower Saxony: LGLN DGM1 (CC BY 4.0) and Denkmalatlas Niedersachsen (CC BY-SA 4.0).

No proposal coordinates are published, matched or unmatched: unmatched proposals may mark unrecorded sites. No register
geometry, record identifier, core identifier or fold assignment from these experiments is in this folder. File names
ending in `_PRIVATE` in the code and protocols name the private inputs the code reads; their contents are not
released.

## Licence

Code: Apache License 2.0 (top-level `LICENSE`), © Thomas Shanahan. Weights (not in this repository): CC BY-SA 4.0.
