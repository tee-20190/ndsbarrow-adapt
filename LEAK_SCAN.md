# Leak scan

Scope: every file in this repository except `LEAK_SCAN.md` and `ALLOWLIST.txt` (226 files; `.npz` arrays are binary masks
and were checked structurally instead: uint8 inside-masks keyed by DGM1 tile name). Those two files were scanned separately after
they were written (result at the end).

## Rule

| Class | Pattern |
|---|---|
| `proj_pair` | projected easting/northing pair (6-7 digit numbers, UTM32 or BNG ranges) |
| `epsg` | EPSG code reference |
| `osgb` | OSGB grid reference (two letters + 3-5 + 3-5 digits) |
| `latlon_pair` | decimal latitude/longitude pair in European ranges (4+ decimals) |
| `latlon_kw` | lat/lon keyword with decimal value |
| `lead_id` | project lead-ID patterns (letter-digit-K-digits, DEV-nnn, Snnn and related) |
| `her_id` | HER / NHLE / SMR / MDO-type record IDs |
| `email` | email address other than the paper correspondence address |
| `token` | API key / token prefixes (sk-, ghp_, AIza, Bearer, ya29., xox) |
| `b64_secret` | long mixed-case alphanumeric strings (possible base64 secret) |
| `drive_id` | Google Drive / Docs URLs or file-ID fields |
| `sealed_test` | sealed English TEST square identifiers or geometry |
| `abs_path` | absolute local paths (WSL drive mount, Windows user folder, Linux home, Colab content directory) |
| `role_word` | role words and AI model/session names |

Allowed only inside byte-identical released evidence (decision D1): absolute local paths and role words. Files written
for the release must have 0 of them, with the declared exceptions below (definitions in `GLOSSARY.md`, verbatim commit
subjects in `COMMIT_LOG.md`, verbatim paths copied into `FREEZE_VERIFY.json` / `WITHHELD.md`, and the diff context of
`PATH_SHIM.patch`).

## Files withheld by the scan

Scanned before copying; real sensitive content, so not in the tree (listed in `WITHHELD.md`):

- `verify/astra_v2_labels.py`: names the sealed English TEST 100 km squares with their grid origins (sealed TEST geometry).
- `verify/v3/v3_fetch_england.py`: names the sealed English TEST 100 km squares (sealed TEST identity) in its refusal guard.

Also withheld before the scan by decision D2: `work/validation_cont_2026-09-26/DATA_MANIFEST.json`,
`work/validation_cont_2026-09-26/BLOCKS.json`, all proposal files.

## Dispositions

| Code | Disposition |
|---|---|
| R-D3 | verbatim private commit subject (D3 extract); defined in GLOSSARY.md |
| R-NHLE | public NHLE list-entry ID naming an English training-mask file or TRAIN-holdout window (public register; not a lead) |
| R-GLOSS | definition of the role word (GLOSSARY.md is the definitions file) |
| R-LIC | verbatim Apache License 2.0 text (legal term for the copyright holder, not a project role) |
| R-SHIM | removed/context line of a byte-identical released file inside the path-shim diff (the roots the shim replaces) |
| R-IDENT | underscore-joined identifier (variable, key or file stem), not a secret |
| R-ENT | registered-monument geometry / tile layout in a reclassified Denkmalatlas-derived manifest (public register; not a lead) |
| R-CRS | CRS identifier (no coordinate) |
| R-PROBE | round-number 420 m request box of an EA ImageServer availability probe (not a site) |
| R-D1 | byte-identical released evidence (D1); defined in GLOSSARY.md |

## Hit counts by class and disposition

| Class | Disposition | Hits | Files |
|---|---|---:|---:|
| `abs_path` | R-D1 | 89 | 41 |
| `abs_path` | R-SHIM | 74 | 1 |
| `b64_secret` | R-IDENT | 26 | 7 |
| `epsg` | R-CRS | 26 | 20 |
| `her_id` | R-NHLE | 33,912 | 6 |
| `proj_pair` | R-ENT | 382,726 | 13 |
| `proj_pair` | R-PROBE | 2 | 1 |
| `role_word` | R-D1 | 166 | 39 |
| `role_word` | R-D3 | 11 | 1 |
| `role_word` | R-GLOSS | 23 | 1 |
| `role_word` | R-LIC | 6 | 1 |

## Bulk hits by file

Hits under the bulk dispositions (R-CRS, R-D1, R-ENT, R-NHLE, R-PATH, R-SHIM) are listed per file with counts.

| File | Class | Disposition | Hits |
|---|---|---|---:|
| `FREEZE_VERIFY.json` | `her_id` | R-NHLE | 22,600 |
| `PATH_SHIM.patch` | `abs_path` | R-SHIM | 74 |
| `outputs/adapt_nds_2026-10-05/confirm/run/GROUP_MANIFEST_PRIVATE.json` | `epsg` | R-CRS | 1 |
| `outputs/adapt_nds_2026-10-05/confirm/run/GROUP_MANIFEST_PRIVATE.json` | `proj_pair` | R-ENT | 33,594 |
| `outputs/adapt_nds_2026-10-05/confirm/run/INDIVIDUAL_MANIFEST_PRIVATE.json` | `epsg` | R-CRS | 1 |
| `outputs/adapt_nds_2026-10-05/confirm/run/INDIVIDUAL_MANIFEST_PRIVATE.json` | `proj_pair` | R-ENT | 33,235 |
| `outputs/adapt_nds_2026-10-05/confirm/run/INFER_MANIFEST_PRIVATE.json` | `epsg` | R-CRS | 1 |
| `outputs/adapt_nds_2026-10-05/confirm/run/INFER_MANIFEST_PRIVATE.json` | `proj_pair` | R-ENT | 10,579 |
| `outputs/adapt_nds_2026-10-05/confirm/run/STAGING_PLAN_PRIVATE.json` | `proj_pair` | R-ENT | 3,590 |
| `outputs/adapt_nds_2026-10-05/fit/FIT_STAGING_PINNED_PRIVATE.json` | `proj_pair` | R-ENT | 23,931 |
| `outputs/adapt_nds_2026-10-05/fit/FIT_STAGING_PLAN_PRIVATE.json` | `proj_pair` | R-ENT | 9,496 |
| `outputs/adapt_nds_2026-10-05/fit/FIT_TABLE_PRIVATE.json` | `epsg` | R-CRS | 1 |
| `outputs/adapt_nds_2026-10-05/fit/FIT_TABLE_PRIVATE.json` | `proj_pair` | R-ENT | 167,561 |
| `outputs/adapt_nds_2026-10-05/fit/FIT_TILE_INDEX_PRIVATE.json` | `proj_pair` | R-ENT | 5,906 |
| `outputs/fresh_region_2026-10-03/step0/GROUP_MANIFEST_PRIVATE.json` | `epsg` | R-CRS | 1 |
| `outputs/fresh_region_2026-10-03/step0/GROUP_MANIFEST_PRIVATE.json` | `proj_pair` | R-ENT | 36,035 |
| `outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json` | `epsg` | R-CRS | 1 |
| `outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json` | `proj_pair` | R-ENT | 35,498 |
| `outputs/fresh_region_2026-10-03/step2/ENCLOSURE_MANIFEST_PRIVATE.json` | `epsg` | R-CRS | 1 |
| `outputs/fresh_region_2026-10-03/step2/ENCLOSURE_MANIFEST_PRIVATE.json` | `proj_pair` | R-ENT | 60 |
| `outputs/fresh_region_2026-10-03/step2/INFER_MANIFEST_PRIVATE.json` | `epsg` | R-CRS | 1 |
| `outputs/fresh_region_2026-10-03/step2/INFER_MANIFEST_PRIVATE.json` | `proj_pair` | R-ENT | 20,005 |
| `outputs/fresh_region_2026-10-03/step2/STAGING_PLAN_PRIVATE.json` | `proj_pair` | R-ENT | 3,236 |
| `verify/astra_v2_access.py` | `epsg` | R-CRS | 1 |
| `verify/v3/v3_e0099_landscape.py` | `epsg` | R-CRS | 1 |
| `verify/v3/v3_frozen_mixture.py` | `role_word` | R-D1 | 5 |
| `work/adaf_head_to_head_2026-09-19/adapter.py` | `epsg` | R-CRS | 1 |
| `work/adapt_match_2026-10-05/match_common.py` | `abs_path` | R-D1 | 1 |
| `work/adapt_nds_2026-10-05/chain/sessions_adapt.py` | `abs_path` | R-D1 | 1 |
| `work/adapt_nds_2026-10-05/freeze/PINS_FREEZE.json` | `her_id` | R-NHLE | 5,648 |
| `work/adapt_nds_2026-10-05/nds_common.py` | `abs_path` | R-D1 | 1 |
| `work/adapt_nds_2026-10-05/nds_common.py` | `epsg` | R-CRS | 1 |
| `work/adapt_nds_2026-10-05/nds_reader.py` | `epsg` | R-CRS | 2 |
| `work/bench1_2026-10-07/FUTILITY_GATE.txt` | `role_word` | R-D1 | 1 |
| `work/bench1_2026-10-07/PACKET_DRAFT_v2.txt` | `abs_path` | R-D1 | 1 |
| `work/bench1_2026-10-07/PACKET_DRAFT_v2.txt` | `role_word` | R-D1 | 5 |
| `work/bench1_2026-10-07/gate/PINS_FREEZE_G.json` | `her_id` | R-NHLE | 2 |
| `work/bgscale_2026-09-26/bg_eval.py` | `abs_path` | R-D1 | 1 |
| `work/bgscale_2026-09-26/bg_train.py` | `abs_path` | R-D1 | 1 |
| `work/campaign_2026-09-22/adaf_configs.py` | `epsg` | R-CRS | 1 |
| `work/colab_assessment_2026-09-29/env/setup_env.sh` | `abs_path` | R-D1 | 10 |
| `work/colab_cli_2026-09-30/cli_common.sh` | `abs_path` | R-D1 | 3 |
| `work/colab_cli_2026-09-30/upload_part.sh` | `abs_path` | R-D1 | 2 |
| `work/counterfactual_2026-09-28/cf2.py` | `role_word` | R-D1 | 2 |
| `work/counterfactual_2026-09-28/paired_train.py` | `abs_path` | R-D1 | 1 |
| `work/counterfactual_2026-09-28/plans.py` | `abs_path` | R-D1 | 1 |
| `work/counterfactual_2026-09-28/score_pairs.py` | `abs_path` | R-D1 | 1 |
| `work/d1_collapse_2026-10-06/PACKET.md` | `role_word` | R-D1 | 3 |
| `work/d1_collapse_2026-10-06/RESULT.md` | `role_word` | R-D1 | 1 |
| `work/d1_collapse_2026-10-06/chain/driver_d1.sh` | `abs_path` | R-D1 | 14 |
| `work/d1_collapse_2026-10-06/chain/driver_d1.sh` | `role_word` | R-D1 | 4 |
| `work/d1_collapse_2026-10-06/chain/vm_job_d1.sh` | `abs_path` | R-D1 | 5 |
| `work/d1_collapse_2026-10-06/d1_common.py` | `abs_path` | R-D1 | 1 |
| `work/d1_collapse_2026-10-06/freeze/PINS_FREEZE.json` | `her_id` | R-NHLE | 2 |
| `work/d1_verden_descriptive_2026-10-07/RESULT.md` | `abs_path` | R-D1 | 1 |
| `work/d1_verden_descriptive_2026-10-07/RESULT.md` | `role_word` | R-D1 | 4 |
| `work/d1_verden_descriptive_2026-10-07/desc_verden.py` | `abs_path` | R-D1 | 2 |
| `work/d1_verden_descriptive_2026-10-07/desc_verden.py` | `role_word` | R-D1 | 2 |
| `work/evaluator_v0_1_2026-10-07/FREEZE.md` | `role_word` | R-D1 | 1 |
| `work/evaluator_v0_1_2026-10-07/SPEC.md` | `role_word` | R-D1 | 2 |
| `work/evaluator_v0_1_2026-10-07/eval_v01.py` | `role_word` | R-D1 | 2 |
| `work/evaluator_v0_1_2026-10-07/run_checks.py` | `abs_path` | R-D1 | 2 |
| `work/fresh_region_2026-10-03/step2/fresh_common.py` | `epsg` | R-CRS | 2 |
| `work/fresh_region_2026-10-03/step2/fresh_common.py` | `role_word` | R-D1 | 1 |
| `work/fresh_region_2026-10-03/step2/lgln_reader.py` | `epsg` | R-CRS | 3 |
| `work/gpt6_pro_collab_2026-10-07/detector_week1/RESULT.md` | `abs_path` | R-D1 | 1 |
| `work/gpt6_pro_collab_2026-10-07/detector_week1/RESULT.md` | `role_word` | R-D1 | 2 |
| `work/gpt6_pro_collab_2026-10-07/detector_week1/bench_v0.py` | `abs_path` | R-D1 | 2 |
| `work/gpt6_pro_collab_2026-10-07/detector_week1/decomp.py` | `abs_path` | R-D1 | 2 |
| `work/gpt6_pro_collab_2026-10-07/detector_week1/ignore_check.py` | `abs_path` | R-D1 | 2 |
| `work/gpt6_pro_collab_2026-10-07/v0_uncertainty/RESULT.md` | `abs_path` | R-D1 | 1 |
| `work/gpt6_pro_collab_2026-10-07/v0_uncertainty/RESULT.md` | `epsg` | R-CRS | 1 |
| `work/gpt6_pro_collab_2026-10-07/v0_uncertainty/RESULT.md` | `role_word` | R-D1 | 2 |
| `work/gpt6_pro_collab_2026-10-07/v0_uncertainty/boot_v0.py` | `abs_path` | R-D1 | 2 |
| `work/gpt6_pro_collab_2026-10-07/v0_uncertainty/engine.py` | `role_word` | R-D1 | 2 |
| `work/gpt6_pro_collab_2026-10-07/v0_uncertainty/ignore_v0.py` | `abs_path` | R-D1 | 2 |
| `work/gpt6_pro_collab_2026-10-07/v0_uncertainty/register_semantics.py` | `abs_path` | R-D1 | 2 |
| `work/holdout_region_2026-10/chain/welsh_train.py` | `abs_path` | R-D1 | 1 |
| `work/holdout_region_2026-10/step2/region_score.py` | `her_id` | R-NHLE | 12 |
| `work/holdout_region_2026-10/step2/region_score.py` | `role_word` | R-D1 | 1 |
| `work/holdout_region_2026-10/step2/run_chain.sh` | `abs_path` | R-D1 | 1 |
| `work/hr3_ireland_2026-10-02/visibility/vis_common.py` | `role_word` | R-D1 | 1 |
| `work/hr3_ireland_2026-10-02/visibility/vis_irl.py` | `epsg` | R-CRS | 2 |
| `work/hr3_ireland_2026-10-02/visibility/vis_irl.py` | `role_word` | R-D1 | 2 |
| `work/joint_campaign_2026-10-03/chain/joint_train.py` | `role_word` | R-D1 | 2 |
| `work/manuscript_2026-10/CLAIMS_RELEASE.md` | `epsg` | R-CRS | 1 |
| `work/manuscript_2026-10/figures/make_figures.py` | `abs_path` | R-D1 | 2 |
| `work/manuscript_2026-10/ignore_rescore/RESULT.md` | `abs_path` | R-D1 | 1 |
| `work/manuscript_2026-10/ignore_rescore/rescore_d1_ignore.py` | `abs_path` | R-D1 | 2 |
| `work/next_experiment_2026-09-23/r3_curation.py` | `epsg` | R-CRS | 2 |
| `work/oldenburg_transfer_2026-10-07/FREEZE.md` | `role_word` | R-D1 | 3 |
| `work/oldenburg_transfer_2026-10-07/INFERENCE_PLAN.md` | `abs_path` | R-D1 | 6 |
| `work/oldenburg_transfer_2026-10-07/INFERENCE_PLAN.md` | `role_word` | R-D1 | 3 |
| `work/oldenburg_transfer_2026-10-07/PACKET.md` | `role_word` | R-D1 | 12 |
| `work/oldenburg_transfer_2026-10-07/RESULT.md` | `role_word` | R-D1 | 6 |
| `work/oldenburg_transfer_2026-10-07/score_oldenburg.py` | `abs_path` | R-D1 | 2 |
| `work/oldenburg_transfer_2026-10-07/score_oldenburg.py` | `role_word` | R-D1 | 1 |
| `work/seeds_2026-09-26/PACKET.md` | `role_word` | R-D1 | 6 |
| `work/seeds_2026-09-26/RESULT.md` | `role_word` | R-D1 | 5 |
| `work/stage1_d1_v0_2026-10-07/FREEZE.md` | `role_word` | R-D1 | 5 |
| `work/stage1_d1_v0_2026-10-07/RESULT.md` | `abs_path` | R-D1 | 1 |
| `work/stage1_d1_v0_2026-10-07/RESULT.md` | `role_word` | R-D1 | 5 |
| `work/stage1_d1_v0_2026-10-07/score_v0.py` | `abs_path` | R-D1 | 2 |
| `work/stage1_d1_v0_2026-10-07/step0.py` | `abs_path` | R-D1 | 2 |
| `work/stage2_d1_controls_2026-10-07/FREEZE.md` | `abs_path` | R-D1 | 1 |
| `work/stage2_d1_controls_2026-10-07/FREEZE.md` | `role_word` | R-D1 | 3 |
| `work/stage2_d1_controls_2026-10-07/PACKET.md` | `abs_path` | R-D1 | 1 |
| `work/stage2_d1_controls_2026-10-07/PACKET.md` | `role_word` | R-D1 | 36 |
| `work/stage2_d1_controls_2026-10-07/RESULT.md` | `role_word` | R-D1 | 8 |
| `work/stage2_d1_controls_2026-10-07/freeze/FREEZE.json` | `role_word` | R-D1 | 2 |
| `work/stage2_d1_controls_2026-10-07/freeze/PINS_FREEZE.json` | `her_id` | R-NHLE | 5,648 |
| `work/stage2_d1_controls_2026-10-07/s2_common.py` | `abs_path` | R-D1 | 1 |
| `work/stage2_d1_controls_2026-10-07/score_s2.py` | `abs_path` | R-D1 | 2 |
| `work/stage2_d1_controls_2026-10-07/score_s2.py` | `role_word` | R-D1 | 3 |
| `work/validation_cont_2026-09-26/CONFIG.json` | `role_word` | R-D1 | 4 |
| `work/validation_cont_2026-09-26/PACKET.md` | `abs_path` | R-D1 | 1 |
| `work/validation_cont_2026-09-26/PACKET.md` | `role_word` | R-D1 | 12 |
| `work/validation_cont_2026-09-26/PACKET_AMENDMENT_B.md` | `role_word` | R-D1 | 2 |
| `work/validation_cont_2026-09-26/RESULT.md` | `role_word` | R-D1 | 1 |
| `work/validation_cont_2026-09-26/stageA_frozen/CONFIG.json` | `role_word` | R-D1 | 4 |

## Itemised hits

Every other hit, with file and line (matched text omitted so that this file does not repeat it).

| File | Line | Class | Disposition |
|---|---:|---|---|
| `COMMIT_LOG.md` | 10 | `role_word` | R-D3 |
| `COMMIT_LOG.md` | 21 | `role_word` | R-D3 |
| `COMMIT_LOG.md` | 21 | `role_word` | R-D3 |
| `COMMIT_LOG.md` | 33 | `role_word` | R-D3 |
| `COMMIT_LOG.md` | 33 | `role_word` | R-D3 |
| `COMMIT_LOG.md` | 46 | `role_word` | R-D3 |
| `COMMIT_LOG.md` | 47 | `role_word` | R-D3 |
| `COMMIT_LOG.md` | 65 | `role_word` | R-D3 |
| `COMMIT_LOG.md` | 72 | `role_word` | R-D3 |
| `COMMIT_LOG.md` | 73 | `role_word` | R-D3 |
| `COMMIT_LOG.md` | 73 | `role_word` | R-D3 |
| `GLOSSARY.md` | 9 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 10 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 11 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 11 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 11 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 11 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 11 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 11 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 11 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 11 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 12 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 12 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 13 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 13 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 13 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 13 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 13 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 14 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 14 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 14 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 14 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 14 | `role_word` | R-GLOSS |
| `GLOSSARY.md` | 14 | `role_word` | R-GLOSS |
| `LICENSE` | 12 | `role_word` | R-LIC |
| `LICENSE` | 13 | `role_word` | R-LIC |
| `LICENSE` | 51 | `role_word` | R-LIC |
| `LICENSE` | 53 | `role_word` | R-LIC |
| `LICENSE` | 60 | `role_word` | R-LIC |
| `LICENSE` | 189 | `role_word` | R-LIC |
| `README.md` | 89 | `b64_secret` | R-IDENT |
| `README.md` | 90 | `b64_secret` | R-IDENT |
| `verify/astra_v2_access.py` | 41 | `proj_pair` | R-PROBE |
| `verify/astra_v2_access.py` | 41 | `proj_pair` | R-PROBE |
| `work/adapt_match_2026-10-05/match_common.py` | 79 | `b64_secret` | R-IDENT |
| `work/adapt_match_2026-10-05/match_common.py` | 83 | `b64_secret` | R-IDENT |
| `work/adapt_nds_2026-10-05/nds_common.py` | 99 | `b64_secret` | R-IDENT |
| `work/adapt_nds_2026-10-05/nds_common.py` | 102 | `b64_secret` | R-IDENT |
| `work/aim_2026-09-24/aim_analyze.py` | 140 | `b64_secret` | R-IDENT |
| `work/aim_2026-09-24/aim_analyze.py` | 143 | `b64_secret` | R-IDENT |
| `work/campaign_2026-09-22/adaf_configs.py` | 37 | `b64_secret` | R-IDENT |
| `work/fresh_region_2026-10-03/step2/fresh_common.py` | 88 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 748 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 749 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 750 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 753 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 754 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 755 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 756 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 757 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 1494 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 1495 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 1496 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 1499 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 1500 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 1501 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 1502 | `b64_secret` | R-IDENT |
| `work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json` | 1503 | `b64_secret` | R-IDENT |

## Result

Total hits: 417,061. Resolved: 417,061. **Unresolved: 0.**

Files written for the release (README.md, LICENSE, NOTICE, CITATION.cff, GLOSSARY.md, WITHHELD.md, COMMIT_LOG.md,
FREEZE_VERIFY.json, PATH_SHIM.patch, release_paths.py, release_shims/, .gitattributes): coordinates, grid references, lead IDs, record IDs
tied to leads, emails other than the correspondence address, tokens, Drive IDs and sealed-test identifiers: 0; absolute
paths and role words: only under the declared exceptions above.

Post-write check of `LEAK_SCAN.md` and `ALLOWLIST.txt` with the same rule: 0 hits, 0 unresolved (scanned after writing; `ALLOWLIST.txt` re-scanned after its final write).

## Addendum, 2026-10-09T17:36Z: reproduction-check fixes

The reproduction check (README, "Reproduction check") led to edits of `README.md`, `PATH_SHIM.patch` (one added hunk for
`work/hr3_ireland_2026-10-02/step2/chain/infer_irl.py`) and `release_paths.py` (docstring), and to three new files:
`requirements-lock.txt`, `tools/score_subset.py`, `tools/subset_manifest.py`. No byte-identical released file changed.
These six files were rescanned with the same rule and dispositions:

| File | Class | Disposition | Hits |
|---|---|---|---:|
| `PATH_SHIM.patch` | `abs_path` | R-SHIM | 74 (unchanged; the new hunk adds none) |
| `README.md` | `b64_secret` | R-IDENT | 2 (the two ADAF weight-file stems; now lines 91 and 92, previously 89 and 90) |
| `release_paths.py` | | | 0 |
| `requirements-lock.txt` | | | 0 |
| `tools/score_subset.py` | | | 0 |
| `tools/subset_manifest.py` | | | 0 |

Total hits: 76. Resolved: 76. **Unresolved: 0.** Coordinates, grid references, lead IDs, record IDs tied to leads, emails
other than the correspondence address, tokens, Drive IDs, local absolute paths and role words in the new or edited text:
0. The repository now holds 231 files (`ALLOWLIST.txt` lists 230; it does not list itself).

Post-write check of `LEAK_SCAN.md` and `ALLOWLIST.txt` after this addendum, same rule: 0 hits, 0 unresolved.

## Addendum, 2026-10-13: weights wording and v10 claim ledger

Edits: `README.md`, `NOTICE` and `WITHHELD.md` now say that the weights deposit on Zenodo is in preparation and that
its DOI will be added in a later release (the `ZENODO_WEIGHTS_DOI` placeholder is gone). The claim ledger
`work/manuscript_2026-10/CLAIMS_RELEASE.md` was rebuilt for paper version 10 (225 rows) and its scrub log
`work/manuscript_2026-10/CLAIMS_RELEASE_SCRUB_LOG.md` was added. The ledger is no longer a byte-identical copy of a
working-repository file, so it is treated as a file written for the release: disposition R-D1 does not apply to it, and
role words or local paths in it would be unresolved. No other file changed.

Same scanner and rule as above (the 14 classes in "Rule"), run on the five changed files:

| File | Class | Disposition | Hits |
|---|---|---|---:|
| `README.md` | `b64_secret` | R-IDENT | 2 (the two ADAF weight-file stems, now lines 90 and 91) |
| `NOTICE` | | | 0 |
| `WITHHELD.md` | | | 0 |
| `work/manuscript_2026-10/CLAIMS_RELEASE.md` | `epsg` | R-CRS | 1 (row C178: DGM1 tile CRS, no coordinate) |
| `work/manuscript_2026-10/CLAIMS_RELEASE_SCRUB_LOG.md` | | | 0 |

Total hits: 3. Resolved: 3. **Unresolved: 0.** Coordinates, grid references, lead IDs, record IDs tied to leads, emails
other than the correspondence address, tokens, Drive IDs, sealed-test identifiers, local absolute paths and role words
in the changed files: 0. The ledger also passed its own scrub checks (scrub log, "Scan output": 0 on every leak check;
all 56 numeric inspection hits are labelled values, DOIs or hash fragments). The repository now holds 232 files
(`ALLOWLIST.txt` lists 231; it does not list itself).

Post-write check of `LEAK_SCAN.md` and `ALLOWLIST.txt` after this addendum, same rule: 0 hits, 0 unresolved (scanned after writing; `ALLOWLIST.txt` re-scanned after its final write).
