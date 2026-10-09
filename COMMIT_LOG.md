# Commit log extract

Extract of the author's private project repository history (decision D3): every commit that touched a file released in
this repository, newest first. Columns: short hash, commit time (UTC), subject line, released paths the commit touched.
The full private history is not published because it also contains unrelated material (correspondence, candidate
sites and private data). Commit hashes cited in the paper (Section 4.7) appear here with their times.

Subjects are verbatim, except that a subject matching the release leak rule (coordinates, grid references, lead IDs,
place names, record IDs, emails, keys, sealed-terrain identifiers) is replaced by `[subject withheld: location-sensitive]`.
Replaced subjects: 0. Role words in subjects (for example "owner") are defined in `GLOSSARY.md`.

Commits: 58. Released files with commit history: 179 of 213.

| Commit | UTC | Subject | Released paths touched |
|---|---|---|---|
| `87ef7d35` | 2026-10-09T15:53:54Z | Manuscript: public claim ledger (215 rows, leak scan 0) and venue shortlist (13 journals, AI policies sourced) | `work/manuscript_2026-10/CLAIMS_RELEASE.md` |
| `9e9cbbdb` | 2026-10-09T06:05:50Z | Manuscript v7: second-referee fixes (Oldenburg v0 timing disclosed, overclaims removed); revised AI statement proposal | `work/manuscript_2026-10/figures/make_figures.py` |
| `de6bc382` | 2026-10-09T05:37:08Z | Manuscript v6: figures renumbered/cleaned, English comparison to supplement, code licence; proposed AI statement | `work/manuscript_2026-10/figures/make_figures.py` |
| `01a2142d` | 2026-10-09T03:33:48Z | Independent rescore REPRODUCED all German headline numbers (96/96 cells) | `work/independent_rescore_2026-10-09/COMPARE.json`<br>`work/independent_rescore_2026-10-09/RESCORE.md`<br>`work/independent_rescore_2026-10-09/RESCORE_RAW.json`<br>`work/independent_rescore_2026-10-09/compare.py`<br>`work/independent_rescore_2026-10-09/rescore.py` |
| `f6844b25` | 2026-10-09T03:20:46Z | Manuscript DRAFT_v1: methods from code, 13 verified refs, D1 IGNORE rescore (36/36 cells LB>0), figures F1-F8 | `work/manuscript_2026-10/figures/make_figures.py`<br>`work/manuscript_2026-10/ignore_rescore/D1_IGNORE_RESCORE.json`<br>`work/manuscript_2026-10/ignore_rescore/RESULT.md`<br>`work/manuscript_2026-10/ignore_rescore/rescore_d1_ignore.py` |
| `b5b8d11b` | 2026-10-09T02:08:32Z | GPT-6 Pro Round 5 reply + review; Oldenburg wording erratum; shortlist addendum; 40-CU detector allocation | `work/oldenburg_transfer_2026-10-07/RESULT.md` |
| `32a43d09` | 2026-10-09T00:11:23Z | OLDENBURG-TRANSFER-1 PASS (clear): D1 - ADAF barrow +3.86 pts RAW b=0.5, 10/10, joint LB +1.37; 32.4 CU | `work/oldenburg_transfer_2026-10-07/RESULT.json`<br>`work/oldenburg_transfer_2026-10-07/RESULT.md` |
| `d1e3a7e0` | 2026-10-08T16:48:12Z | STAGE2-D1-CONTROLS PASS (clear): D1 - replay control +12.3 pts [+8.4, +17.2], 10/10; control 6.1 pts below ADAF barrow | `work/stage2_d1_controls_2026-10-07/BOOT_S2.json`<br>`work/stage2_d1_controls_2026-10-07/RESULT.json`<br>`work/stage2_d1_controls_2026-10-07/RESULT.md` |
| `03a35fec` | 2026-10-08T01:12:25Z | OLDENBURG-TRANSFER-1: launch-ready Colab inference driver (checked copy of D1 session 3), tests, dry runs, restore plan | `work/oldenburg_transfer_2026-10-07/chain/DRIVER_FREEZE.json`<br>`work/oldenburg_transfer_2026-10-07/chain/DRIVER_FREEZE_LOCK.json` |
| `2174a8cf` | 2026-10-07T23:36:05Z | Evaluator v0.1: interface adapter for STAGE2 prespecified sensitivity hook (pre-scoring; no frozen file changed); hook test COMPUTED, reproduces Stage 1 +0.0625 | `work/evaluator_v0_1_2026-10-07/score_v0_1.py` |
| `bd144db0` | 2026-10-07T23:33:30Z | STAGE2-D1-CONTROLS frozen: Round 4 decisions applied (equal-update nonlocal mixed-class replay control) | `work/stage2_d1_controls_2026-10-07/COMPOSITION.json`<br>`work/stage2_d1_controls_2026-10-07/FREEZE.md`<br>`work/stage2_d1_controls_2026-10-07/PACKET.md`<br>`work/stage2_d1_controls_2026-10-07/boot_s2.py`<br>`work/stage2_d1_controls_2026-10-07/chain/sessions_s2.py`<br>`work/stage2_d1_controls_2026-10-07/freeze/FREEZE.json`<br>`work/stage2_d1_controls_2026-10-07/freeze/FREEZE_LOCK.json`<br>`work/stage2_d1_controls_2026-10-07/s2_barrier.py`<br>`work/stage2_d1_controls_2026-10-07/s2_composition.py`<br>`work/stage2_d1_controls_2026-10-07/s2_train.py`<br>`work/stage2_d1_controls_2026-10-07/score_s2.py` |
| `e6cf35cb` | 2026-10-07T22:55:58Z | OLDENBURG-TRANSFER-1 frozen: one-time transfer check of the ten D1 checkpoints; deployable checkpoint fixed | `work/oldenburg_transfer_2026-10-07/FREEZE.json`<br>`work/oldenburg_transfer_2026-10-07/FREEZE.md`<br>`work/oldenburg_transfer_2026-10-07/INFERENCE_PLAN.md`<br>`work/oldenburg_transfer_2026-10-07/LEGITIMACY.json`<br>`work/oldenburg_transfer_2026-10-07/PACKET.md`<br>`work/oldenburg_transfer_2026-10-07/freeze_oldenburg.py`<br>`work/oldenburg_transfer_2026-10-07/score_oldenburg.py` |
| `89189563` | 2026-10-07T22:45:20Z | D1 vs ADAF barrow on Verden: descriptive v0 bootstrap and pool terms | `work/d1_verden_descriptive_2026-10-07/RESULT.json`<br>`work/d1_verden_descriptive_2026-10-07/RESULT.md`<br>`work/d1_verden_descriptive_2026-10-07/TABLES.md`<br>`work/d1_verden_descriptive_2026-10-07/desc_verden.py`<br>`work/d1_verden_descriptive_2026-10-07/tables_desc.py` |
| `0f3443ea` | 2026-10-07T22:36:58Z | Evaluator v0.1 frozen: role-based records, explicit voids, member-supported complex endpoint | `work/evaluator_v0_1_2026-10-07/CHECKS.json`<br>`work/evaluator_v0_1_2026-10-07/FREEZE.json`<br>`work/evaluator_v0_1_2026-10-07/FREEZE.md`<br>`work/evaluator_v0_1_2026-10-07/SPEC.md`<br>`work/evaluator_v0_1_2026-10-07/VOIDS.json`<br>`work/evaluator_v0_1_2026-10-07/eval_v01.py`<br>`work/evaluator_v0_1_2026-10-07/freeze_v01.py`<br>`work/evaluator_v0_1_2026-10-07/run_checks.py` |
| `cb87b08a` | 2026-10-07T21:44:13Z | STAGE2 matched-control packet built (not frozen, not launched); 3 open design questions | `work/stage2_d1_controls_2026-10-07/PACKET.md`<br>`work/stage2_d1_controls_2026-10-07/s2_common.py`<br>`work/stage2_d1_controls_2026-10-07/s2_schedule.py`<br>`work/stage2_d1_controls_2026-10-07/s2_train.py`<br>`work/stage2_d1_controls_2026-10-07/score_s2.py` |
| `426c8be9` | 2026-10-07T21:32:43Z | v0 uncertainty engine: block bootstrap (13/13 tests), A-C lower bounds >= +1.8 pts, A-ADAF barrow spans 0; IGNORE readings; register semantics; A improved, C flat | `work/gpt6_pro_collab_2026-10-07/v0_uncertainty/RESULT.md`<br>`work/gpt6_pro_collab_2026-10-07/v0_uncertainty/boot_v0.py`<br>`work/gpt6_pro_collab_2026-10-07/v0_uncertainty/engine.py`<br>`work/gpt6_pro_collab_2026-10-07/v0_uncertainty/ignore_v0.py`<br>`work/gpt6_pro_collab_2026-10-07/v0_uncertainty/register_semantics.py`<br>`work/gpt6_pro_collab_2026-10-07/v0_uncertainty/tables_v0.py`<br>`work/gpt6_pro_collab_2026-10-07/v0_uncertainty/v0data.py` |
| `70fd2e03` | 2026-10-07T21:09:31Z | STAGE1-D1-V0 PASS (+6.25 pts at b=0.5 RAW, 10/10 vs unchanged ADAF barrow on Verden); MODEL-EVAL-1 grading rule frozen + session prompt | `work/stage1_d1_v0_2026-10-07/RESULT.json`<br>`work/stage1_d1_v0_2026-10-07/RESULT.md`<br>`work/stage1_d1_v0_2026-10-07/STEP0.json`<br>`work/stage1_d1_v0_2026-10-07/score_v0.py`<br>`work/stage1_d1_v0_2026-10-07/step0.py`<br>`work/stage1_d1_v0_2026-10-07/tables.py` |
| `1ea377ca` | 2026-10-07T21:03:12Z | Freeze STAGE1-D1-V0 gate (Pro R3-3) before scoring; file Pro Round 3 reply | `work/stage1_d1_v0_2026-10-07/FREEZE.md` |
| `a7296c05` | 2026-10-07T20:03:13Z | Week-1 detector checks (union decomposition, protocol-scorer AGREE, v0 rescoring); Round 3 prompt + forecasts | `work/gpt6_pro_collab_2026-10-07/detector_week1/BENCH_V0.json`<br>`work/gpt6_pro_collab_2026-10-07/detector_week1/IGNORE_CHECK.json`<br>`work/gpt6_pro_collab_2026-10-07/detector_week1/RESULT.md`<br>`work/gpt6_pro_collab_2026-10-07/detector_week1/UNION_DECOMP_ALL.json`<br>`work/gpt6_pro_collab_2026-10-07/detector_week1/bench_v0.py`<br>`work/gpt6_pro_collab_2026-10-07/detector_week1/decomp.py`<br>`work/gpt6_pro_collab_2026-10-07/detector_week1/ignore_check.py`<br>`work/gpt6_pro_collab_2026-10-07/detector_week1/ignore_types.py`<br>`work/gpt6_pro_collab_2026-10-07/detector_week1/tables.py` |
| `18ecba60` | 2026-10-07T12:15:03Z | BENCH-1 futility gate G scored: FUTILE (NO_LAUNCH); D3 starts from ADAF barrow + F1 | `work/bench1_2026-10-07/gate/G1_COLLAPSE.json`<br>`work/bench1_2026-10-07/gate/GATE_RESULT.json`<br>`work/bench1_2026-10-07/gate/GATE_RESULT.txt` |
| `4547bbd7` | 2026-10-07T08:30:37Z | BENCH-1 futility gate G frozen (e1a69394) before any GPU use | `work/bench1_2026-10-07/gate/FREEZE_G.json`<br>`work/bench1_2026-10-07/gate/FREEZE_LOCK_G.json`<br>`work/bench1_2026-10-07/gate/PINS_FREEZE_G.json` |
| `3bbe79e1` | 2026-10-07T08:10:38Z | BENCH-1 draft v2 after critic review: futility gate first, R1 reuses D1 weights | `work/bench1_2026-10-07/FUTILITY_GATE.txt`<br>`work/bench1_2026-10-07/PACKET_DRAFT_v2.txt` |
| `108f9000` | 2026-10-07T07:18:41Z | D1-COLLAPSE-1 scored: frozen exit rule PASS (F1, 0/10 collapsed per arm) | `work/d1_collapse_2026-10-06/RECIPE.md`<br>`work/d1_collapse_2026-10-06/RESULT.json`<br>`work/d1_collapse_2026-10-06/RESULT.md` |
| `c826ddd4` | 2026-10-06T22:10:32Z | D1 development: F1 selected (0/3 collapsed, BARROW_L Verden 0.701/0.716/0.699); recipe frozen | `work/d1_collapse_2026-10-06/RECIPE_SELECTION.json` |
| `3e40aa5a` | 2026-10-06T20:01:12Z | D1-COLLAPSE-1: diagnosis, packet and frozen chain (f323e17e) before any GPU run | `work/d1_collapse_2026-10-06/DIAGNOSIS.md`<br>`work/d1_collapse_2026-10-06/PACKET.md`<br>`work/d1_collapse_2026-10-06/chain/bundle_d1.py`<br>`work/d1_collapse_2026-10-06/chain/chain_d1_run.sh`<br>`work/d1_collapse_2026-10-06/chain/chain_lib_d1.sh`<br>`work/d1_collapse_2026-10-06/chain/d1_gates.py`<br>`work/d1_collapse_2026-10-06/chain/driver_d1.sh`<br>`work/d1_collapse_2026-10-06/chain/freeze_d1.py`<br>`work/d1_collapse_2026-10-06/chain/sessions_d1.py`<br>`work/d1_collapse_2026-10-06/chain/vm_job_d1.sh`<br>`work/d1_collapse_2026-10-06/d1_common.py`<br>`work/d1_collapse_2026-10-06/d1_train.py`<br>`work/d1_collapse_2026-10-06/freeze/FREEZE.json`<br>`work/d1_collapse_2026-10-06/freeze/FREEZE_LOCK.json`<br>`work/d1_collapse_2026-10-06/freeze/PINS_FREEZE.json`<br>`work/d1_collapse_2026-10-06/score_d1.py` |
| `cd234691` | 2026-10-05T17:24:33Z | ADAPT-NDS-1 scored: ADAPT_GAIN on CONFIRM Oldenburg (A - C +0.141 [+0.090, +0.181]); A - ADAF barrow NOT_DISTINGUISHED (+0.045 [-0.011, +0.096]) | `work/adapt_nds_2026-10-05/ERRATUM_1.md`<br>`work/adapt_nds_2026-10-05/RECOUNT.json`<br>`work/adapt_nds_2026-10-05/RESULT.json`<br>`work/adapt_nds_2026-10-05/RESULT.md` |
| `08e291e3` | 2026-10-05T07:09:38Z | ADAPT-MATCH-0 built (not frozen): trainer, chain, scorer, 91 tests pass, dry run PASS | `work/adapt_match_2026-10-05/chain/bundle_match.py`<br>`work/adapt_match_2026-10-05/chain/freeze_match.py`<br>`work/adapt_match_2026-10-05/match_common.py`<br>`work/adapt_match_2026-10-05/match_schedule.py`<br>`work/adapt_match_2026-10-05/match_train.py`<br>`work/adapt_match_2026-10-05/score_match.py` |
| `e1897f06` | 2026-10-05T04:35:57Z | ADAPT-NDS-1 built and frozen (ba07548b): CONFIRM = Oldenburg by count rule (272 groups, 300.5 km2 background after NODATA), FIT 852+852 windows from Uelzen/Harburg/Heidekreis, A vs equal-update C, 58 tests, GPU smoke PASS, dry storage PASS; superseded freeze db427571 kept | `work/adapt_nds_2026-10-05/chain/adapt_gates.py`<br>`work/adapt_nds_2026-10-05/chain/package_results.sh`<br>`work/adapt_nds_2026-10-05/chain/sessions_adapt.py`<br>`work/adapt_nds_2026-10-05/freeze/FREEZE.json`<br>`work/adapt_nds_2026-10-05/freeze/FREEZE_LOCK.json`<br>`work/adapt_nds_2026-10-05/freeze/PINS_FREEZE.json`<br>`work/adapt_nds_2026-10-05/nds_common.py`<br>`work/adapt_nds_2026-10-05/nds_reader.py`<br>`work/adapt_nds_2026-10-05/nds_schedule.py`<br>`work/adapt_nds_2026-10-05/nds_train.py`<br>`work/adapt_nds_2026-10-05/score_adapt.py` |
| `94c74217` | 2026-10-05T00:43:01Z | ADAPT-NDS-1: forecasts revised before any data (calibration rule, FRESH-1 power) | `work/adapt_nds_2026-10-05/PACKET.md` |
| `c1cc68f1` | 2026-10-05T00:41:46Z | ADAPT-NDS-1 packet fixed before any data: Lower Saxony register-barrow adaptation of J (3 FIT districts, Verden DEV, CONFIRM district chosen by count rule), A vs equal-update control vs ADAF barrow; outcome rules and forecasts | `work/adapt_nds_2026-10-05/PACKET.md` |
| `4e978699` | 2026-10-03T17:26:40Z | WIP (owner-requested pause): FRESH-1 Step 2 amendment A1 incomplete — ADAF barrow HRNet as core reference, A1 verdict; 101 pass / 2 fail (STEP2_AMENDMENT_A1.md not yet written); remaining steps in A1_STATUS.md. Freeze refuses until A1 is finished. | `work/fresh_region_2026-10-03/step2/fresh_common.py` |
| `801a3e97` | 2026-10-03T12:00:26Z | FRESH-1 Step 2 chain built and tested (76 tests, 3-tile dry run PASS; no launch) + coordinator decisions | `work/fresh_region_2026-10-03/step2/fresh_common.py`<br>`work/fresh_region_2026-10-03/step2/fresh_gates.py`<br>`work/fresh_region_2026-10-03/step2/lgln_reader.py` |
| `3d629d0d` | 2026-10-03T09:40:27Z | JOINT-1 frozen (FREEZE 0fbdab1b, 5,739 pins) after two Colab smokes; session 1 launched | `work/joint_campaign_2026-10-03/chain/joint_train.py` |
| `ea03267a` | 2026-10-03T08:26:05Z | Gap table (all regions), FUSION-1 result, JOINT-1 packet draft, mixture audit and joint trainer | `work/fusion_2026-10-03/fuse_eval.py`<br>`work/gap_analysis_2026-10-03/gap_region.py`<br>`work/joint_campaign_2026-10-03/chain/joint_common.py`<br>`work/joint_campaign_2026-10-03/chain/joint_train.py` |
| `8fcfdbde` | 2026-10-03T07:58:13Z | HR3 Step 2 classified: ADAPT_GAIN on both contrasts (exploratory), independent recount MATCH; Irish gap table | `work/gap_analysis_2026-10-03/gap_region.py` |
| `8131b33f` | 2026-10-02T23:39:56Z | HR3 Step 2 chain: Irish manifests (per-part ownership tiles, AMENDMENT_2 background 156.2525 km2), staging, masks, trainer, inference, gates, driver, freeze and Step 5 scorer; 29 CPU tests pass; bit-identity with Step 1 | `work/hr3_ireland_2026-10-02/step2/chain/infer_irl.py`<br>`work/hr3_ireland_2026-10-02/step2/chain/irl_common.py` |
| `db888dff` | 2026-10-02T21:25:04Z | HR3 Step 1: fetch (72 GSI tiles in memory, 237 windows), Amendment 2 (controls must pass V2; 7 replaced in draw order), NODATA 160/214 eligible (floor 120 met), audit sample 9 barrows + 50 enclosures + 20 controls, AUDIT_FREEZE 21:23:11Z, 79 panels rendered | `work/hr3_ireland_2026-10-02/visibility/vis_common.py`<br>`work/hr3_ireland_2026-10-02/visibility/vis_irl.py` |
| `11fef773` | 2026-10-02T21:14:38Z | HR3 Ireland: pre-pixel packet, Amendment 1 (2 km DEV squares, IGNORE discs), Step 1 procedure + Amendment 1 (GeoTIFF tiles, bilinear onto 1 m ITM) locked before any DEV/FIT pixel; audit code adapted from HR2 (34 tests pass); LOWCONF-1 rank-matched erratum wording | `work/hr3_ireland_2026-10-02/visibility/vis_common.py`<br>`work/hr3_ireland_2026-10-02/visibility/vis_irl.py` |
| `21d9c505` | 2026-10-01T00:30:19Z | Freeze verified HR1 T2 inputs and sequential seed sessions | `work/holdout_region_2026-10/chain/bundle_lists.py`<br>`work/holdout_region_2026-10/chain/bundle_stream.py` |
| `9c813825` | 2026-09-30T19:30:17Z | HOLDOUT-REGION-1 Stage A FROZEN (K=3, EARLY_START) | `work/holdout_region_2026-10/chain/bundle_lists.py`<br>`work/holdout_region_2026-10/chain/bundle_stream.py`<br>`work/holdout_region_2026-10/chain/common.py`<br>`work/holdout_region_2026-10/chain/infer_dev.py`<br>`work/holdout_region_2026-10/chain/welsh_train.py`<br>`work/holdout_region_2026-10/step2/region_score.py` |
| `61f719e5` | 2026-09-30T16:28:49Z | HOLDOUT-REGION-1: repair scorer overlap and watchdog race before freeze | `work/holdout_region_2026-10/step2/region_score.py`<br>`work/holdout_region_2026-10/step2/run_chain.sh` |
| `35899df9` | 2026-09-30T16:12:18Z | Adopt the official Colab CLI as the GPU transport (COLAB-CLI-1 PASS) | `work/colab_cli_2026-09-30/cli_common.sh`<br>`work/colab_cli_2026-09-30/upload_part.sh` |
| `a6266c91` | 2026-09-30T08:34:54Z | HOLDOUT-REGION-1 Step 2a: harness built, sensitivity EXPLORATORY | `work/holdout_region_2026-10/step2/hier_uncertainty.py`<br>`work/holdout_region_2026-10/step2/region_score.py`<br>`work/holdout_region_2026-10/step2/run_chain.sh` |
| `4902ce29` | 2026-09-30T06:42:24Z | Measure Colab capacity and adopt it for GPU packets (COLAB-ASSESS-1) | `work/colab_assessment_2026-09-29/env/freeze.txt`<br>`work/colab_assessment_2026-09-29/env/setup_env.sh`<br>`work/colab_assessment_2026-09-29/make_bundle.py` |
| `e38f8653` | 2026-09-28T18:25:11Z | Amazon DEV feasibility: supervision, void policy and frozen AMZ-ADAPT-1 pilot packet | `work/amazon_adapt_2026-09-28/adapt_train.py` |
| `0db45d91` | 2026-09-28T01:42:47Z | Experiment 1 v2: consolidated protocol, qualified pipeline, pilot STOP_THIS_VERSION | `work/counterfactual_2026-09-28/cf2.py`<br>`work/counterfactual_2026-09-28/cf_v1/counterfactual.py`<br>`work/counterfactual_2026-09-28/ledger.py`<br>`work/counterfactual_2026-09-28/paired_train.py`<br>`work/counterfactual_2026-09-28/plans.py`<br>`work/counterfactual_2026-09-28/score_pairs.py` |
| `2f935566` | 2026-09-27T00:24:36Z | Restore frozen storage_guard.py; move archive-first gate to archive_gate.py; R1 preflight PASS | `verify/v3/storage_guard.py` |
| `8b5d50ee` | 2026-09-27T00:06:45Z | storage_guard: archive-first gate before any new guarded launch | `verify/v3/storage_guard.py` |
| `c9aa5f3a` | 2026-09-26T23:11:08Z | Owner corrections to the seeds write-up; seeds archive staged | `work/seeds_2026-09-26/RESULT.md` |
| `c5b56fb0` | 2026-09-26T22:56:06Z | Three-seed result: H1 PARTIAL, H2 HARM; TEST stays sealed (owner) | `work/seeds_2026-09-26/RESULT.md` |
| `882a8d3b` | 2026-09-26T20:06:54Z | Commit the clean-checkout closure, its kit and the TEST runner | `verify/v3/v3_frozen_mixture.py`<br>`work/adaf_head_to_head_2026-09-19/adapter.py`<br>`work/campaign_2026-09-22/adaf_configs.py`<br>`work/campaign_2026-09-22/screen_scorer.py`<br>`work/tta_ensemble_2026-09-24/tta_run.py` |
| `e73bf132` | 2026-09-26T18:32:16Z | Freeze the three-seed replication packet for CONT e41 (not launched) | `work/seeds_2026-09-26/FREEZE.json`<br>`work/seeds_2026-09-26/PACKET.md` |
| `adb6ba79` | 2026-09-26T05:31:53Z | VALIDATION look: CONT e41 beats public ADAF by 9.4 points (success rule met) | `work/validation_cont_2026-09-26/CONFIG.json`<br>`work/validation_cont_2026-09-26/FREEZE.json`<br>`work/validation_cont_2026-09-26/FREEZE_B.json`<br>`work/validation_cont_2026-09-26/PACKET.md`<br>`work/validation_cont_2026-09-26/PACKET_AMENDMENT_B.md`<br>`work/validation_cont_2026-09-26/RESULT.md`<br>`work/validation_cont_2026-09-26/stageA_frozen/CONFIG.json` |
| `9b3d201b` | 2026-09-26T04:38:36Z | Freeze Phase A of the background scale-up experiment | `work/bgscale_2026-09-26/bg_eval.py`<br>`work/bgscale_2026-09-26/bg_rule.py`<br>`work/bgscale_2026-09-26/bg_train.py` |
| `c2caf307` | 2026-09-26T02:17:04Z | Add verified Drive archival and disk guards for project runs | `verify/v3/storage_guard.py` |
| `819f0b15` | 2026-09-25T21:04:27Z | Diagnose CONT e41 failure stages on TRAIN and freeze a matched annealing test | `work/anneal_2026-09-25/anneal_holdout.py`<br>`work/anneal_2026-09-25/anneal_train.py` |
| `00c763a6` | 2026-09-22T04:11:27Z | Prepare smoke-tested E0099 landscape control for Luna | `verify/v3/v3_e0099_landscape.py` |
| `ecb269f6` | 2026-09-11T16:26:30Z | Publish research snapshot and GPT-6 Pro handoff | `verify/astra_v2_access.py`<br>`verify/astra_v2_classes.py` |

## Released files without commit history

These files are released byte-identical but were never committed to the private repository (git-ignored output
directories or untracked files). Their bytes are pinned by the freeze files listed in `FREEZE_VERIFY.json` where a freeze pins them.

- `outputs/adapt_nds_2026-10-05/confirm/run/GROUP_MANIFEST_PRIVATE.json`
- `outputs/adapt_nds_2026-10-05/confirm/run/INDIVIDUAL_MANIFEST_PRIVATE.json`
- `outputs/adapt_nds_2026-10-05/confirm/run/INFER_MANIFEST_PRIVATE.json`
- `outputs/adapt_nds_2026-10-05/confirm/run/INSIDE_MASKS_PRIVATE.npz`
- `outputs/adapt_nds_2026-10-05/confirm/run/STAGING_PLAN_PRIVATE.json`
- `outputs/adapt_nds_2026-10-05/fit/FIT_INSIDE_MASKS_PRIVATE.npz`
- `outputs/adapt_nds_2026-10-05/fit/FIT_STAGING_PINNED_PRIVATE.json`
- `outputs/adapt_nds_2026-10-05/fit/FIT_STAGING_PLAN_PRIVATE.json`
- `outputs/adapt_nds_2026-10-05/fit/FIT_TABLE_PRIVATE.json`
- `outputs/adapt_nds_2026-10-05/fit/FIT_TILE_INDEX_PRIVATE.json`
- `outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z/SESSION_VERIFIED.json`
- `outputs/fresh_region_2026-10-03/step0/EDGE_SUPPLEMENT_PRIVATE.json`
- `outputs/fresh_region_2026-10-03/step0/GROUP_MANIFEST_PRIVATE.json`
- `outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json`
- `outputs/fresh_region_2026-10-03/step0/NODATA_MEMBERS_PRIVATE.json`
- `outputs/fresh_region_2026-10-03/step0/TILE_INDEX_PRIVATE.json`
- `outputs/fresh_region_2026-10-03/step0/TILE_LEDGER_PRIVATE.jsonl`
- `outputs/fresh_region_2026-10-03/step2/ENCLOSURE_MANIFEST_PRIVATE.json`
- `outputs/fresh_region_2026-10-03/step2/INFER_MANIFEST_PRIVATE.json`
- `outputs/fresh_region_2026-10-03/step2/INSIDE_MASKS_PRIVATE.npz`
- `outputs/fresh_region_2026-10-03/step2/STAGING_PLAN_PRIVATE.json`
- `work/aim_2026-09-24/aim_analyze.py`
- `work/aim_train_audit_2026-09-24/cont_train.py`
- `work/aim_train_audit_2026-09-24/holdout_eval.py`
- `work/campaign_2026-09-22/r2_masks.py`
- `work/campaign_2026-09-22/r2_train.py`
- `work/gpt6_pro_collab_2026-10-07/v0_uncertainty/BOOT_V0.json`
- `work/gpt6_pro_collab_2026-10-07/v0_uncertainty/IGNORE_V0.json`
- `work/gpt6_pro_collab_2026-10-07/v0_uncertainty/REGISTER_SEMANTICS.json`
- `work/next_experiment_2026-09-23/r3_curation.py`
- `work/stage2_2026-09-24/stage2_run.py`
- `work/stage2_d1_controls_2026-10-07/freeze/PINS_FREEZE.json`
- `work/tta_ensemble_2026-09-24/score_holdout.py`
- `work/tta_ensemble_2026-09-24/score_screen.py`
