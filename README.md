# ndsbarrow-adapt

Code, frozen evaluation protocols, per-run aggregate results, register-derived entity manifests and the claim ledger for
the paper:

> Thomas Shanahan. *Adapting a public LiDAR barrow detector to a new region: specialist-start fine-tuning, an
> equal-update replay comparison, and evaluation choices that move apparent gains.* Preprint, 2026.

Model weights (54 checkpoints, CC BY-SA 4.0) are deposited separately: **`ZENODO_WEIGHTS_DOI`**.
`ZENODO_WEIGHTS_DOI` is the one placeholder in this repository; it is replaced by the Zenodo DOI of the weights record at
release.

## What is here

Every released file sits at the repository-relative path it had in the author's working repository, byte-identical to
the bytes that produced the paper's numbers. This keeps the SHA-256 values in the freeze files checkable
(`FREEZE_VERIFY.json`). Because the files are unedited, they contain the working vocabulary of the project (role words,
`_PRIVATE` file names, paths of the original machine); `GLOSSARY.md` explains them.

| Path | Content |
|---|---|
| `work/<experiment>/` | protocols (`PACKET.md`, `FREEZE*.json`/`.md`), results (`RESULT.json`/`.md`) and the code of each experiment |
| `work/hr3_ireland_2026-10-02/step2/chain/infer_irl.py` | inference and decoder entry point (Section 4.1) |
| `work/fresh_region_2026-10-03/step2/lgln_reader.py`, `work/adapt_nds_2026-10-05/nds_reader.py` | LGLN DGM1 tile staging (Verden, Oldenburg, FIT districts) |
| `work/stage1_d1_v0_2026-10-07/score_v0.py` | benchmark v0 scorer (Stage 1) |
| `work/d1_collapse_2026-10-06/d1_train.py` (+ imports, `RECIPE.md`) | D1 training recipe (Section 4.2) |
| `work/stage2_d1_controls_2026-10-07/s2_train.py` | Stage 2 replay-control trainer (Section 4.3) |
| `outputs/fresh_region_2026-10-03/`, `outputs/adapt_nds_2026-10-05/` | entity manifests, tile layout, staging plans and inside masks derived from the Denkmalatlas Niedersachsen and LGLN tile indexes (see "Data") |
| `work/manuscript_2026-10/CLAIMS_RELEASE.md` | claim ledger: every number in the paper with its source file |
| `FREEZE_VERIFY.json` | every hash pinned by every freeze file, re-checked on the released bytes |
| `WITHHELD.md` | files referenced or relied on but not released, with size, SHA-256 and reason |
| `COMMIT_LOG.md` | private commit history restricted to released files (hash, UTC time, subject) |
| `ALLOWLIST.txt` | every file in this repository with its SHA-256 |
| `LEAK_SCAN.md` | the release leak scan and the disposition of every hit |
| `PATH_SHIM.patch`, `release_paths.py` | path shim that makes a runnable working copy (see "Running the code") |
| `release_shims/` | fail-closed replacements for two withheld modules, used only in the runnable working copy |
| `tools/subset_manifest.py`, `tools/score_subset.py` | written for the release: subset inference manifest; scoring of any proposal file on any tile list with the independent scorer's functions (see "Running and scoring a subset of tiles") |
| `requirements-lock.txt` | exact package pins of the reproduction environment (see "Environment") |

## Freeze files and paper sections

Each decision rule was frozen as a hashed file before the outcome it decides was computed (paper Section 4.7).
`FREEZE_VERIFY.json` re-hashes every pinned file that is in this repository; pinned files that are not released are
listed there as `withheld` and in `WITHHELD.md`.

| Experiment | Freeze file(s) | SHA-256 prefix | Commit | Paper |
|---|---|---|---|---|
| Stage 1: Verden, D1 vs ADAF barrow | `work/stage1_d1_v0_2026-10-07/FREEZE.md` | `2d2ccda9` | `1ea377ca` | 4.7 (rule 1), 5.1, S1, S3 |
| Oldenburg transfer check | `work/oldenburg_transfer_2026-10-07/FREEZE.json`; `chain/DRIVER_FREEZE.json` + `DRIVER_FREEZE_LOCK.json` | **`19289f9d`**; `158dd5a5` | `e6cf35cb`; `03a35fec` | 4.7 (rule 3), 5.2, S2 |
| Stage 2: equal-update replay control | `work/stage2_d1_controls_2026-10-07/freeze/FREEZE.json` + `FREEZE_LOCK.json` + `PINS_FREEZE.json`; `FREEZE.md` | **`b5fdd0a9`** | `bd144db0` | 4.3, 4.7 (rule 2), 5.3 |
| D1 recipe selection and confirmation | `work/d1_collapse_2026-10-06/freeze/FREEZE.json` + `FREEZE_LOCK.json` + `PINS_FREEZE.json` | `f323e17e` | `3e40aa5a`, scored `108f9000` | 4.2, 4.7 |
| Earlier adaptation (ADAPT-NDS-1) | `work/adapt_nds_2026-10-05/freeze/FREEZE.json` + `FREEZE_LOCK.json` + `PINS_FREEZE.json` | `ba07548b` | `e1897f06` | 5.4 |
| F1 start comparison (BENCH-1 gate) | `work/bench1_2026-10-07/gate/FREEZE_G.json` + `FREEZE_LOCK_G.json` + `PINS_FREEZE_G.json` | `e1a69394` | `4547bbd7` | 5.4 |
| Evaluator v0.1 (sensitivity) | `work/evaluator_v0_1_2026-10-07/FREEZE.json`, `FREEZE.md` | `7ed3114d` | `0f3443ea` | 4.6, S5 |
| English comparison | `work/validation_cont_2026-09-26/FREEZE.json`, `FREEZE_B.json`; `work/seeds_2026-09-26/FREEZE.json` | `a8eae6f4`, `aff30b48`; `2087150e` | `adb6ba79` (committed with the result; freeze times are recorded inside the files); `e73bf132` | 5.7, S7 |

The two hashes printed in the paper (bold) verify on the released bytes (`FREEZE_VERIFY.json`, `cited_in_paper`).
The English comparison's Phase A freeze pins an earlier `CONFIG.json`; the current file was amended under `FREEZE_B.json`
and the Phase A bytes are released at `work/validation_cont_2026-09-26/stageA_frozen/CONFIG.json`. Its
`DATA_MANIFEST.json` and `BLOCKS.json` are withheld (sealed test-square identity and block coordinates; `WITHHELD.md`).

Analyses without their own freeze: benchmark v0 rescoring of earlier systems
(`work/gpt6_pro_collab_2026-10-07/detector_week1/`, Sections 4.4, 5.5, 5.6, Table 5); bootstrap engine and IGNORE
readings (`work/gpt6_pro_collab_2026-10-07/v0_uncertainty/`, Sections 4.5-4.6); Verden descriptive bootstrap
(`work/d1_verden_descriptive_2026-10-07/`, 5.1); D1 IGNORE rescore (`work/manuscript_2026-10/ignore_rescore/`, 5.6,
Table 6); independent second scorer (`work/independent_rescore_2026-10-09/`, 4.6); figures
(`work/manuscript_2026-10/figures/make_figures.py`, reads aggregate RESULT files only).

## Data

- **Terrain:** LGLN DGM1 (CC BY 4.0), not redistributed. The staging plans (`*/STAGING_PLAN_PRIVATE.json`,
  `FIT_STAGING_*_PRIVATE.json`) list every tile URL with the byte count and SHA-256 (or Content-Length and ETag) the runs
  used, so upstream changes are detected at staging.
- **Register:** Denkmalatlas Niedersachsen (CC BY-SA 4.0), not redistributed as raw snapshots. The released entity
  manifests (`INDIVIDUAL_`, `GROUP_`, `INFER_`, `ENCLOSURE_MANIFEST_PRIVATE.json`, `FIT_TABLE_PRIVATE.json`, tile indexes,
  ledgers, inside masks) are derivatives of the Denkmalatlas and LGLN data and carry CC BY-SA 4.0 with attribution
  (`NOTICE`). They hold registered monuments, tile layout and sampling cells only; they hold no detector proposals.
  Their names still end in `_PRIVATE` because the code reads them under those names (`GLOSSARY.md`).
- **Detector proposals are not released.** Unmatched proposals may mark unrecorded archaeological sites. They can be
  regenerated from the released weights and public terrain (below).

## Upstream ADAF (not redistributed)

The code loads the unedited ADAF model definitions and the AiTLAS package distributed with ADAF (decision D6):

| Item | Pin |
|---|---|
| ADAF code | https://github.com/EarthObservation/adaf at commit `a44acdd678773666458ea50f544b5d2292bcef51` (Apache-2.0) |
| AiTLAS wheel | `installation/aitlas-0.0.1-py3-none-any.whl` in that commit, 154,602 bytes, SHA-256 `be2dbb7e3c39c30a91a79b00f7dc3365320cbc57584a8601e286a8b9c91ab69a` |
| ADAF weights | Zenodo record 15848663 (ADAF ML model weights v1.0.0; concept DOI 10.5281/zenodo.15848662), CC BY-SA 4.0 |
| ADAF barrow HRNet (start checkpoint and comparator) | `barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar`, 861,530,273 bytes, SHA-256 `51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216` |
| ADAF all-objects HRNet | `AO_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar`, 861,530,273 bytes, SHA-256 `a2232cc15d5849d75c79c35af9b8b72cbd512766b880851c9c72493a36c79dbb` (published MD5 `fba3667a886b7b55d6e745f836951919`) |

Place the ADAF checkout at `work/adaf_head_to_head_2026-09-19/vendor/adaf/` (the adapter loads the wheel from
`vendor/adaf/installation/` by relative path) and the two `.tar` files at the paths the code names
(`work/adaf_head_to_head_2026-09-19/weights/` for all-objects, `work/campaign_2026-09-22/adaf_weights/` for barrow).

## Running the code

### Environment

The runs used Python 3.12.14 on Linux x86_64, an NVIDIA driver that supports CUDA 13, and the exact package set in
`work/colab_assessment_2026-09-29/env/freeze.txt`, installed without dependency resolution
(`work/colab_assessment_2026-09-29/env/setup_env.sh`). `requirements-lock.txt` holds the same 56 versions plus pytest and
its three dependencies; every line is a public PyPI pin.

```bash
python3.12 -m venv .venv                 # or: uv venv .venv --python 3.12.14
. .venv/bin/activate
pip install --no-deps -r requirements-lock.txt
python -c "import torch, rvt, rasterio, shapely; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

- **torch:** the PyPI wheel `torch==2.14.0` for Linux is the CUDA 13.0 build; its declared CUDA dependencies are the
  `cuda-toolkit==13.0.3.0` and `nvidia-*-cu13` pins in the lock file. The reproduction environment reports
  `2.14.0+cu130`, the same build from the PyTorch wheel index. CPU-only scoring needs only numpy, scipy, shapely and
  networkx.
- **rvt-py 2.2.1 is required:** `work/adaf_head_to_head_2026-09-19/adapter.py` imports `rvt` at module level, so
  inference fails with `No module named 'rvt'` without it.
- **AiTLAS is not pip-installed:** `adapter.py` reads the model definitions directly from
  `installation/aitlas-0.0.1-py3-none-any.whl` inside the ADAF checkout (see "Upstream ADAF"), so the wheel only has to
  be present at that path.
- An install from `requirements-lock.txt` into a fresh venv has not been tested; the reproduction check used an existing
  environment with the same 56 versions.

Run every script from the repository root.

### Runnable working copy

The released files are byte-identical to the original runs, so a few still name the original machine or the remote
GPU VM (decision D5). Make a separate runnable working copy instead of editing them:

```bash
python release_paths.py ../ndsbarrow-runnable          # verifies ALLOWLIST.txt, copies, applies PATH_SHIM.patch
cd ../ndsbarrow-runnable
export NDSBARROW_ROOT="$PWD" NDSBARROW_PY="$(command -v python)"
export NDSBARROW_VM_BASE=...                            # only for the remote-VM training/inference drivers
```

`python release_paths.py OUT_DIR --check` verifies and test-applies the patch without writing. The patch changes 32
files: run-instruction docstrings, the remote-VM drivers (`driver_d1.sh`, `vm_job_d1.sh`, `setup_env.sh`, Colab CLI
helpers), module-level VM path constants, and one line of `infer_irl.py` that makes a relative `WIN_DIR` absolute
(see the notes below). Scorers read their inputs by repository-relative paths and need no other change.

Two modules are withheld because they identify sealed English test terrain (`verify/astra_v2_labels.py`,
`verify/v3/v3_fetch_england.py`; `WITHHELD.md`), but released code imports them at module level (for example
`score_v0.py` -> `region_score.py` -> `score_holdout.py` -> `r2_masks.py`). `release_paths.py` therefore places the
fail-closed replacements from `release_shims/` at those paths in the working copy: imports succeed, the label-term rules
are kept, and any use of the sealed squares or the English fetcher raises. The German inference, training and scoring
paths do not use them; English mask building and English holdout scoring cannot run from this release.

### Regenerating proposals from public DGM1 and the released weights (Verden example)

```bash
P=outputs/fresh_region_2026-10-03
# 1. stage the 0.5 m relief windows from the pinned LGLN DGM1 tiles (downloads; checks bytes and SHA-256 per tile)
"$NDSBARROW_PY" -B work/fresh_region_2026-10-03/step2/lgln_reader.py stage \
  $P/step2/STAGING_PLAN_PRIVATE.json $P/step0/TILE_INDEX_PRIVATE.json $P/step0/TILE_LEDGER_PRIVATE.jsonl \
  $P/step2/INSIDE_MASKS_PRIVATE.npz WIN_DIR FETCH_LOG.jsonl \
  --manifest main=$P/step2/INFER_MANIFEST_PRIVATE.json --manifest enclosure=$P/step2/ENCLOSURE_MANIFEST_PRIVATE.json
"$NDSBARROW_PY" -B work/fresh_region_2026-10-03/step2/lgln_reader.py verify \
  $P/step2/STAGING_PLAN_PRIVATE.json $P/step0/TILE_INDEX_PRIVATE.json $P/step0/TILE_LEDGER_PRIVATE.jsonl \
  WIN_DIR FETCH_LOG.jsonl STAGING_RECEIPT.json \
  --manifest main=$P/step2/INFER_MANIFEST_PRIVATE.json --manifest enclosure=$P/step2/ENCLOSURE_MANIFEST_PRIVATE.json
# 2. inference + decoder for any checkpoint (NAME=PATH@SHA256; SHA-256 values are in the weights record)
"$NDSBARROW_PY" -B work/hr3_ireland_2026-10-02/step2/chain/infer_irl.py run \
  $P/step2/INFER_MANIFEST_PRIVATE.json WIN_DIR OUT --model D1F1_BARROW_L_2026092805=PATH/final.pt@SHA256
# 3. assemble the per-window outputs into PROPOSALS_<NAME>_PRIVATE.json
"$NDSBARROW_PY" -B work/hr3_ireland_2026-10-02/step2/chain/infer_irl.py assemble \
  $P/step2/INFER_MANIFEST_PRIVATE.json OUT D1F1_BARROW_L_2026092805 DEST_DIR
```

The D1 chain ran exactly these steps (`work/d1_collapse_2026-10-06/chain/chain_lib_d1.sh`, functions `do_stage_dev`
and `lgln_pass`). Oldenburg uses `work/adapt_nds_2026-10-05/nds_reader.py stage confirm` with the
`outputs/adapt_nds_2026-10-05/confirm/run/` plan, masks and manifest. `score_v0.py` then reads the proposal files at
the run paths it names (for D1: `outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z/v1/results/dev/`;
ADAF barrow: `outputs/fresh_region_2026-10-03/step2/score/individual/`) and checks them against the session receipt
`outputs/d1_collapse_2026-10-06/chain_runs/run_s3_20261006T222546Z/SESSION_VERIFIED.json`, which pins the SHA-256 of
the original proposal files. As frozen, `score_v0.py` refuses to score unless every proposal file matches that pin
byte-for-byte and covers all 717 Verden tiles. To score regenerated proposals that are not byte-identical, or a subset
of tiles, use `tools/score_subset.py` (next section), which applies the functions of the independent scorer
`work/independent_rescore_2026-10-09/rescore.py` to any proposal file and tile list; compare matches and recall, not
file hashes. Do not run `rescore.py` itself on regenerated proposals: its `main()` reads the original runs' proposal
files (D1, Stage 2 controls, ADAF barrow, Oldenburg) at fixed paths, and those files are not released.

Notes on step 2 (from the reproduction check below):

- **`WIN_DIR` must be an absolute path under the repository root when the released `infer_irl.py` is run directly.**
  A relative `WIN_DIR` passes the under-root check in `infer_irl.py` but then fails in
  `work/tta_ensemble_2026-09-24/tta_run.py` (`path.relative_to(ROOT)` with an absolute `ROOT`: `ValueError ... is not
  in the subpath of ...`). In the runnable working copy, `PATH_SHIM.patch` changes one line of `infer_irl.py` to
  `Path(WIN_DIR).absolute()`, which resolves a relative `WIN_DIR` against the current directory (the repository root)
  without following symlinks; a relative `WIN_DIR` then behaves exactly like its absolute form.
- **The window path is recorded in every proposal.** Each proposal's `source` field is the window file's path relative
  to the repository root. To regenerate Verden proposals whose `source` fields equal the original run's, stage the
  windows to and run with `WIN_DIR=$NDSBARROW_ROOT/outputs/d1_collapse_2026-10-06/vm/dev_windows` (the original
  path). Another location gives the same detections with a different `source` string.
- **Weights needed:** inference with a released checkpoint (`--model NAME=PATH@SHA256`) needs the ADAF code checkout
  and its AiTLAS wheel (model definitions), but not the ADAF `.tar` weights; those are read only when the ADAF models
  themselves are run or used as a training start.
- **TF32:** the original runs set `NVIDIA_TF32_OVERRIDE=0`, and `irl_common.torch_flags()` also turns TF32 off inside
  PyTorch. Set `export NVIDIA_TF32_OVERRIDE=0` to match.
- **Different GPUs give float32 rounding differences only.** Measured in the reproduction check (original run on an
  NVIDIA RTX PRO 6000 Blackwell, rerun on an RTX 2060): `score` differed by at most 2.4e-7 and `mean_probability` by
  at most 7.2e-7, with identical proposal ids, positions, areas, ranking order and matches; a repeat on the same GPU was
  bit-identical. Proposals whose sort keys tie within about 1e-6 at a budget cut could swap order on other hardware;
  none did in the check.

### Running and scoring a subset of tiles

`infer_irl.py run` and `assemble` process every window of the manifest they are given and fail closed when one is
missing. For a few tiles, stage only their windows (`lgln_reader.py stage` and `verify` both take `--only WINDOW_ID
...`), write a subset manifest with `tools/subset_manifest.py`, run steps 2-3 on it, and score with
`tools/score_subset.py`. Window ids are the tile id plus `-w0` (one window per Verden tile).

```bash
P=outputs/fresh_region_2026-10-03
IDS="VER-P-000-w0 VER-P-007-w0"
"$NDSBARROW_PY" -B work/fresh_region_2026-10-03/step2/lgln_reader.py stage \
  $P/step2/STAGING_PLAN_PRIVATE.json $P/step0/TILE_INDEX_PRIVATE.json $P/step0/TILE_LEDGER_PRIVATE.jsonl \
  $P/step2/INSIDE_MASKS_PRIVATE.npz WIN_DIR FETCH_LOG.jsonl \
  --manifest main=$P/step2/INFER_MANIFEST_PRIVATE.json --manifest enclosure=$P/step2/ENCLOSURE_MANIFEST_PRIVATE.json \
  --only $IDS
"$NDSBARROW_PY" -B work/fresh_region_2026-10-03/step2/lgln_reader.py verify \
  $P/step2/STAGING_PLAN_PRIVATE.json $P/step0/TILE_INDEX_PRIVATE.json $P/step0/TILE_LEDGER_PRIVATE.jsonl \
  WIN_DIR FETCH_LOG.jsonl STAGING_RECEIPT.json \
  --manifest main=$P/step2/INFER_MANIFEST_PRIVATE.json --manifest enclosure=$P/step2/ENCLOSURE_MANIFEST_PRIVATE.json \
  --only $IDS
"$NDSBARROW_PY" -B tools/subset_manifest.py $P/step2/INFER_MANIFEST_PRIVATE.json SUBSET_MANIFEST.json --ids $IDS
"$NDSBARROW_PY" -B work/hr3_ireland_2026-10-02/step2/chain/infer_irl.py run \
  SUBSET_MANIFEST.json WIN_DIR OUT --model D1F1_BARROW_L_2026092805=PATH/final.pt@SHA256
"$NDSBARROW_PY" -B work/hr3_ireland_2026-10-02/step2/chain/infer_irl.py assemble \
  SUBSET_MANIFEST.json OUT D1F1_BARROW_L_2026092805 DEST_DIR
"$NDSBARROW_PY" -B tools/score_subset.py --manifest $P/step0/INDIVIDUAL_MANIFEST_PRIVATE.json \
  --proposals regen=DEST_DIR/PROPOSALS_D1F1_BARROW_L_2026092805_PRIVATE.json \
  --tiles VER-P-000 VER-P-007 --out SCORE.json
```

`tools/score_subset.py` takes one or more `LABEL=PATH` proposal files, a tile list (`--tiles` or `--tiles-file`;
default: every manifest tile) and budgets (`--budgets`, default 0.5 1 2 proposals per km2). It pools the in-core
proposals of the listed tiles in RAW order, sets K = floor(b x core area of those tiles), and matches against every
entity of the manifest, reporting pool matches, K, matches and matches per tile (aggregates only). Without `--tiles` it
reproduces the RAW row of `rescore.py` for that file (164 / 274 / 369 at b = 0.5 / 1 / 2 for the original D1 seed
2026092805 proposals).

## Reproduction check

On 2026-10-09 the release candidate was cloned locally and run from a `release_paths.py` working copy. Verden
proposals were regenerated for 6 of the 717 tiles (62 proposals) with one released checkpoint
(`D1F1_BARROW_L_2026092805`, identified by SHA-256), from public LGLN DGM1 fetched through the staging route above (29
DGM1 tiles; every byte count and SHA-256 equal to the plan pins; the 6 relief windows byte-identical to the original
run's).

- The regenerated proposals have the same ids, positions, areas, roundness and ranking order as the original saved
  proposals. `score` differs by at most 2.4e-7 and `mean_probability` by at most 7.2e-7 (float32 rounding on a
  different GPU model); a repeat on the same GPU was bit-identical.
- Matches are equal: 2 / 5 / 10 at b = 0.5 / 1 / 2 on the 6-tile subset (K = 2 / 5 / 10), and 164 / 274 / 369 with
  the 6 tiles swapped into the full 717-tile region, the values in the released `RESCORE_RAW.json` for this seed.
- The usability defects it found are fixed in this version: relative `WIN_DIR` (path shim and note), the window path
  recorded in `source`, no subset tooling (`tools/subset_manifest.py`, `--only`), no runnable scorer for regenerated
  proposals (`tools/score_subset.py`), an underspecified install (`requirements-lock.txt`), and the TF32 and ADAF-weights
  notes. The subset scoring was then rerun with the shipped `tools/score_subset.py` from a fresh working copy and gave
  2 / 5 / 10 for both the original and the regenerated proposals.

Limitations: one checkpoint and seed; 6 tiles, chosen by a fixed rule among tiles with at least 2 matches and with
small downloads, and no background tiles; the subset budgets are tiny (K = 2 / 5 / 10), so the pool matching and the
full-region swap-in are the more informative checks. An existing Python environment with the pinned versions was
used, so a fresh install from `requirements-lock.txt` was not tested. The checkpoint came from a local copy with its recorded
SHA-256, not from the Zenodo record. The relative-`WIN_DIR` shim was checked by applying the patch and by
path logic, not by a second GPU run.

## Licences

- Code written for this project: Apache License 2.0 (`LICENSE`).
- ADAF code and the AiTLAS package (not included): Apache License 2.0, © 2024 ZRC SAZU (`NOTICE`).
- Model weights (Zenodo): CC BY-SA 4.0, as derivatives of the ADAF weights.
- Entity manifests derived from the Denkmalatlas Niedersachsen: CC BY-SA 4.0 with attribution (`NOTICE`).

Cite this repository with `CITATION.cff`.
