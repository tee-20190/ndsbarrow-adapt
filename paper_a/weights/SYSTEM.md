# Paper A detection systems: exact configuration

This file describes how the deposited checkpoints are used in the paper "Cross-border recovery of registered burial
mounds by an adapted public LiDAR detector at fixed review workloads" (Shanahan, in preparation). Each item names the
frozen protocol section that fixed it and the code file that implements it. `CODE_PINS.json` gives each file's SHA-256.

## 1. Network

- **Architecture.** ADAF's HRNet semantic-segmentation model for the barrow class (Čož et al. 2026; ADAF repository
  `EarthObservation/adaf` at commit `a44acdd678773666458ea50f544b5d2292bcef51`; AiTLAS wheel SHA-256
  `be2dbb7e3c39c30a91a79b00f7dc3365320cbc57584a8601e286a8b9c91ab69a`). Two output channels; channel 1 (sigmoid) is the
  barrow probability.
- **Start.** Every deposited checkpoint descends from the public ADAF barrow checkpoint
  `barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar`, SHA-256
  `51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216` (Zenodo 15848663). The checkpoints record it in
  their `start` field.
- **File format.** PyTorch zip files. Top-level keys: `state_dict` (1,964 tensors; no optimiser state) plus plain
  labels (`epoch`, `seed`, `arm`, `recipe`, `start`; D-recipe files add `dt1_arm`, `channels`, `experiment`). Load with
  `torch.load(path, map_location="cpu", weights_only=True)` and a strict `load_state_dict`. Tensors were saved on
  `cuda:0`, so pass `map_location`. SSL encoder files hold `encoder` (stem + backbone only).

## 2. Inputs

**Legacy three-channel route** (D1, A_k, EB, CONT, ADAF). Simple local relief model (SLRM) on a 0.5 m view, radius
20 px (10 m), normalised as clip(SLRM, −0.5 m, +0.5 m) + 0.5 and repeated over three channels, exactly as ADAF
prepares it (`adapter.py`, `adaf_configs.py`; training: `adapt_train.relief_v`, non-finite input → 0.5 and its label
IGNORE; inference: non-finite → 0). Dutch AHN4
tiles are read at their native 0.5 m. In Denmark the legacy route was confirmed bitwise identical to the earlier chain
on two development cores (CONFIRM-D FREEZE §3).

**Four-channel adapter** (D_k, B, G, C_k, FAINT-LEARN-2; DENSE-TERRAIN-1 FREEZE §3, `dt1_channels.py`,
`dt1_model.py`):

    X = [ℓ, asinh(r₂ / 0.1 m), asinh(r₃₂ / 1 m), v]

- ℓ is the legacy SLRM channel above, unchanged.
- r_σ = z − (G_σ ∗ (z·m)) / (G_σ ∗ m) for σ = 2 m and σ = 32 m, on the native 1 m grid: a separable Gaussian
  truncated at 3σ, with m the validity mask.
- v is the validity mask. Where v = 0 the two residual channels are 0.
- Terrain is read with a 97 m halo (96 m filter support plus one cell).
- **Stem.** ADAF's `backbone.conv1` (3 → 64, 3 × 3, stride 2, no bias) becomes 4 → 64. The original three input
  filters are summed into the first channel; the new channels start at zero (FREEZE §3.3).

## 3. Decoder (all systems)

ADAF-faithful decoder of the earlier preprint (CR-1 §2.4; `dt1_decode.py`, `ot_infer.py`, `tta_run.py`):

- 1,024 px tiles with a 64 px halo (1,152 px views), pixelwise maximum where views overlap;
- probability threshold 0.5;
- connected objects with area above 30 m² and roundness above 0.4;
- no non-maximum suppression; a proposal belongs to the core containing its centroid;
- **RAW ranking:** maximum probability, then mean probability, then area (`rank_key`).

## 4. Systems

- **D1:** checkpoint `D1-F1_BARROW_L_2026092805` alone, RAW ranking.
- **S_D (MERGE-1):** members D_101 (`DT1D-F1_BARROW_L_101`) and D_202 (`DT1D-F1_BARROW_L_202`). Per core, each
  member's list is taken in `rank_key` order and the lists are interleaved with D_101 first. A proposal within 15 m of
  one already kept is dropped (`merge1.py`, SHA-256 `ee4074b4…`; CONFIRM-D FREEZE §3).
- **S_D1:** the same merge of D1_05 and D1_06.
- **PROX** at budget B per core (`cr1_score.py`): keep the top ⌈0.7 B⌉ proposals; fill the remaining slots with the
  best-ranked further proposals within R = 200 m of a kept proposal; if fewer remain, add the next-ranked. PROX uses
  no register information.
- **Danish strict map mask** (all systems equally): building footprints, major roads, roundabouts, quarries and water
  from a frozen OpenStreetMap snapshot per core (`strict_mask.py`).

## 5. Training recipes

**F1 (D1 recipe; `d1_train.py`, `adapt_train.py`; preprint v1 §4.2):**
- start: ADAF barrow;
- data: 852 positive and 852 background 423 m windows from the Lower Saxony training districts (Uelzen, Harburg,
  Heidekreis) with Denkmalatlas Niedersachsen barrow masks;
- loss: binary cross-entropy everywhere, plus soft Dice only on windows with a non-empty target;
- optimiser: fresh AdamW, learning rate 1e-4, weight decay 1e-4 on every parameter, cosine decay to 1e-6;
- 5,116 updates, batch 1, alternating positive and background windows, random crop and dihedral transform;
- BatchNorm frozen; fp32 with TF32 disabled; `torch.manual_seed(seed)` after the model is built; final weights only.

**D recipe (DENSE-TERRAIN-1 FREEZE §5; `dt1_ssl.py`, then `dt1_train.py --arm D`):**
1. Dense self-supervised pretraining of the four-channel encoder, 4,096 updates, on the same 1,704 FIT windows:
   - teacher–student cross-view objective at 256 matched points per window;
   - two 256 m (512 px) views per window, 8 windows per update;
   - projector 976 → 2048 → 2048 → 256; student temperature 0.10; teacher 0.04 → 0.07;
   - EMA momentum cosine 0.996 → 1.0; AdamW (weight decay 0.04), 128-update warm-up to 1e-4, cosine to 1e-6;
   - gradient-norm clip 3.
2. The final EMA-teacher stem and backbone, with the unchanged ADAF segmentation head, are fine-tuned with F1 as above.

Arms B and G (development only) are four-channel F1 without SSL, and four-channel F1 after pooled rather than dense SSL.

**EB (ENG-ADAPT-1; `ea_train.py`):** F1 with the same start, update count and number of positive units, using English
scheduled barrows (Historic England) and Environment Agency LiDAR in place of Lower Saxony data.

**C_k (DK-REGSUP-1 controls):** the D recipe with new seeds 1010–1515; Lower Saxony data only.

## 6. Environment of the reference decodes

CPython 3.12.14 and torch 2.14.0 for CUDA 13.0, with the exact pins in `requirements-sd-prox.txt`. TF32 is disabled
for decoding. On another GPU, scores differ in float32 rounding: the D1 Verden reproduction below differed by at most
2.4e-7 in score and 7.2e-7 in mean probability, with identical proposals and matches.
