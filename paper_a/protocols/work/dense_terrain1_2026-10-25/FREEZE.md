# DENSE-TERRAIN-1 (DT1) — FREEZE for arms A, B, G and D

**Status: frozen at the commit that adds this file, before any DT1 training, inference or score.** Capability line:
sensitivity to faint forms and transferable terrain representation (Pro's dense-terrain bet), measured as reproducible
detector improvement on exposed Lower Saxony development terrain.

**Sources.** Pro's proposal `work/dense_terrain1_review_2026-10-24_codex/PRO_SPEC.txt` as amended by Codex's
`HANDOFF.md` (same folder), the coordinator's `work/coord_assessment_2026-10-24/CODEX_THREAD_01a16ef7_ASSESSMENT.md`, and
the corrections of `work/post_rerank_review_2026-10-25_codex/REVIEW.md`. Where they differ, this file decides; every
departure from PRO_SPEC is listed in §15.

**Authority.** Owner standing execution authority (AGENTS.md, 2026-09-24): bounded Colab training and inference of an
eligible, frozen experiment, licensed public-data downloads and local evaluation are ordinary project work. No separate
owner review is claimed. This freeze fixes definitions; code is pinned by addenda before each launch (§14). An addendum may
pin code and record measured facts. It may not change a definition here.

**Single builder.** This session is the only writer of `work/dense_terrain1_2026-10-25/` and
`outputs/dense_terrain1_2026-10-25_PRIVATE/`. Before this freeze, no other DT1 implementation existed under `work/` or
`outputs/` (checked 2026-10-25 ~03:50Z: only the Codex review/handoff folders).

## 1. Questions and contrasts

| Contrast | Question | Role |
|---|---|---|
| **B − A** | Does the four-channel input (legacy + two signed residual scales + validity) help ordinary F1 supervision? | input-preservation contrast |
| **D − A, D − B, D − G** | Does dense correspondence SSL at matched points add beyond the matched legacy arm, the rich-input arm and the matched pooled SSL control? | dense-learning contrasts (all three required) |
| **G − B** | Does pooled SSL on the same points help or harm? | control |
| **D − R_O** (Δ_op) | Does dense learning beat the operating detectors at the same workload? | operational ambition |

**B is a package intervention** (REVIEW.md correction 2). A B − A difference is credited to the four-channel package
only. It is not attributed to one channel, and no promise is made that it appears only on large records. Coordinator
prediction S1, recorded here before any score as a prediction and not as a rule: if B > A, the gain shows on large or
strong records (legacy saturation) and/or near data gaps (validity), and on faint records only through the
asinh(r₂/0.1 m) contrast gain. §9 reports the strata that test it, descriptively.

## 2. Arms

| Arm | Input | Pre-training | Fine-tune |
|---|---|---|---|
| **A** | legacy 3-channel repeat (unchanged ADAF/D1 path) | none (public ADAF barrow start) | F1 |
| **B** | four channels (§3) | none (ADAF barrow start, summed-weight stem) | F1 |
| **G** | four channels | pooled SSL at the shared point multiset (§5) | F1 |
| **D** | four channels | dense SSL at the same points (§5) | F1 |

- **Start:** the public ADAF barrow tar (`51b679fa…`, `match_train.start_weights("BARROW_L")`, size and sha256
  checked), official AiTLAS HRNet (source sha256 `4fd87062…75ce`), strict state dict.
- **Seeds:** 101, 202 and 303. Chain k of every arm uses seed k. Pair k is (A_k, B_k, G_k, D_k).
- **Head, stitcher, decoder and proposal operations** are unchanged for every arm (§7).

## 3. Inputs

### 3.1 Legacy channel ℓ (all arms; bytes unchanged)

- **Training:** exactly D1's path: `adapt_train.relief_v` (bilinear 0.5 m view of the 1 m window, SLRM radius 20 px,
  `normalize_image('slrm', −0.5, 0.5, 'value')` = clip(slrm, −0.5, 0.5) + 0.5) with void policy V (`apply_v`: a
  non-finite input becomes 0.5 and its label IGNORE), then `r2_train.crop` and `r2_train.dihedral`.
- **Inference:** exactly the frozen decoder's per-tile chain (`adaf_configs.tile_probabilities` /
  `tta_run.tile_probabilities_multi`): same SLRM and normalisation, NaN → 0.
- Arm A feeds `[ℓ, ℓ, ℓ]` (`r2_train.to_input`). B, G and D feed ℓ as channel 0.

### 3.2 Residual and validity channels (B, G, D)

On the **native 1 m source grid**, with m = 1 where the source cell is valid (finite, not NoData) and 0 elsewhere, and a
separable Gaussian G_σ truncated at 3σ (radius 6 and 96 cells; `scipy.ndimage.gaussian_filter`, `truncate=3.0`,
`mode="constant"`, cval 0, float64):

- w_σ = G_σ ∗ m (retained kernel weight; the kernel sums to 1);
- r_σ = z − (G_σ ∗ (z·m)) / w_σ, for σ = 2 m and σ = 32 m;
- s = 1 iff m = 1 and w₂ ≥ 0.99 and w₃₂ ≥ 0.99 (the 99 % support rule), else 0.

**Rendering:** r₂, r₃₂ and s are rendered to the 0.5 m view by the same 2× bilinear geometry as `adapter.source_view`
(GDAL bilinear, output pixel p at source index p/2 − 0.25; separable weights 0.75/0.25). Before rendering, r is set to 0
where s = 0. The four channels are then

> X = [ℓ, asinh(r₂ / 0.1 m), asinh(r₃₂ / 1 m), v]

with **v = 1 iff every native cell in the pixel's bilinear stencil has s = 1 and the legacy source value at the pixel is
finite**; where v = 0 the two residual channels are 0. ℓ is never changed by v. In training the same crop and dihedral
apply to all four channels.

- **Halo:** a 97 m read halo (96 m filter support + 1 source cell) around every training window and every evaluation
  read footprint.
- **F1 targets, IGNORE masks and evaluation denominators are not changed by v.**

### 3.3 Four-channel stem (B, G, D)

The stem convolution `backbone.conv1` (3 → 64, 3 × 3, stride 2, no bias) is replaced by a 4 → 64 convolution with
weight[:, 0] = Σ_c W[:, c] (the original three input-channel weights summed) and weight[:, 1:4] = 0. Every other module is
the ADAF tensor. At initialisation B therefore equals A on legacy input up to floating-point order. **Qualification
(Addendum P):** FP32 logit parity between A on `[ℓ, ℓ, ℓ]` and B on `[ℓ, x₁, x₂, x₃]` (arbitrary x) at 512 px and at
1,152 px on real windows: max |Δ logit| ≤ 1e-3 (the decoder's equivalence tolerance), actual value reported.

### 3.4 Source and halo eligibility (before any terrain is read)

- **FIT/SSL windows:** D1's 1,704 FIT windows (`FIT_TABLE_PRIVATE.json` `58d6aa16…`). Each window's halo footprint is
  the 423 m window + 97 m. It is read from the LGLN DGM1 records meeting it under D1's tile rule (pixel centres outside
  the FIT union are NoData; tiles pasted in name order, first valid value wins). The halo staging manifest (tile names,
  URLs, byte and md5 pins) is pinned by Addendum H before staging. The legacy 423 m windows written from the same fetched
  tiles must pair **1,704/1,704 byte-equal** with D1's `LS_RENDER_LOG.json`; otherwise stop.
- **Evaluation cores:** the read footprint of a core is the core + 129 m (32 m decoder tile halo + 97 m). It must lie
  inside the core's pinned CR-1 §2.4 / SUBTLE-1 stage-B mosaic (core + 256 m). A square without an LGLN tile stays NoData
  for every arm (v = 0), as for A. **Selection** (`dt1_select.py`, `SELECTION_COUNTS.json`, private
  `SELECTION_PRIVATE.json` `568f1c3a…`): **210 of 210 cores eligible** (0 outside the mosaic; 0 within any FIT window's
  halo footprint, minimum gap 2,249 m; 0 LS-POOL-2 exposed cores; 1 core has one missing halo tile). Budget 10,496.
- **No new evaluation terrain is fetched.** The mosaics are rebuilt from stage B's recorded tile choices and must be
  byte-identical to `PREP.json` (`PARITY` as in CR-1 eval_infer phase 1a). The 129 m read strip may lie in a neighbouring
  core; it was read by CR-1 and stage B, and no output is kept outside the 210 cores.
- No sealed or reserved terrain. No pool core of `work/ls_pool2_2026-10-25/EXPOSED_CORES.json` or of the 738-core pool
  is read beyond those existing halo strips.

## 4. F1 (all arms)

- **Recipe F1 unchanged** (`work/d1_collapse_2026-10-06/d1_train.update`): BCE + 1 − soft Dice on windows with a positive
  target, BCE only otherwise; fresh AdamW lr 1e-4, wd 1e-4 on every parameter; cosine 1e-4 → 1e-6 over **5,116
  updates**; batch 1; BatchNorm frozen (`r2_train.set_train_mode`); fp32, TF32 off; `torch.manual_seed(seed)` after the
  model is built; final weights only. D1's online collapse detector records and never intervenes.
- **Plan:** `match_train.schedule(seed, epoch, ls_context)` (D1's all-local plan: alternating positive and background
  windows, ±160 px offsets, dihedral). Its per-epoch draw digests are hashed once per seed and must be **identical across
  A, B, G and D** of that seed.
- **Start of G and D:** the final EMA-teacher encoder and stem from §5, with the unchanged ADAF segmentation head.
- **Environment gate on each training VM:** D1's step-0 rule (|loss − 1.0099719719146378| ≤ 1e-6, replay bit-equal),
  as CR-1.
- **B/101 is the F1 throughput benchmark** and is retained as one of B's three scientific runs.

## 5. SSL for G and D

Shared plumbing (D4R's teacher/student pattern, re-implemented in the DT1 trainer): student and teacher are the
four-channel encoder (stem + backbone, BatchNorm running statistics frozen at ADAF values; student BN affine parameters
learn, teacher's follow the EMA) with a projector. The segmentation head is not used.

- **Data:** the 1,704 FIT windows (§3.4). Each optimiser update uses **8 distinct windows** (four microbatches of two),
  drawn by a seeded permutation of the 1,704 cycling without replacement.
- **Views:** two 256 m (512 px) crops per window, each centred at the window centre + an independent integer-metre offset
  in [−32, 32] m per axis, each with an independent dihedral k ∈ 0..7. No resizing, amplitude change, polarity inversion,
  blur or noise.
- **Points (Codex amendment 1):** the **identical 256-point multiset** serves G and D. Candidates are the window's 0.5 m
  pixel-centre lattice points valid in both views (v = 1) and ≥ 32 m from both crop boundaries. 128 are uniform (without
  replacement; with replacement only if fewer than 128 exist); 128 are balanced over five FIT-fixed quintiles of
  e = √((r₂/0.1 m)² + (r₃₂/1 m)²) (26, 26, 26, 25, 25; with replacement within a stratum when needed; an empty
  stratum's allocation goes to the non-empty strata in order). The quintile edges are computed once, before SSL, from
  every v = 1 pixel at stride 8 px of the central unshifted 256 m view of all 1,704 windows, and hashed.
- **Features:** at a point p, f[i, v, p] = concatenation of layer1 (256) and the four stage-4 branches (48, 96, 192, 384):
  **976 dimensions**, each branch sampled bilinearly at its own world-to-feature coordinate (§6).
- **Objectives:** G[i, v] = projector(mean_p f[i, v, p]); D[i, v, p] = projector(f[i, v, p]). Both use the same points,
  mask, margin and sampler.
- **Projector:** 976 → 2048 → 2048 → 256, GELU, no BatchNorm; L2-normalised bottleneck; weight-normalised prototype
  layer 256 → 4,096 with gain fixed at 1; prototype directions frozen for the first 128 updates.
- **Loss:** symmetric cross-view cross-entropy H(softmax((t − c)/τ_t), log-softmax(s/τ_s)) (teacher of view 1 → student
  of view 2 and vice versa; for D, at the same physical point). D averages over points within an image, then over
  images. All softmax, centring and cross-entropy in float32.
- **Centre** (EMA 0.9): G uses, per image, the mean of its two pooled-view teacher logits; D uses, per image, the mean over
  points and views; both then average over the 8 images and update once per optimiser step.
- **Temperatures:** student 0.10; teacher 0.04 → 0.07 linearly over 512 updates, then 0.07.
- **Teacher:** an exact student copy at start; EMA momentum cosine 0.996 → 1.0 over the chosen horizon.
- **Optimiser:** AdamW, weight decay 0.04 excluding biases and normalisation parameters; lr warm-up over 128 updates to
  1e-4, then cosine to 1e-6; gradient-norm clip 3. Accumulate all 8 windows before the optimiser, centre and EMA updates.
- **Horizon:** U = **4,096 updates** (fallback 2,048, §11). Three independent chains per arm (seeds 101, 202, 303). G_k
  and D_k share seed k's window/view/point manifest (hashed) and initial projector weights.
- **Transfer:** the final EMA teacher encoder and stem into F1 (§4). Final checkpoints only; no development-selected
  checkpoint.
- **Benchmarks:** one disposable 200-update G run and one 200-update D run (seed 101 manifest), discarded afterwards; they
  are not production starts.
- **Hard failures:** non-finite training, coordinate test failure, changed targets, unavailable eligible source support,
  failed teacher/centre updates, insufficient verified storage. A collapse warning (teacher entropy at its maximum or
  minimum for 256 consecutive updates) is inspected against the initial representation; a low early score is not a
  stopping rule.

## 6. Coordinates and sampling conventions

- **World → view index:** for a view with 1 m source origin (x₀, y₁), the 0.5 m pixel (r, c) has its centre at
  x = x₀ + 0.25 + 0.5 c, y = y₁ − 0.25 − 0.5 r. Training crops use `r2_train.crop`'s offsets; the dihedral k applies
  `np.rot90(·, k % 4)` then a column flip if k ≥ 4, to continuous centre-index coordinates
  ((r, c) → (n − 1 − c, r) per quarter turn; (r, c) → (r, n − 1 − c) for the flip).
- **View index → branch feature index:** for a branch of stride s ∈ {4, 8, 16, 32} (layer1 and stage-4 branch 0: 4;
  branches 1-3: 8, 16, 32), u = r / s, w = c / s (each stride-2 3 × 3 convolution with padding 1 centres output i on input
  2i). Sampling is bilinear in feature-index coordinates (`grid_sample`, `align_corners=True`). The head's own
  nearest-neighbour resize to 120 × 120 is not used by the sampler.
- **Qualification (Addendum P):** traced shapes at 512 and 1,152 px; coordinate round trips and sampling of synthetic
  world ramps through crop, all eight dihedral transforms and each branch's lattice within **10⁻⁴ feature cell**; the
  gradient-centroid offset of real branch cells reported (descriptive). The untrained network is not required to be
  equivariant.
- **Numerical conventions (signed-shape readout, §10):** Gaussian aspect ratio = major σ / minor σ, the listed width is the
  minor σ; the perturbation δ is evaluated analytically on the native grid (no truncation); the target smoothing is
  G_{σ = 1 m} truncated at 3σ on the native grid; offsets p ± r·u_θ are interpolated bilinearly on the native grid; a
  target is computable only if every native cell of its smoothing and interpolation support is valid; unperturbed
  descriptive DEV extraction uses phase (0, 0).

## 7. Evaluation

- **Terrain:** the 210 CR-1 evaluation cores (§3.4), **exposed development terrain** (third exposure, after CR-1 and
  SUBTLE-1 stage B). Every DT1 result there is a development result.
- **Truth and strata:** CR-1 §2.2 unchanged (`EVAL_TRUTH_PRIVATE.json` `fe9575b9…`, 4,381 records; E0's H and T;
  `cr1_analysis.strata`); small/large = register polygon area < 200 m² / ≥ 200 m² (SUBTLE-1).
- **Decoder:** CR-1 §2.4 unchanged (1,024 px tiles, 64 px halo, i.e. 1,152 px views; sigmoid channel 1; pixelwise max;
  threshold 0.5; area > 30 m², roundness > 0.4; no NMS; object kept iff its centroid is in its core). Arm A and the
  comparators go through the frozen route; B, G and D through an experiment-owned four-channel adapter whose legacy channel
  is the frozen chain's array. **Parity gate before evaluation-core output:** the adapter must reproduce the frozen
  route's maps bitwise for a 3-channel model on 2 cores, and D1 run 05's existing proposals on those cores.
- **Workload:** B = round(2 × land km²) per core, never pooled (208 × 50, 2 × 48, total 10,496); S-v1 strict mask from
  E0's snapshots; `scot_score.rank_key`; plain (top B) and PROX (`cr1_score.prox`, R 200 m, f 0.3, candidates the decoded
  objects beyond the budget); 15 m one-to-one maximum matching (`cr1_score.score`, unchanged). The extra incumbent-fixed
  budget cap of PRO_SPEC is **removed** (Codex amendment 3). Each model's actual kept count and underfilled cores are
  recorded; no arm is forced down to another's realised count.
- **Diagnostic workload 20/km²:** B₂₀ = round(20 × land km²) from the same decoded pool and threshold. This is
  availability in the unchanged pool; a miss there does not show the backbone lacks the signal.
- **Comparators:**
  - **R0:** D1 runs 05, 06, 07 (stage-B proposals) with `rank_key`, plain; **R0 + PROX**;
  - **D1 run 05** (`d9ece1fb…`) plain and + PROX (the operating policy);
  - **public ADAF barrow** (`51b679fa…`) plain and + PROX, decoded once by the frozen route.
  - PROX is included: its rule is pinned (CR-1 §2.6) and applicable. No new PROX fitting.
- **31 m placement check:** every DT1 checkpoint, D1 run 05 and ADAF are decoded again with the tiling phase shifted by
  31 m in both axes (`phase_m = 31`) on the preselected acquisition-stratified shift subset (42 cores; 18/13/4/7 by
  delivery year 2016/2017/2018–2022/2023–2025). The evaluated world footprint is unchanged.
- **Statistics:** `cr1_analysis.contrast` unchanged: paired seeds k; 95 % interval Δ ± 1.96 √(se²_core + var_seed),
  core-cluster ratio-estimator variance and var_seed = max(observed between-pair variance / K, (1.3 points)² × 2/3).
  Single-model comparators enter as K copies (var_seed at its floor). R0 is paired with DT1 seeds by index (101↔05,
  202↔06, 303↔07). Records, nearby cores and three predictions of one record are not independent; three seeds do not
  describe the whole seed distribution.

## 8. Decision rules (all at 2/km², all records, plain unless stated)

- **Operating comparator:** R_O = max{R(D1 run 05 plain), R(public ADAF plain), R(D1 run 05 + PROX)}. Δ_r = mean over
  the three D runs (single models, not an ensemble) of recall − R(r), for each of the three references r.
  **Δ_op = min_r Δ_r**, with a simultaneous interval [min_r(Δ_r − z₃ s_r), min_r(Δ_r + z₃ s_r)], z₃ = 2.394 (Bonferroni,
  three contrasts), s_r the §7 standard error. The three individual contrasts are reported too. The maximum is
  predeclared; its winner is not chosen after scoring.
- **Input preservation (B − A):** reported with its interval. "B > A shown" iff the lower bound > 0. Its point sign is
  the forecast event of §13.
- **Useful smaller gain:** Δ_op > 0 (point), reported with interval and seed spread. **Dense-learning credit** additionally
  needs paired D − A > 0, D − B > 0 and D − G > 0 (point means). A gain of B alone is input preservation; a gain of A
  alone is a training/seed result, not dense-learning credit.
- **Pilot ambition met:** Δ_op ≥ +5.0 points (point estimate) **and** paired D − A, D − B and D − G each with a positive
  mean and ≥ 2 of 3 chains positive **and** all three guards PASS.
- **Guards (−2-point non-inferiority):** D − R_O on large, strong-grouped and isolated records, with the same simultaneous
  min construction: **PASS** if the lower bound > −2 points, **HARM** if the upper bound < −2 points, otherwise
  **UNRESOLVED** (not proof of harm). The same three guards are reported for B − A (descriptive).
- **Interpretation labels:** implementation failure (the intervention did not run as specified); mechanism null (no
  evidence of improved signed-shape response over B/G, especially after F1; it limits the mechanism claim and does not
  reject a detector gain); operational null for the ambition (Δ_op's interval upper bound < +5); underpowered (the
  interval contains both 0 and +5).
- **Later demonstrator** (not this experiment): ≈ +10 points with lower bound > +5 against a frozen strong operating
  policy in a separately powered prospective comparison. Reordering D1's existing candidates is capped at **+5.77
  points** over R0 + PROX at this budget (REVIEW.md, certified maximum flow), so +10 needs better candidate availability
  or localisation, not ranking.

## 9. Descriptive reports (no decision)

For every arm and comparator, plain and PROX, at 2/km² and 20/km²: recall on all, small, large, grouped, faint
grouped, strong grouped and isolated records; per-acquisition-year recall; kept counts and underfilled cores; recall on
the shift subset at phase 0 and 31 m and their difference.

**B − A strata (S1 test, descriptive):** paired B − A and recall per arm by

- **actual legacy clipping:** a record is *clipped* if any 0.5 m pixel within 10 m of its centroid has |SLRM| ≥ 0.5 m
  (the legacy channel saturated at 0 or 1), else *unclipped*; computed from the frozen legacy chain on the rebuilt
  mosaic as 5 m cell counts (a record takes the cells whose centres lie within 12.5 m of its centroid);
- **validity:** *gap-near* if any rendered pixel within 32 m of the centroid has v = 0 (same 5 m cell method), else
  *full-support*;
- **size:** small / large;
- **relief:** H < T, T ≤ H < 2T, H ≥ 2T, unmeasured.

## 10. Signed-shape readout Q (instrument; no veto, no gate)

- **Target:** z̄ = G_{1 m} ∗ z on the native grid; T_{r,θ}(z, p) = 2 z̄(p) − z̄(p + r u_θ) − z̄(p − r u_θ), r ∈ {2, 4, 8, 16} m,
  θ ∈ {0, 45, 90, 135}°: 16 components. Complete source support required (§6).
- **FIT anchors:** 8,192 = 64 in each of 128 FIT windows (64 positive + 64 background, `SELECTION_PRIVATE.json`). Per
  window: the central unshifted 256 m view (k = 0); anchors are drawn by `default_rng(20261030 + window index)` without
  replacement from the 0.5 m pixel centres with v = 1, complete target support and ≥ 32 m from the view boundary.
- **DEV chips:** 128 eligible exposed cores (`SELECTION_PRIVATE.json`), one chip each. Chip centre: the first of up to 50
  candidates drawn by `random.Random(20261031 + 100000 i + j)` uniformly on integer metres in [x₀ + 300, x₀ + 4,700] ×
  [y₀ + 300, y₀ + 4,700] whose 256 m view + 97 m halo is wholly valid; a core with no such candidate is dropped and the
  reduced denominator reported. Uncertainty blocks are the 5 km cores; acquisition composition is reported.
- **Perturbation:** at each DEV chip, δ = A · exp(−½ (a²/σ_major² + b²/σ_minor²)) in the rotated frame, centred at the chip
  centre + a sub-cell offset ~ U[0, 1)² m, with seeded uniform orientation (`default_rng(20261032 + chip index)`).
  Balanced 16 per combination of minor σ ∈ {1.5, 4} m × aspect ∈ {1, 3} × peak A ∈ {0.05, 0.15} m (combination =
  chip index mod 8 in the sorted chip order). z + δ and z − δ are each rendered through the pinned input pipeline of the
  state being read, at grid phases (0, 0) and (0.25, 0.25) m. Background, validity and phase are identical within a pair.
- **Features:** the 976-d vector at the anchor (δ's centre), §6 sampler, from the 512 px view centred on the chip.
- **Readout:** an affine ridge map from features to the 16 targets, FIT-only standardisation, effective degrees of
  freedom fixed at **64 per output** (λ solves Σ_j s_j²/(s_j² + λ) = 64 from the FIT feature matrix alone; if the numerical
  rank is below 64, the rank is used and reported). No DEV tuning, no hidden layer.
- **Q** = 1 − Σ_{i,φ} ‖S[(g(F(X_φ(z_i + δ_i))) − g(F(X_φ(z_i − δ_i))))/2 − T(δ_i)]‖² / Σ_{i,φ} ‖S T(δ_i)‖², S the FIT
  target standard deviations floored at 10⁻³ m.
- **States (19):** the common ADAF initial state; G-SSL and D-SSL × 3 (pre-F1); A, B, G and D × 3 (post-F1). The readout is
  refitted on FIT for every state. **Baselines:** a 17 × 17 grid at 2 m spacing from the distinct legacy channel and from
  B's four channels, same ridge rule. Unperturbed DEV prediction error is reported.
- **Reported:** paired D − B and D − G (and D − A post-F1) differences, per core block, before and after F1, by phase,
  with a 10,000-replicate core-cluster bootstrap (`default_rng(20261029)`). Improvement surviving F1 supports accessible
  signed local geometry; it does not establish archaeological specificity or discovery ability. A copy of terrain can
  pass. **No proxy-score cutoff can veto a downstream gain**, and a readout win without a downstream gain does not earn the
  larger programme.

## 11. Resources, benchmark and fallback

- **Ceiling (whole DT1, including preparation, staging and evaluation):** 48 elapsed hours from the first job-owned VM
  allocation and **48 aggregate job-owned Colab VM-hours**. At most 4 job-started workers at once and never more than 6
  Colab machines in total including other jobs (CLI active-assignment count read before every `new`). Act only on
  machines this job started. The local GPU is not used for DT1 training or decoding.
- **Storage:** `verify/v3/archive_gate_v2.py check` and `verify/v3/storage_guard.py` (honest peak + 20 GB reserve, actual
  volume) before every new download, training or inference launch, locally and on each VM. Local downloads are weights and
  receipts only. Archive verification by size and checksum; until verified, both copies count.
- **LGLN:** at most 1 request/s aggregate for this job; HTTP 429 stops the reader.
- **Benchmarks before G/D production:** the two disposable 200-update SSL runs; B/101's full F1; continuous inference on
  the 4 preselected bench cores (no efficacy score inspected); readout extraction timing.
- **Forecast:** T_train = (200 + 3U)(t_G + t_D) + 3 f_A + 9 f_B with measured t_G, t_D (s per SSL update), f_A, f_B (full F1
  runtimes; f_B also stands for G and D's F1), U = 4,096; plus measured staging, all primary / shifted / benchmark
  inference, feature extraction, ridge fitting, proposal processing and archival, plus the VM-hours already spent.
- **Predeclared fallback (before any G/D efficacy score):** if the forecast exceeds either ceiling, both G and D use
  **U = 2,048**, keeping three chains, all twelve F1 runs and the same evaluation. If that still cannot fit, stop after
  preparation and benchmarks and report resource infeasibility. No single chain or shortened F1 is reported as a completed
  experiment.

## 12. Order of work and exposure

1. This freeze (committed before any DT1 arm is trained or scored).
2. Addendum P: adapter, FP32 parity, coordinate tests. Addendum H: halo staging manifest.
3. A and B: six F1 runs, then decoding on the 210 cores, the shift subset and the comparators.
4. **A/B scoring is allowed once this freeze is committed** (it covers G and D). A/B results change no definition of G or D.
   Any change made after A/B is seen is recorded as a deviation with its timing.
5. G/D benchmarks, resource forecast, then G/D production if it fits (§11).
6. Signed-shape readout over the 19 states; final G/D scoring.

No output on an evaluation core is used for any selection, threshold or design choice.

## 13. Forecasts (recorded before any DT1 code)

Coordinator (`CODEX_THREAD_01a16ef7_ASSESSMENT.md` S3), 3 seeds, 2/km², 210 exposed cores:

| Event | P |
|---|---:|
| **B − A paired mean > 0, all-record recall** | **0.45** |
| B − A ≥ +3 points | 0.20 |
| D − B paired mean > 0 and ≥ 2/3 chains positive | 0.25 |
| **Pilot ambition met (§8)** | **0.07** |
| Implementation or resource failure before scores (fallback or stop) | 0.25 |

## 14. Pins

- **Data pins:** `EVAL_CORES_PRIVATE.json` `1d82dee8…`, `EVAL_TRUTH_PRIVATE.json` `fe9575b9…`, E0 (`E0_RESULT.json`),
  stage-B `RESULTS_LSFETCH.tar` `c7cc89ed…cd43`, `FIT_TABLE_PRIVATE.json` `58d6aa16…`, D1 run 05 `d9ece1fb…`, ADAF barrow
  `51b679fa…`, SUBTLE-1 stage-B D1 05/06/07 proposals (hashes in CR-1 / RERANK-1 receipts), `SELECTION_PRIVATE.json`
  `568f1c3a…`.
- **Frozen code imported unchanged:** `d1_train.py`, `match_train.py`, `match_schedule.py`, `nds_schedule.py`,
  `adapt_train.py`, `r2_train.py`, `adaf_configs.py`, `adapter.py`, `tta_run.py`, `ot_infer.py`, `ls_mosaic.py`,
  `lgln_reader.py`, `nds_reader.py` helpers, `masks.py`, `cr1_score.py`, `cr1_analysis.py`, `scot_score.py`,
  `strict_mask.py`. None is edited.
- **New code** (this folder) is pinned by sha256 in Addendum P (adapter, channels, coordinates, trainer), Addendum H (halo
  manifest and stager) and each launch note, before the step it serves.

## 15. Departures from PRO_SPEC and limits

1. G and D share one point multiset; G pools over those points, not the whole field (Codex 1).
2. D − A joins D − B and D − G in every dense-learning rule (Codex 2).
3. The incumbent-fixed budget-cap recall is removed; CR-1's per-core budgets already prevent reallocation (Codex 3).
4. The −2-point non-inferiority guards with PASS / UNRESOLVED / HARM labels define "material subgroup sacrifice"
   (Codex 5).
5. The maximum operating comparator has a predeclared simultaneous interval (§8).
6. R0 / R0 + PROX (D1 05–07) are reported alongside the single operating references (task brief).
7. A/B may be scored before G/D are trained, under this one freeze (task brief; coordinator S2).
8. Seeds 101/202/303 are new F1 plans: A is not a replay of CR-1's A or of D1 run 05.
9. The legacy channel keeps its two existing NaN policies (training V = 0.5 with IGNORE; inference 0). Codex's "NaN → 0"
   statement describes the inference chain.
10. The 210 cores are exposed three times; nothing here is a confirmation or a discovery claim. The 738-core pool cannot
    confirm gains of this size (PTR-3; REVIEW.md correction 3), and no fresh confirmation cohort is qualified.
11. Survey, land-use or catalogue-density effects within a core are not excluded by the placement, acquisition and
    readout checks.
