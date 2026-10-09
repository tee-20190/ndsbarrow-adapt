# STAGE2-D1-CONTROLS: the local barrow-training package against an equal-update nonlocal mixed-class replay control (packet, 2026-10-07)

Status: **FROZEN, not launched.** The freeze is `freeze/FREEZE.json` (its sha256, the test record and the archive-gate
check are in `FREEZE.md`). Nothing in this directory has used a GPU or a Colab VM. The coordinator launches
(`CHECKLIST.md`, "Launch").

Capability line: reliable regional adaptation (adapting a public specialist to a new country from its public register).

Sources of the rules:

- GPT-6 Pro Round 3, R3-3, "Stage 2: only if Stage 1 passes, run matched controls"
  (`work/gpt6_pro_collab_2026-10-07/ROUND3_REPLY.md:297-326`).
- GPT-6 Pro Round 4, R4-1 "Stage 2 control class" and R4-2 "Stage 2 execution"
  (`work/gpt6_pro_collab_2026-10-07/ROUND4_REPLY.md`), adopted in `ROUND4_REVIEW.md`, "Adopted under standing
  authority", items 1-4.
- Stage 1 passed: `work/stage1_d1_v0_2026-10-07/RESULT.md` (commit 70fd2e03; `RESULT.json` sha256 `3818ac2d…`). At
  b = 0.5 RAW the mean over the ten D1 runs minus unchanged ADAF barrow was +0.0625 (0.231 against 0.169), with 10/10
  runs positive.

## Question

Pro, R4-1, "Decision: accept the existing control for a comparison of training packages" (verbatim):

> **On Verden development terrain, does the selected Lower Saxony barrow-training package outperform the specified nonlocal mixed-class replay package, when both start from ADAF barrow and receive the same update budget under the selected recipe?**

It does **not** answer (Pro, same section, verbatim):

> Does geographical locality itself improve performance, independently of class specificity, exposure, label composition and terrain source?

The control is called the **equal-update nonlocal mixed-class replay control** in every table and method description
here. It is not an unqualified "matched control": class, exposure, label composition and terrain source differ (see
"Declared deviations" and "Composition, measured").

Pro (R3-3) separates three questions:

- Does additional training beat the unchanged public checkpoint? (Stage 1: yes, on Verden.)
- Does the local training package beat an equal-update alternative use of additional training? (This packet.)
- Does the resulting pipeline beat the strongest applicable public detector?

A method can pass one and fail another.

## Design: ten equal-update nonlocal mixed-class replay controls, one per D1 run

Each D1 run `D1F1_BARROW_L_<s>` (s = 2026092805 … 2026092814) gets one control `S2CTL_BARROW_<s>`.

| Property | D1 run s (D1-COLLAPSE-1 session 3, freeze `f323e17e…`) | Control s | How it is held equal |
|---|---|---|---|
| Start | ADAF barrow tar, sha256 `51b679fa…`, 861,530,273 B | The same tar | `match_train.start_weights` / `fresh_model` (size and sha256 checked) |
| Recipe and code | F1 (`d1_train.update`): BCE + 1 − soft Dice, BCE only on empty-target windows; fresh AdamW, lr 1e-4, wd 1e-4; cosine 1e-4 → 1e-6 over 5,116 updates, no warm-up; frozen BatchNorm; batch 1; fp32, TF32 off | The same | `s2_train.py` imports `d1_train.update`, `lr_at`, `Detector` unchanged (sha256 `2e232c77…`) |
| Collapse detector and curves | `d1_train.Detector` (trailing 256; records only); per-update curves | The same | Same objects |
| Updates and schedule | 5,116 updates in two epochs numbered 4 and 5; update t = (e − 4) × 2,558 + step | The same | `nds_schedule.t_index` / `lr_at`; the dry run checks t = 0…5,115 once each; the barrier checks the executed counter |
| Checkpoint selection | Final weights only | Final weights only | `final.pt` at t = 5,115 |
| RNG and pairing | `torch.manual_seed(s)`; draw streams of seed s | `torch.manual_seed(s)`; arm C's draw streams of seed s | Paired by seed and by update index (identical learning rate at every t) |
| Positive vs background windows | Positive at every even step, background at every odd step: 2,558 + 2,558 | The same: 2,558 + 2,558 | Verified for all ten seeds (`DRYRUN_PLAN.json`) |
| Window size and crops | 846 × 846 px windows (423 m at 0.5 m); 512 px crops with ±160 px offsets and 8 dihedral transforms (`r2_train.draw`) | The same: every English, Welsh and Irish window is 846 × 846 px; same crop and draw function | R2 manifest shapes; `adapt_train.AdaptSteps` + loader (void policy V) |
| Training data | ADAPT-NDS-1's Lower Saxony FIT table: 852 positive + 852 background windows, register barrows only | **No local data.** ADAPT-NDS-1 arm C's nonlocal mixed-class replay pool (below) | The substantive difference (with the deviations it brings) |
| DEV inference | `infer_irl.py` (`f3855551…`), FRESH-1's 717-tile inference manifest | The same, in a later session | Inference is VM-independent (EQ-1, bit-identical) |

### What the control trains on (exact)

The control's 5,116 steps are **ADAPT-NDS-1 arm C's step map at seed s**: its equal-update control without Lower
Saxony data (`work/adapt_nds_2026-10-05/PACKET.md`, "Arms"; `nds_schedule.py`; `nds_train.schedule("C", …)`). C took
J's own structure for epochs 4 and 5. The only change here is the seed. ADAPT-MATCH-0
(`work/adapt_match_2026-10-05/`) contributes the equal-update structure:

- 5,116 updates from the ADAF start;
- positive/background alternation;
- no replay in the D1 arm;
- final weights only.

ADAPT-MATCH-0 has no nonlocal arm of its own. So the nonlocal pool comes from ADAPT-NDS-1's arm C, the documented
control.

| Source (pair rule within each epoch e) | Updates (pos + bg) | Windows and labels |
|---|---|---|
| English: every pair not taken by Welsh or Irish (`plans.real_plan(s, e, …)`, the generator of J's plans) | 2,558 (1,279 + 1,279) | R2 FIT windows: 1,279 positive and 741 negative, with NHLE masks (`R2_MASKS_MANIFEST_PRIVATE.json`, sha256 `f44a36bd…`). The 804 holdout windows are never drawn |
| Welsh: pairs with (p − e) % 4 == 0 (HR1 A_W30 draws, `welsh_train.welsh_plan`) | 1,280 (640 + 640) | The seed's 30-monument subset of HR1's FIT table (`FIT_TABLES_CANDIDATE.json`, sha256 `65c494fd…`); background from its 741 background windows |
| Irish: pairs with (p − e) % 4 == 2 (HR3 A_I30 streams, `joint_train.irish_plan`) | 1,278 (639 + 639) | The seed's 30-monument subset of HR3's FIT table (sha256 `9ef37686…`); background from its 676 background windows |
| Lower Saxony | **0** | Never read. No Lower Saxony path enters the control's context (`s2_schedule.control_context` refuses one). The bundle refuses the FIT table, plan and masks (`bundle_s2.py`). `TRAIN_RESULT.json` records `local_labels_read: false` |

- **Labels.** All replay labels are a single archaeology channel (output channel 1), as J and C trained it
  (`JOINT-1 AUDIT.json` class_mapping). English labels are NHLE barrows and enclosures. Welsh and Irish labels are
  barrows and enclosures; the Irish enclosures include raths and cashels.
- **Subsets for seeds 07–14.** JOINT-1 recorded the 30-monument subsets for seeds 00–06 only. Seeds 07–14 use the same
  documented rule:
  - the rule is `audit_mixture.subset`, `default_rng([seed, 30]).choice(sorted(monuments), 30, replace=False)`;
  - `s2_schedule.subsets` fails closed unless the rule reproduces every recorded seed (00–06) exactly. It does, for
    both regions (tested).
- **Generator identity (measured).** At ADAPT-NDS-1's seeds 02–04 this generator reproduces C exactly. The per-epoch
  draw digests equal C's `TRAIN_RESULT.json`, and the English epoch 4–5 plans equal C's pinned `ENGLISH_E45` files
  (`DRYRUN_PLAN.json`, `c_identity_02_04: true`). So the control is C's step map with the D1 start and recipe, not a
  re-specification.

### Composition, measured (CPU, all ten seeds per arm; `COMPOSITION.json`, `DRYRUN_PLAN.json`)

Pro R4-1 items 1-5, from the actual step tables (the frozen draw functions: every update's window, (dy, dx) and
dihedral k, applied with `rt.crop` and `rt.dihedral`) and the masks built locally from their tables
(`build_masks.sh`, under `storage_guard.py`). Target definition exactly as `d1_train.terms`: valid = label ≠
IGNORE, positive = label = 1. **Descriptive only: none of these numbers changed the replay.** Ranges are over the
ten seeds; a single value means all ten are equal.

| Quantity | D1 (local barrow package) | Control (equal-update nonlocal mixed-class replay) |
|---|---|---|
| (a) Positive draws per distinct source window | 852 windows: 850 drawn 3×, 2 drawn 4× | English 941–968 windows (≤ 2×); Welsh 32–40 (≤ 22×); Irish 30 (≤ 22×) |
| (a) Background draws per distinct source window | 852 windows, 3× or 4× | English 651–673 windows (≤ 4–5×); Welsh 508–532 (≤ 2×); Irish 573–580 (≤ 2×) |
| (b) Distinct primary monuments in positive draws | 852 | 967–988 (English 907–928, Welsh 30, Irish 30) |
| (b) Positive draws per monument: median / 95th percentile / max | 3 / 3 / 4 | 1 / 21 / 22 (English monuments mostly 1–2, max 7–11; each Welsh / Irish subset monument 20–22) |
| (b) Share of positive draws whose window holds further (incidental) monuments | 68.1 %–68.2 % | 28.3 %–32.0 % |
| **(c) Positive-role crops with an empty target** (F1: BCE only) | **0.0 %** (0 of 2,558) | **4.1 %–4.9 %** (104–125 of 2,558): English 8.1–9.8 %, Welsh 0.0 %, Irish 0.0 % |
| **(c) Background-role crops with an empty target** | **100.0 %** | **100.0 %** (no background crop holds a positive pixel) |
| (c) All-IGNORE crops (skipped updates) | 0 | 0 |
| **(d) Foreground fraction of positive-role crops: mean** | **1.80 %–1.98 %** | **5.90 %–7.92 %** |
| (d) Positive-role foreground fraction, zeros included: min / 5 / 25 / 50 / 75 / 95 % / max | 0.03 %–0.03 % / 0.10 %–0.11 % / 0.24 %–0.25 % / 0.50 %–0.52 % / 0.99 %–1.03 % / 2.99 %–3.21 % / 100.00 % | 0.00 % / 0.28 %–0.63 % / 1.53 %–1.92 % / 2.27 %–3.15 % / 5.41 %–6.87 % / 20.77 %–33.82 % / 97.08 %–100.00 % |
| (d) Zero share of positive-role crops / background-role foreground | 0.0 % / 0 in every crop | 4.1 %–4.9 % / 0 in every crop |
| (e) Class of positive-role steps | barrow 100.0 % | barrow 49.1 %–58.6 %, enclosure 33.4 %–43.2 %, mixed-class English windows (barrow or enclosure with hillfort, moated or motte parts) 67–80 steps, English windows without a recorded own part 119–136 steps |
| (e) Barrow share of positive steps by update decile (t // 512) | 100 % in every decile | 45.7–61.3 % over all deciles and seeds (no drift over training) |

Class by region at seed 05 (positive-role steps, barrow / enclosure): English 905 / 170, Welsh 299 / 341, Irish 64 / 575.

Readings (descriptive):

- **Empty positive-role targets.** D1 never sees an empty positive crop; the control sees one in 4.1–4.9 % of its
  positive steps, all of them English (8.1–9.8 % of English positive steps; Welsh and Irish 0 %). The English
  positive windows without a recorded own part (119–136 steps, class "unknown" above) plausibly account for most of
  them; that overlap was not measured separately. F1 scores empty targets with BCE only, so the control's 2,558
  positive steps carry 2,433–2,454 Dice-bearing updates against D1's 2,558 (Pro R4-1: alternating source windows
  does not guarantee equal positive targets).
- **Foreground content.** Control positive crops hold about 3–4× more foreground on average (enclosures are large);
  D1's barrows are small (median 0.5 % of the crop).
- **Exposure.** D1 spreads its 2,558 positive draws evenly over 852 monuments (3–4 each). The control spreads 1,279
  over 907–928 English monuments (mostly 1–2 each) and concentrates 1,279 on its 60 Welsh / Irish subset monuments
  (20–22 each).
- **Background.** Both arms' background crops are entirely empty targets at mask level.
- **Mask-level approximation.** Void policy V (non-finite relief → IGNORE) needs the terrain. For the English
  windows (terrain local) the exact V-applied targets were computed: 0 English crops change (no English relief
  window read by the control has a void). Lower Saxony, Welsh and Irish targets are mask-level; on the VM the
  barrier's mask-bound check compares every control update's target with its staged mask crop.

### Declared deviations (exact matching impossible or not attempted)

1. **Label content.** D1 positives are register barrows (Grabhügel) only. Control positives are mixed barrows and
   enclosures on one channel, the labels C used (class shares by step: "Composition, measured", (e)).
   - The control therefore also teaches the barrow model to fire on enclosures.
   - A PASS means: the local barrow-training package beats this nonlocal mixed-class replay package. It does not
     isolate locality from task specificity (Pro R4-1: "It does **not** answer …").
   - A class-matched barrow-only control is a separate design (Pro R4-1, "What a class-matched experiment would
     require"), not a prerequisite here.
2. **Exposure distribution.** Measured in consistent units in "Composition, measured" (a), (b). These are C's
   documented draws. Equalising them would change C and define a different comparison (Pro R4-1, "The exposure
   mismatch need not be fixed for this question").
3. **Positive-pixel content per crop.** Not matched by construction. Measured before the freeze ((c), (d)): it decides
   where F1 applies Dice.
4. **Augmentation realisations.** Crop offsets and dihedral indices come from C's draw streams (seed + 20 for Welsh and
   Irish; the English plan's own generator), not from D1's Lower Saxony streams (seed + 30). The distribution
   (`r2_train.draw`) is the same. Pairing is by seed, `torch.manual_seed` and update index. Equal seed numbers do not
   establish common-random-number noise cancellation (Pro R4-2); the independent-resampling sensitivity covers this.
5. **Terrain sources.** English, Welsh and Irish LiDAR and their preprocessing differ from LGLN DGM1. This is part of
   the intervention (nonlocal data), not a nuisance to remove.
6. **Separate session and VM.** The D1 runs trained on 2026-10-06/07 on a different G4 VM. G4 training is compatible
   but not bit-identical across VMs (TR-1; step-0 Δ 8.9e-8). The environment gate (`setup_env.sh`, step 0, parity) is
   the same. Every control records its software environment, precision flags, hardware and update counters
   (`RUN_CHECK.json`). Pro R4-2 rule: if the executable configuration and numerical treatment are compatible,
   proceed; a material software or precision change is resolved before scaling. No bridge run is planned.
7. **Selection asymmetry.** The F1 recipe was selected on Verden recall of D1's development seeds (02–04), which were
   trained on local data. This can favour D1 over the control. Disclosed; not correctable inside this design. Ten
   later runs establish repeatability conditional on that recipe; they do not remove the selection optimism (Pro R4-1).

## Endpoint and gate (frozen with the packet)

Pro's text (R3-3, verbatim):

> My full practical gate would retain the three-point threshold against the matched control as well, with at least
> 8/10 positive paired differences. To call the result a **clear local improvement**, I would additionally require a
> positive lower joint 95% interval at the primary budget, with the spatial sensitivity results disclosed.

Renamed by Pro R4-1 ("Retain the numerical gate; rename the inferential claim"): the "clear local improvement" label
is **"clear advantage over the prespecified replay control on Verden"**.

**Primary scorer.** Benchmark v0 on Verden, exactly as Stage 1 scored it: `score_v0.py` functions imported read-only.
That means all 717 tiles and 728 individual barrows, 422.84 km², one-to-one maximum-cardinality matching, a 15 m
footprint reach, K = floor(b × 422.84) with every proposal counted, and the legacy IGNORE handling. The primary decoder
is RAW and b = 0.5 (K = 211).

**Gate, decided on unrounded values (integer matches).**

- d_s = matches(D1_s) − matches(CTL_s) at RAW, b = 0.5, for s = 05 … 14 (recall difference = d_s / 728).
- Size criterion, **inclusive**: mean(d_s / 728) ≥ +0.030, evaluated exactly as 100 × Σ d_s ≥ 3 × 728 × 10, i.e. a
  mean of at least 21.84 additional matches (Σ d_s ≥ 219). Displayed three-decimal recalls never decide a boundary
  case.
- Consistency criterion, **strict**: at least 8 of 10 d_s > 0. A zero difference is not positive.
- **PASS** iff both hold. Otherwise **FAIL**.
- **Clear advantage over the prespecified replay control on Verden** iff PASS **and** the primary paired bootstrap's
  unrounded lower 95 % bound is > 0 (strict). Sensitivity results are disclosed and never substituted.
- Not allowed: best-seed substitution, ensembles, redraws, a lowered threshold, dropping a seed, a smaller denominator.

**Paired bootstrap (`boot_s2.py`, built and frozen before any control exists).** Pro R4-2, "Freeze the bootstrap
before any control is scored". The engine (`v0_uncertainty/engine.py`) and the scorer (`score_v0.py`) are imported
read-only.

| Item | Frozen value |
|---|---|
| Scorer | `score_v0.py` (benchmark v0, legacy IGNORE). All-ones block weights must reproduce score_v0 recall of every system exactly, else the bootstrap refuses |
| Seed correspondence | The ten prespecified pairs (D1F1_BARROW_L_s, S2CTL_BARROW_s), s = 05 … 14 |
| Geography (primary) | 2 km cells from origin (0, 0), R3-2 anchor + 15 m reach merge (`engine.build_blocks`); block weights ~ Multinomial(J; 1/J), shared by all systems and both arms within a replicate; K* = floor(b × A*); recall = M*/N* |
| Training resampling (primary) | Ten pair indices drawn with replacement per replicate; statistic = mean over drawn pairs of recall(D1_i) − recall(CTL_i) on the shared geography |
| Replicates, RNG, quantiles | 5,000; `numpy.random.SeedSequence(2026100702)` spawned into four named streams (geo, train_paired, train_independent_d1, train_independent_ctl); percentile 2.5 / 97.5 with numpy linear interpolation |
| Primary budget | RAW, b = 0.5 |
| Positive lower bound | Unrounded lower95 > 0 (strict) |
| Sensitivity (prespecified, not substitutable) | (i) training runs resampled independently between arms, paired geography; (ii) the 4 km grid and the 2 km grid offset by (1,000 m, 1,000 m), under both designs. If the interval's sign depends materially on these, the result description says so (Pro R4-2) |
| Undefined replicates | N* = 0: dropped and counted (expected 0; none in the pseudo-control run) |
| Incomplete / invalid runs | Fewer than ten technically valid pairs: the bootstrap is not computed and the gate is INCOMPLETE (no smaller denominator). Collapsed controls are valid runs and stay in |

End-to-end proof on existing data (`BOOT_S2_PSEUDO.json`, `tests/test_boot_s2.py`): the ten D1 runs against
unchanged ADAF barrow as a pseudo-control reproduce Stage 1's point estimate exactly (+0.0625).
Pseudo-control primary (2 km, paired, RAW b = 0.5, 5,000 replicates): point +0.0625, 95 % [+0.0389, +0.0849], 0 undefined replicates. Sensitivity:
2km/paired [+0.0389, +0.0849]; 2km/independent [+0.0391, +0.0852]; 4km/paired [+0.0380, +0.0869]; 4km/independent [+0.0383, +0.0869]; 2km_offset1000/paired [+0.0331, +0.0895]; 2km_offset1000/independent [+0.0333, +0.0899].
With identical arms the paired design gives exactly [0, 0] and the independent design an interval centred on 0;
exchangeable arms give an interval centred near 0; swapping arms negates every replicate (`tests/test_boot_s2.py`).

**Prespecified sensitivity: evaluator v0.1** (Pro R4-6, "Preserve the already-started experiment"; ROUND4_REVIEW.md
"Adopted" item 8). The primary stays the exact Stage 1 scorer. `score_s2.v01_sensitivity` re-scores the same
proposals with evaluator v0.1 (`work/evaluator_v0_1_2026-10-07/`, the role-based IGNORE rule) **only if that directory
holds a FREEZE receipt** (`FREEZE.json`, `FREEZE.md` or `freeze/FREEZE.json`) at scoring time. Otherwise it records
`REFUSED_NO_FREEZE` and the primary is unaffected. When it runs it records the sha256 of every receipt and of the entry
module, refuses on drift against the v0.1 freeze's own pins, and reports the RAW b = 0.5 paired differences and the
gate form as a sensitivity. Interface (fixed here): the module named by `FREEZE.json` `entrypoint` (default
`score_v0_1.py`) exposes `function` (default `score_systems`), called as `score_systems(objs)` and returning
`{system: {"RAW": {"0.5": {"matches", "recall"}}}}` plus optional `_entities_N`. A missing interface is recorded as
`REFUSED_INTERFACE_ABSENT`.

**Collapsed controls are results.** A control is collapsed under the RECIPE.md definition if any of these holds:

- (a) zero Verden proposals;
- (b) FRESH-1 group primary below 0.2778, half of ADAF barrow's 0.5556;
- (c) the detector is ON at the end of training.

A collapsed control keeps its measured recall in the gate and the bootstrap. A line with collapsed pairs excluded is
reported, and it is descriptive only.

**Reporting (`score_s2.py` → `RESULT.json`, `BOOT_S2.json`).**

- Every seed: D1, control, ADAF barrow; each arm against unchanged ADAF barrow (Pro R4-1: a deteriorating and an
  improving control lead to different readings of the same difference).
- b = 0.5, 1 and 2, for RAW and NMS60.
- The three-way comparison: D1 − ADAF barrow (Stage 1, re-derived and checked identical); control − ADAF barrow;
  D1 − control.
- Matches per 100 slots, ceiling c = min(1, K/N), supply, and the collapse table.
- Caveat: ADAF barrow's in-core supply (794 RAW) is below K = 845 at b = 2.

**Reading the outcome** (Pro R4-1, "Retain the numerical gate; rename the inferential claim", table rows verbatim):

| Outcome | What you may say |
|---|---|
| Practical gate passes and primary joint lower bound is positive | The selected local barrow-training package exceeded the specified replay package by the reported amount and met both predefined criteria. |
| Practical gate passes but interval criterion fails | The practical replication criterion was met; the predefined uncertainty criterion was not. |
| Mean improvement is positive but below three points | A positive estimated advantage was observed, but it did not meet the material-improvement threshold. |
| Mean is at least three points but fewer than 8/10 differences are positive | The average improvement met the size threshold but failed the run-consistency criterion. |
| Neither treatment clearly leads | The study did not demonstrate a material incremental advantage over this replay package. Both may nevertheless improve on unchanged ADAF barrow. |
| Replay performs better | The specified replay package performed better on Verden under this recipe. Stage 1’s improvement over unchanged ADAF barrow still stands. |
| Gate passes with collapsed controls | The protocol-level comparison passed, including the prespecified training failures; the difference includes replay instability. |
| Ten prescribed runs are not completed within the cap | The experiment is incomplete under its registered gate. Do not substitute a smaller denominator. |

Row selection (`score_s2.outcome_row`, prespecified; Pro gave the wording, these are the operational boundaries):
PASS → row 1 if the primary lower bound > 0, else row 2, plus row 7 when any control is collapsed. FAIL → row 4 if the
size criterion holds; else row 3 if Σ d_s > 0; else row 6 if Σ d_s < 0 and the primary interval's upper bound < 0;
else row 5. Fewer than ten complete, verified controls → row 8, and nothing archaeological is scored.

Scope: Verden is development terrain, evaluated before by FRESH-1, ADAPT-MATCH-0, D1-COLLAPSE-1 and Stage 1. Any result
is a local development result, not confirmation across two regions (R3-3). No checkpoint is adopted by this packet.
All controls are registered `candidate_not_adopted`.

## Execution policy (Pro R4-2, "Use one session with an explicit first-run barrier")

Order, all in one session after the smoke:

1. **Session 0, smoke** (production artifact route, below). Must be `SESSION_VERIFIED.json` PASS, including its own
   barrier on the smoke run.
2. **Session 1, seed 2026092805 first** (frozen run order, not chosen by expected outcome), a full 5,116-update run.
3. **Technical barrier** on that run (`s2_barrier.py`; `chain_lib_s2.sh` `run_check … barrier`). It must PASS before
   seed 06 starts; the chain holds until then (`chain.log` order is verified afterwards):
   - **identities**: every trainer, schedule and recipe module and every table the run read equals the session pins
     and the hashes in `TRAIN_RESULT.json`; the bundle manifests (`MANIFEST_code` / `MANIFEST_english.sha256`) are
     hashed and every English raw window and mask the run read is listed there with its on-disk sha256; the Welsh /
     Irish masks it read are hashed (digest recorded); the plan summary re-derives from the pinned tables;
   - **updates**: `curves.jsonl` holds t = 0 … 5,115 once each, alternation and arm C's region counts hold, every
     learning rate equals `d1_train.lr_at("F1", t)` exactly, and the AdamW step counter equals 5,116 minus skipped
     all-IGNORE updates (expected 0, as in D1; the counter is recorded in `TRAIN_RESULT.json`);
   - **targets**: F1 target handling on every update (Dice used iff the valid target has a positive pixel; loss form);
     each update's target counts are bounded by the staged mask's crop at its (dy, dx) (equality unless void policy V
     removed pixels; an excess means wrong data); per-role empty-target share and foreground-fraction quantiles are
     recorded;
   - **checkpoint**: sha256 and size equal `TRAIN_RESULT.json`; independent reload in a fresh process through the
     inference loader (`adaf_configs.load_models`, strict); every tensor finite; parameters differ from the ADAF
     barrow start;
   - **fixture**: valid outputs of the reloaded model on a small fixed inference fixture (four English R2 FIT windows
     by a fixed id rule, through `tta_run.objects_multi`): every probability finite and in [0, 1]; map sha256 and
     object counts recorded; no score;
   - **durable readback**: the driver harvests the weight (with retries), re-reads the local copy under
     `outputs/stage2_d1_controls_2026-10-07/weights/` from disk, checks its sha256 and size against `VERIFIED.json`
     and `TRAIN_RESULT.json`, and uploads an ACK; the VM's `wait-ack` accepts only an ACK equal to the run's
     `TRAIN_RESULT.json` (600 s limit);
   - **environment**: python, numpy, torch, CUDA, cuDNN, GPU, TF32 / cuDNN flags, fp32.
   - **A technically valid all-background checkpoint passes the barrier.** No check reads recall
     (`tests/test_barrier.py::test_all_background_checkpoint_passes_the_barrier`).
4. **Seeds 06 … 14 in the same session**, in order, each verified by the same checks (`run_check … check`: everything
   above except the start difference, the data hashing and the readback wait, which the barrier established; the
   weight is still harvested and verified locally as it finishes, and the session verifier requires every run's
   `RUN_CHECK.json` PASS and every weight `VERIFIED.json`).
5. One shared Verden DEV pass of the ten controls, export, verification (`SESSION_VERIFIED.json`).

**No early stop on scores.** No archaeological score of any control is computed until all ten are complete and
verified (`score_s2.completeness`). The chain computes none.

**Recovery policy (prespecified).** A barrier or run-check failure stops the chain (mandatory gate); the export and
weight rules still apply (no VM stop before verified copies).

| Failure | Classification | Recovery |
|---|---|---|
| Wrong data: identity mismatch, mask-bound violation, plan summary does not re-derive, region counts differ | Technical invalidity (the run did not test the registered treatment) | The run is INVALID and preserved as evidence (never scored, never registered as a control). Fix the cause with a reproduction test; documented freeze amendment with identical design, data, gate and analysis; rerun **the same seed** from the ADAF start, then continue in run order. No other seed substitutes |
| Unexecuted updates: counter or curves short, LR off schedule, parameters equal to the start | Technical invalidity | As above |
| Corrupt checkpoint: sha256 / size mismatch, reload or fixture failure, non-finite tensors | Technical invalidity if the VM copy is corrupt; transfer fault if only the local copy is | Transfer fault: re-download (harvest retries, export hold). VM copy corrupt: as above |
| Readback ACK not delivered within 600 s (transfer, no local room) | Barrier not passed | Chain stops before seed 06; the verified weight is kept; a new session re-runs from seed 05 after the cause is fixed |
| Collapse, all-background output, poor or good-looking predictions | **A result**, not a failure | None. The run stays in the gate and the bootstrap |

If recovery consumes the 60 CU ceiling before ten valid runs exist, the experiment is reported **INCOMPLETE** under its
registered gate (outcome row 8). Two failures at the same subproblem: stop and diagnose (AGENTS.md).

## Compute and storage

**D1 actual spend (ledger).**

| Session | Spend | Wall time | Rate |
|---|---:|---:|---:|
| Session 3 (10 BARROW_L runs + 10-model Verden DEV pass) | 37.09 CU | 15,223 s | 8.77 CU/h |
| Session 4 (AO_L) | 37.25 CU | 15,146 s | 8.85 CU/h |

- **Per run: 3.71 CU** (session 3 / 10). Of that, training (338.5 s per run) is about 0.84 CU and the DEV inference
  share (about 874 s per model) about 2.13 CU. About 7.4 CU per session is shared: setup, staging, the DEV base pass
  and export.

**Projection** (`CU_PROJECTION.json`, from `chain/sessions_s2.py projection`). It uses D1's measured stages, plus
ADAPT-NDS-1's measured English upload (≤ 390 s) and Welsh/Irish staging (≤ 993 s / ≤ 77 s), at 8.854 CU/h. The
technical checks are **estimates, not yet measured**: barrier 300 s, each later run check 75 s, smoke barrier 240 s
(D1's harvest of a 320 MB weight took 12–22 s after `WEIGHTS_READY`). Session 0 measures the smoke barrier.

| Session | Contents | Projected | Cap (1.25×) |
|---|---|---:|---:|
| 0 (smoke) | One control seed, 64 updates, production artifact route with forced transfer faults, the barrier | 4.46 CU | 5.58 CU |
| 1 (run) | Ten controls in run order (barrier after seed 05, a check after each later seed) + one 10-model Verden DEV pass | 43.20 CU | 54.00 CU |
| **Total** | | **47.66 CU** | **59.58 CU** |

**Ceiling: 60 CU, hard stop, cumulative over both sessions** (`chain/CU_SPEND.jsonl`).

- A launch is refused unless spend so far + 1.10 × the projection ≤ 60.
- The VM-side wall cap is the stop. `sessions_s2.py plan` sizes it so that spend so far + setup + chain + export
  stays ≤ 59 CU, keeping 1 CU for the export hold. HR1's `gates.py budget` then refuses every stage that would pass
  it (the barrier and run checks have their own budgets: 900 s / 240 s).
- The CU monitor alerts and records overruns but stops no VM. Owner rule: no VM stop before verified copies of every
  weight and result.
- Pro F4-01 forecasts 90 % that all ten technically valid runs complete within 60 CU including smoke and recovery.
  Headroom at caps is 0.42 CU, so one recovery session would exceed the ceiling; the budget-revision policy (owner,
  2026-10-05: re-price with a documented decision) applies to a measured overrun.

**Route.** Official Colab CLI (`work/colab_cli_2026-09-30`), G4, one VM, one heavy job at a time, sessions strictly
sequential. Session 1 starts only after session 0's `SESSION_VERIFIED.json` is PASS. Nothing runs beside another GPU
chain: the monitors read the account-wide balance. `chain/driver_s2.sh` is D1's `driver_d1.sh` transformed by
`chain/make_driver_s2.py`. That script fails closed on any unmatched replacement. The changes are:

- paths;
- two bundles (code, and ADAPT-NDS-1's English R2 bundle of 1.25 GB);
- the Welsh/Irish inputs;
- a HEAD check of the Verden DGM1 tiles, the GSI zips and the Welsh tile URLs;
- the JOINT-2 recovery keep-alive during uploads (the colab idle-reclaim lesson);
- the first-run barrier: the barrier run's weight is harvested with bounded retries, re-read from the local disk
  (`s2_barrier.py ack`) and the ACK uploaded to `/content/cli/` on the VM; the driver refuses `HARVEST=0`;
- FAULT_INJECT, required in session 0 and refused in session 1;
- one CU ledger.

The VM job, the chain library and the gates are checked copies of D1's (`chain/vm_job_s2.sh`, `chain_lib_s2.sh`,
`s2_gates.py`), with `run_check` added after every trained run. The Welsh/Irish staging follows ADAPT-NDS-1's
`do_stage_w` / `do_stage_i` unchanged.

**Local dry run of the barrier route** (`chain/barrier_dryrun.sh` → `DRYRUN_BARRIER.json`, PASS): the real driver
functions over `chain/colab_local_s2.sh` (an emulated `/content` with download and upload) against a fake VM job that
runs the real `wait-ack`. With `FAULT_INJECT="harvest:1 ack_upload:1"`: the first harvest and the first ACK upload
fail and are retried; the ACK arrives; the next seed starts only after `BARRIER PASS`; both weights are VERIFIED; a
corrupted ACK is refused. The D1 export route (results parts, export hold) is unchanged and was not re-run locally.

**Weights.**

- Each finished run is announced `WEIGHTS_READY`. The driver downloads its `final.pt` while the chain continues and
  verifies sha256 and size against `TRAIN_RESULT.json` and the VM sidecar. It writes
  `outputs/stage2_d1_controls_2026-10-07/weights/<run>/VERIFIED.json`.
- The VM stops only after every weight and the results tar are verified (`export_with_hold`).
- After the session the weights are archived to Drive with an independent read, as for D1
  (`archive_completed_2026-10-07k1`), before any local reclaim.

**Local storage.**

| Session | storage_guard declared peak | Local launch rule (20 GB reserve + peak + 2 GB page-file headroom) |
|---|---:|---:|
| 1 | 3.7 GB (10 × 319,970,247 B + 0.5 GB) | ≥ 25.7 GB |
| 0 | 0.82 GB | ≥ 22.8 GB |

- A weight download waits while C: free minus the weight is below 22 GB. The barrier run cannot wait: with less room
  its readback ACK is not delivered and the barrier fails before seed 06 (recovery table).
- The pre-freeze mask build (CPU) ran under `verify/v3/storage_guard.py` with a declared peak of 0.5 GB; actual
  5.6 MB (3,445 small npz files under `outputs/stage2_d1_controls_2026-10-07/dryrun_masks/`).
- Every launch runs under `verify/v3/archive_gate.py run -- … storage_guard.py` (inside the driver). Both are
  re-measured at launch.

**Output directories (one writer each).**

- `outputs/stage2_d1_controls_2026-10-07/` (private: plans, plan checks, dry-run masks, chain runs, weights);
- `work/stage2_d1_controls_2026-10-07/` (code, receipts, RESULT).

## Smoke plan

1. **CPU dry run of the data loader composition** (`s2_dryrun.py`): plan level and mask level, PASS for all 10 control
   seeds plus the C-identity seeds 02–04 (`DRYRUN_PLAN.json`, `COMPOSITION.json`).
2. **GPU smoke on the production artifact route** (session 0, after the freeze):
   - the same driver, bundles, environment gate, pins, Welsh/Irish staging tools (on the `--only` windows that the
     first 64 updates of seed 05 read), step 0, plan check and trainer;
   - 64 F1 updates of `S2CTL_BARROW_2026092805` on the staged windows;
   - the barrier on the smoke run (vm-check with start difference, durable-readback ACK);
   - `final.pt` through the keep sink and the `WEIGHTS_READY` harvest, with `FAULT_INJECT="harvest:1 corrupt_res_part:1"`
     forcing one failed weight download (retried by the barrier harvest) and one corrupt results part;
   - a verified export, then the VM stop.
   - PASS = `SESSION_VERIFIED.json` PASS: finite smoke loss, `RUN_CHECK.json` PASS, the barrier ACK PASS on both
     sides, the weight verified locally, the export VERIFIED and the fault record present.
   - DEV staging and inference are not smoked. That path is D1's, unchanged, and verified twice (sessions 3 and 4).
3. **Local barrier dry run** (`chain/barrier_dryrun.sh`), done before the freeze (above).

## Forecast (filled before the freeze)

| Forecast | P | Source |
|---|---:|---|
| P(PASS) at the gate | **0.70** | Coordinator. Rationale: in ADAPT-NDS-1 the equal-update control C stayed level with its start (+0.1 to +0.7 points); D1 leads ADAF barrow by 6.2 points, so the replay may gain up to 3.2 points before the gate fails. The own-hypothesis cap of 0.5 is waived because an external estimate supports more (Pro F4-02 = 0.80) |
| P(clear advantage over the prespecified replay control on Verden) | **0.60** | Coordinator |
| F4-01: all ten technically valid prescribed replay runs completed, harvested and reload-verified within 60 total CU, including smoke/verification and recovery, by 21 October 2026, single-session barrier design | 0.90 | Pro, ROUND4_REPLY.md "Precisely scoreable forecasts" |
| F4-02: Stage 2 meets the practical gate (mean raw recall difference at b = 0.5 at least +0.03, at least 8/10 strictly positive), conditional on all ten runs completed and scored unchanged | 0.80 | Pro, same table |
| F4-03: F4-02 occurs and the primary 2 km paired-joint 95 % interval has an unrounded lower bound greater than zero, sensitivity disclosed but not substituted | 0.75 | Pro, same table |
| Pro's Round 3 forecast for the whole staged sequence (both substantive comparisons) | 0.30 | Pro, R3-3 |

Pro's F4-01/02/03 rows, verbatim (ROUND4_REPLY.md, "Precisely scoreable forecasts"):

> | **F4-01** | All ten technically valid prescribed replay runs are completed, harvested and reload-verified within **60 total compute units**, including smoke/verification and recovery costs | By **21 October 2026**, using the single-session barrier design | **90%** |
> | **F4-02** | Stage 2 meets the practical gate: mean raw recall difference at \(b=0.5\) is **at least +0.03**, and **at least 8/10** differences are strictly positive | Conditional on all ten runs under the proposed mixed-replay design being completed and scored unchanged | **80%** |
> | **F4-03** | F4-02 occurs **and** the primary 2 km paired-joint 95% interval has an **unrounded lower bound greater than zero** | Same condition; sensitivity results disclosed but not substituted for the primary interval | **75%** |

Evidence available at the forecast:

- ADAPT-NDS-1, J start, FRESH-1 group primary (not v0): A − C on Verden DEV +0.141 [+0.071, +0.192]; on CONFIRM +0.141
  [+0.090, +0.181]; C − J on CONFIRM +0.028 [−0.002, +0.055].
- The gate needs control mean ≤ 0.201 at b = 0.5 RAW. D1's mean is 0.231 and ADAF barrow's 0.169.

## Coordinator questions: closed

- **Q1. Class-matched control.** Closed: the existing mixed-class pool is accepted for a comparison of training
  packages, under the renamed question and control name (ROUND4_REPLY.md R4-1, "Decision: accept the existing control
  for a comparison of training packages" and "What a class-matched experiment would require": not a prerequisite;
  ROUND4_REVIEW.md "Not adopted / deferred": a class-matched control becomes a separate design only if the adaptation
  strategy depends on it). Exposure is described, not equalised (R4-1, "The exposure mismatch need not be fixed for
  this question").
- **Q2. Session split.** Closed: one session with an explicit first-run barrier after the smoke, no separate 1-seed
  session (ROUND4_REPLY.md R4-2, "Use one session with an explicit first-run barrier"; "The different VM is a
  limitation to manage, not an automatic demand for ten reruns"; ROUND4_REVIEW.md "Adopted" item 3). Implemented in
  "Execution policy".
- **Q3. Bootstrap.** Closed: built in this packet (`boot_s2.py`, read-only use of the v0_uncertainty engine) and
  frozen with it before any control exists (ROUND4_REPLY.md R4-2, "Freeze the bootstrap before any control is
  scored": "another independent bootstrap implementation is unnecessary"; ROUND4_REVIEW.md "Adopted" item 4).
  `score_s2.py` calls it directly; the "clear" label is no longer `NOT_EVALUATED` by default.

## Files

| File | Role |
|---|---|
| `s2_common.py` | Constants, the pairing table (ten D1 final hashes, D1 session-3 `SESSION_VERIFIED` sha256 `b93ac07c…`), sessions, the gate |
| `s2_schedule.py` | The control step map (C's), extended subsets, plan-level composition, smoke windows |
| `s2_composition.py` | Pro R4-1 items 1-5 from the actual step tables and locally built masks (descriptive) |
| `s2_train.py` | Trainer: pin-plans, plan-check, step0, train, smoke, smoke-windows, detector; records the AdamW step counter |
| `s2_barrier.py` | The technical barrier and per-run checks (VM), the durable-readback wait (VM) and ACK (local) |
| `s2_dryrun.py` → `DRYRUN_PLAN.json`, `COMPOSITION.json`, `dryrun/PLAN_CHECK_<seed>.json` | CPU dry run (coordinate-free aggregates) |
| `build_masks.sh` | Builds the Lower Saxony, Welsh and Irish masks from their tables (run under `storage_guard.py`) |
| `score_s2.py` | Exact Stage 1 v0 scoring, the integer gate, collapse, the three-way comparison, the clear-advantage label, Pro's outcome rows, the evaluator v0.1 sensitivity hook |
| `boot_s2.py` → `BOOT_S2.json` (at scoring), `BOOT_S2_PSEUDO.json` | The frozen paired bootstrap; the pseudo-control proof |
| `chain/` | `driver_s2.sh` (generated by `make_driver_s2.py`), `vm_job_s2.sh`, `chain_lib_s2.sh`, `chain_s2_run.sh`, `chain_s2_smoke.sh`, `s2_gates.py`, `sessions_s2.py`, `bundle_s2.py`, `freeze_s2.py`, `barrier_dryrun.sh`, `colab_local_s2.sh` |
| `DRYRUN_BARRIER.json` | Local barrier-route dry run |
| `CU_PROJECTION.json` | Projection with evidence |
| `tests/` | CPU tests (`record_tests.py` → `TESTS.json`): packet, bootstrap, barrier |
| `CHECKLIST.md` | Pre-freeze and launch steps |
| `FREEZE.md`, `freeze/` | The freeze receipt and its files |

**Attribution.**

- ADAF weights: CC BY-SA 4.0 (Čož et al. 2026; Zenodo 10.5281/zenodo.15848662). Control weights are adaptations and
  stay CC BY-SA 4.0.
- Replay data carry the attribution statements of JOINT-1, HR1 and HR3 (`s2_common.ATTRIBUTION`).
- No Lower Saxony data is used by the controls.
