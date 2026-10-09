# STAGE1-D1-V0: frozen before any v0 score of the ten D1 runs is computed (2026-10-07, Opus)

Source of the rule: GPT-6 Pro Round 3, R3-3 (work/gpt6_pro_collab_2026-10-07/ROUND3_REPLY.md).

**Question.** Do the ten saved D1 fresh-confirmation runs (public ADAF barrow + recipe F1, seeds 05-14, "BARROW_L") beat
the unchanged public ADAF barrow checkpoint under benchmark v0?

**Step 0, the legitimacy check.** It must PASS before any scoring:
- Identify the evaluation terrain and labels that lie outside every run's training windows and training-context buffers.
- If no such subset exists (all relevant Verden terrain or labels entered training), the verdict is
  NOT_EVALUABLE and nothing is scored.
- Prohibited:
  - inventing a post-hoc split;
  - using Oldenburg, any reserved area or any TEST area;
  - substituting another region.

**Pipeline**, frozen before scoring:
- Primary decoder: RAW, meaning every saved in-core proposal, no 60 m NMS, in frozen score order with a deterministic
  tie order.
- Secondary: the frozen 60 m NMS list.
- Comparator: the unchanged ADAF barrow checkpoint, same decoder, same proposals extraction.

**Scoring (benchmark v0):**
- Entities are individual registered barrows from the existing individual manifest, restricted to the eligible subset.
- One-to-one maximum-cardinality matching, with a 15 m proposal-to-footprint edge.
- K = floor(b × total scored area of the eligible subset), counting every proposal.
- b ∈ {0.5, 1, 2}.
- The legacy IGNORE handling is kept: the historical implementation.

**Gate, at b = 0.5 on the RAW decoder.** PASS requires both:
- the mean over the 10 runs of (run recall − ADAF barrow recall) is ≥ +0.030;
- at least 8 of 10 per-run differences are > 0.

**Reporting:**
- every run;
- b = 1 and b = 2;
- the secondary NMS decoder;
- the ceiling c = min(1, K/N);
- matches per 100 slots.

Not allowed: best-seed substitution, ensembles, or lowering the threshold.

**Consequence.**
- PASS → Stage 2: 10 matched controls, about 40 CU. That needs its own freeze, packet and storage/archive gates.
- FAIL or NOT_EVALUABLE → no Stage 2. The incumbent for barrow surveys stays unchanged public ADAF barrow.

**Forecast.** P(PASS) = 0.40 (Opus); Pro 0.45.
