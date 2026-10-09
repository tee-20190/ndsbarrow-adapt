# Collapse-safe fine-tune recipe (D1, F1) for reuse

The source of this recipe is D1-COLLAPSE-1. Its frozen exit rule PASSED: 0/10 collapsed per arm on fresh seeds, and
Verden validation was not below the reference. See RESULT.md, and RESULT.json for the verdict. Reuse it as specified
below; any change to it is a new recipe and needs its own check.

## The recipe

| Item | Setting |
|---|---|
| Loss per window (batch 1) | BCE over the window's non-IGNORE pixels, plus 1 − soft Dice **only if** the window's valid target has at least one positive pixel |
| Empty-target windows (all background) | BCE only |
| Soft Dice | `(2 Σ p·g + 1) / (Σ p + Σ g + 1)` on sigmoid(channel-1 logit), as `r2_train.masked_loss` |
| Optimiser | Fresh AdamW, lr 1e-4, weight decay 1e-4 |
| Schedule | Cosine 1e-4 → 1e-6 over the run's updates; no warm-up |
| BatchNorm | Frozen (running statistics from the start; `r2_train.set_train_mode`); batch 1 |
| Output | Final weights only, with no checkpoint selection |

- **Reference implementation:** `work/d1_collapse_2026-10-06/d1_train.py`:
  - `terms` and `update` with recipe `F1`;
  - `uses_dice` gives the rule;
  - `lr_at("F1", t)` gives the schedule.
- **Measured for:** ADAF AO and ADAF barrow HRNet starts, 5,116 updates alternating positive and background windows
  (ADAPT-MATCH-0's all-local plan).

## Mandatory guard for every run that uses it

- **Log per update:**
  - role;
  - Dice;
  - Σp over valid pixels;
  - maximum logit;
  - the analytic BCE and Dice logit-gradient sums.
  - Use `d1_train.terms`, which costs about one extra GPU sync per update.
- **Run the online detector** (`d1_train.Detector`): trailing 256 updates; ON iff mean positive-window Dice < 0.05
  **and** mean background-window Σp < 1.0. It records only.
- **Collapsed definition for any evaluation that uses these runs.** A run is collapsed if any of the following holds:
  - zero proposals on the evaluation manifest;
  - primary below half of the arm's reference;
  - the detector ON at the end.
- **A collapsed run is reported as collapsed.** It is never silently redrawn or dropped. A packet that wants a redraw
  rule must pre-register it.

## Scope and limits

- **Use it for:**
  - BENCH-1's equally adapted public comparators, and the adapted contender if the packet chooses one recipe for all;
  - region-in-a-box fine-tunes;
  - the pretraining pilot's fine-tune stage.
- **Not measured:**
  - J-lineage starts under F1;
  - schedules much longer or shorter than 5,116 updates;
  - regions other than Lower Saxony;
  - precision outside Verden.
  - A packet that relies on any of these should include a smoke run with the detector.
- **Background mass is higher.** F1 keeps more background mass than a collapsed model; Verden recall at 0.5/1/2
  background proposals per km² did not fall.
- **Known residual risk:** the first fresh-AdamW step still knocks the start down (update 1). F1 lets it recover within
  about 36 updates. A much more fragile start, or a much shorter run, may need the pre-registered F2 warm-up (256
  updates, peak 5e-5).
