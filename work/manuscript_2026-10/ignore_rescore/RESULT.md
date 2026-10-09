# D1 − ADAF barrow under the three IGNORE readings (saved proposals, CPU only)

Run 2026-10-09 by `rescore_d1_ignore.py` (WSL, `/home/tee-/adaf-h2h-venv`, 204 s). Output `D1_IGNORE_RESCORE.json`
(aggregates only). No inference, no new proposals, no frozen file changed; project modules imported read-only.

- **Inputs.** Verden: the Stage 1 proposal sets (sha256-checked against `work/stage1_d1_v0_2026-10-07/RESULT.json`
  provenance via `desc_verden.load()`). Oldenburg: the verified transfer session `run_s1_20261008T203551Z` (sidecar and
  session pins checked by `score_oldenburg.load_d1`), ADAF barrow's frozen Oldenburg proposals (pin `21421a9c…`).
- **Readings.** Exactly `ignore_v0.py` masks: (i) historical = no mask (= Stage 1 / Oldenburg primary = evaluator
  v0.1); (ii) literal = all non-entity register + supplement records buffered 30 m, proposals and area removed before
  K, N unchanged; (iii) explicit precedence = (ii) without cemetery/protection polygons, minus every eligible 15 m reach.
- **Uncertainty.** Frozen v0 engine, 2 km grid, geography stream `geo_<district>`, shared D1 training draws
  (`desc_verden.train_draws`), 5,000 replicates, joint percentile 95 % intervals (pointwise).
- **Self-checks (all PASS).** Reading (i) reproduces the published recalls and joint intervals exactly (Verden:
  D1-VERDEN-DESC; Oldenburg: OLDENBURG-TRANSFER-1 RESULT.json). ADAF barrow under (ii) and (iii) reproduces
  `IGNORE_V0.json` exactly; masked areas and unreachable counts equal `IGNORE_V0.json`.

## RAW, D1 ten-run mean − ADAF barrow (points), joint 95 % (2 km)

| District | Reading | b = 0.5 | b = 1 | b = 2 | Runs > 0 (each b) |
|---|---|---|---|---|---|
| Verden | (i) | +6.25 [+3.89, +8.58] | +9.37 [+6.01, +12.98] | +12.46 [+8.68, +16.77] | 10/10 |
| Verden | (ii) | +3.97 [+2.21, +5.88] | +5.52 [+3.41, +7.74] | +6.47 [+4.00, +9.31] | 10/10 |
| Verden | (iii) | +6.35 [+4.10, +8.66] | +9.86 [+6.17, +13.13] | +12.68 [+8.86, +16.90] | 10/10 |
| Oldenburg | (i) | +3.86 [+1.37, +7.19] | +4.07 [+0.79, +8.12] | +6.04 [+2.00, +9.56] | 10/10 |
| Oldenburg | (ii) | +1.92 [+0.62, +3.70] | +2.73 [+1.33, +4.44] | +3.82 [+2.20, +6.34] | 10/10 |
| Oldenburg | (iii) | +3.82 [+1.38, +7.20] | +4.05 [+0.75, +8.19] | +6.09 [+2.11, +9.56] | 10/10 |

## NMS60, same layout

| District | Reading | b = 0.5 | b = 1 | b = 2 | Runs > 0 |
|---|---|---|---|---|---|
| Verden | (i) | +5.10 [+3.52, +6.85] | +5.58 [+3.67, +7.36] | +7.32 [+5.32, +9.57] | 10/10 |
| Verden | (ii) | +3.05 [+1.80, +4.94] | +4.09 [+2.39, +5.91] | +4.89 [+2.78, +7.10] | 10/10 |
| Verden | (iii) | +4.93 [+3.57, +6.91] | +5.45 [+3.76, +7.40] | +7.35 [+5.38, +9.64] | 10/10 |
| Oldenburg | (i) | +3.46 [+1.67, +5.71] | +2.92 [+1.31, +4.80] | +3.10 [+1.64, +5.56] | 10/10 |
| Oldenburg | (ii) | +1.27 [+0.26, +2.69] | +1.50 [+0.61, +2.91] | +2.44 [+1.19, +4.40] | 10/10 |
| Oldenburg | (iii) | +3.45 [+1.65, +5.76] | +2.92 [+1.32, +4.85] | +3.16 [+1.66, +5.59] | 10/10 |

## Context (RAW, b = 0.5)

| District | Reading | ADAF recall | D1 mean recall | Per-run range (points) | Proposals removed ADAF / D1 | Unreachable / N |
|---|---|---:|---:|---|---|---:|
| Verden | (ii) | 0.109 | 0.148 | +3.57 to +4.67 | 181 / 294–325 | 445 / 728 |
| Verden | (iii) | 0.173 | 0.237 | +5.49 to +7.42 | 10 / 15–22 | 0 |
| Oldenburg | (ii) | 0.062 | 0.081 | +1.25 to +2.29 | 311 / 501–560 | 665 / 959 |
| Oldenburg | (iii) | 0.101 | 0.139 | +3.02 to +4.48 | 8 / 14–26 | 0 |

Under (ii), recall keeps the registered N as denominator (as `ignore_v0.py`). Rescaled to reachable entities only, the
RAW b = 0.5 difference is +10.21 points (Verden, 283 reachable) and +6.26 (Oldenburg, 294 reachable); this rescaling
multiplies both arms by the same factor and changes no sign.

## Reading

D1 beats ADAF barrow under every reading, district, decoder and budget, in all ten runs, with every joint lower bound
above zero. Reading (iii) moves the primary contrast by at most 0.1 points at b = 0.5 (Verden +6.25 → +6.35;
Oldenburg +3.86 → +3.82). Reading (ii) roughly halves it (Verden +3.97; Oldenburg +1.92, joint LB +0.62), because it
removes most eligible barrows from reach. The primary conclusion does not depend on the IGNORE reading; its size does.
