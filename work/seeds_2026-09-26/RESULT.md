# Three-seed confirmation: result (H1 PARTIAL, H2 HARM)

Scored 2026-09-26 22:49Z by the frozen packet (`PACKET.md`, pre-registered and frozen at e73bf132, 18:32Z). Checked
by the coordinator's independent recount, which uses red-team pass 2's library and shares no scorer code with the
project: it agrees on every count and interval (`outputs/seeds_2026-09-26/validation/RECOUNT_INDEPENDENT.json`).
VALIDATION reads: seq 17–19 (S2\*, S3\*, ENS3). Model judgments are not involved.

## Outcome

- **H1 (three-seed confirmation): `PARTIAL`.** All three seeds have D > 0, and the mean is 7.55 points (≥ 5.0).
  However, S3\*'s 90 % interval includes zero, so the rule's `CONFIRMED` (every lower bound > 0) is not met.
- **H2 (ENS3 vs CONT e41): `HARM`.** ENS3 − CONT = −8 parts at 2/km², 90 % [−15, −2]. It is `HARM` at every
  density: −9 [−18, −1] and −10 [−19, −2].
- **TEST consequence.** Because H2 is not `GAIN`, M\* stays CONT e41. TEST precondition 1 (`three_seed_confirmation`)
  is **not met**, so under the frozen TEST packet TEST stays sealed; the packet allows no partial unseal.

## Numbers (45 VALIDATION blocks, 265 common-class parts, K = 560 / 1,120 / 2,240)

| Arm | 0.5/km² | 1/km² | 2/km² | Supply | Δ vs AO at 2/km², 90 % (parts) |
|---|---|---|---|---|---|
| ADAF AO (seq 15) | 90 | 114 | 132 | 2,117 | — |
| CONT e41 (seq 16) | 125 | 146 | 157 | 9,407 | +25 [+14, +37] |
| S2\* = e44 | 119 | 144 | 163 | 3,648 | +31 [+19, +43] |
| S3\* = e43 | 90 | 113 | 136 | 6,024 | +4 [−3, +11] |
| ENS3 | 116 | 136 | 149 | 4,118 | +17 [+8, +27] |

- **S3\* vs AO at every density:** 0 [−15, +15], −1 [−11, +10] and +4 [−3, +11]. S3\* is indistinguishable from
  public ADAF.
- **S2\* vs AO:** +29, +30 and +31, each with a lower bound of at least 17.
- **Seed spread at 2/km²:** 27 parts, about 10 points.
- **Holdout selection** (TRAIN holdout, 804 windows, the rule of CONT's own packet):
  - S2\* e44: primary 0.764.
  - S3\* e43: primary 0.742.
  - CONT e41, for reference: 0.769.
  - The lowest holdout seed is also the lowest on VALIDATION. With n = 3 this is suggestive only.

## Checks and execution

- **Stage-B smoke:** PASS. CONT reproduced its seq-16 objects on both development blocks.
- **CONT guard:** 0 mismatched blocks of 45; the re-run CONT reproduced seq 16 on every VALIDATION block.
- **Locks:** the seq-15/16 lock verified unchanged before scoring.
- **Resources:** 4.12 GPU-h of the 5.0 h ceiling; no disk interruption.
- **Execution defect (fixed by a versioned addendum; no number affected).**
  - The chain's real reservation refused. `seed_validation.reserve()` never called `seed_train.configure(None)`,
    so its budget check read CONT's old ledger (1,562 s left) instead of this packet's (8,216 s left).
  - `addendum_reserve_configure.py` made that call and ran the frozen `reserve()` unchanged. The frozen files'
    hashes still verify.

## Interpretation

1. **CONT e41's VALIDATION result stands.** It was a single look for that checkpoint, which had been selected on the
   TRAIN holdout and SCREEN, not on VALIDATION. The result was reproduced by three clean red-team passes.
2. **The training recipe's advantage over ADAF is seed-sensitive.**
   - Two of three continuations clearly beat public ADAF AO; one ties it.
   - Three seeds gave an observed range of 27 parts (about 10 points) at 2/km². The block bootstrap does not
     include this training randomness. Three seeds give only an observed range, not a reliable general estimate of
     training uncertainty. (Corrected 2026-09-26 on the owner's qualification; the earlier text said "about ±5
     points".)
   - "The detector beats ADAF" is established for the delivered checkpoint on VALIDATION, and for the recipe in 2
     of 3 seeds.
3. **Averaging seed maps does not help.** ENS3 lost 11 parts that CONT found and gained 3.
4. **Forecasts** (decisions/2026-09-26-three-seed-confirmation.md): F1 and F5 were YES; F2, F3, F4, F6, F7 and F8
   were NO. Mean Brier 0.231. I overestimated replication (F3 0.65) and underestimated seed spread (F8: ≤ 10 parts
   at 0.55; actual 27).

## Corrections (2026-09-26, after the owner's review; numbers and verdicts unchanged)

- The 27-part range is an observed range across three seeds, not a general "±5 points" estimate of training
  uncertainty.
- Selecting among seeds on the TRAIN holdout is a candidate strategy that needs prospective evaluation.
- Before any further training: diagnose S3's loss and ENS3's harm from eligible development evidence, and propose
  one bounded intervention with a fixed comparison and success rule. VALIDATION per-part outcomes are not used for
  that design.

## Next

- **TEST stays sealed under the frozen packet.** Running it anyway, for the delivered checkpoint alone, would be a
  declared deviation from precondition 1 and is the owner's decision.
- **Discovery sweep R1** runs with M\* = CONT e41, as its packet fixes.
- **Recipe follow-ups** (not selected yet):
  - selecting among several seeds on the TRAIN holdout. This is a candidate strategy only, not an established fix.
    Holdout selection already chose each seed's epoch, so choosing among seeds adds a second selection stage. It
    would need prospective evaluation, and must not be a rule chosen retrospectively because S3 is now known to be
    weak (owner's qualification, 2026-09-26);
  - lower-variance continuation, such as checkpoint averaging.
  - Each needs its own fixed comparison and costs VALIDATION reads: 19 of 40 are used.
- **Storage:** archive the unselected epochs (S2 e40–43 and e45, S3 e40–42, e44 and e45) and both `last.pt` files
  through the verified Drive route.
