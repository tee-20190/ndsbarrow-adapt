# OPEN-TRANSFER-TEST freeze: one confirmatory read of OPEN-TRANSFER-1 on the sealed Dutch TEST provinces

Frozen 2026-10-22 by Opus (open-transfer-test session), sole writer of `work/open_transfer_test_2026-10-22/` and
`outputs/open_transfer_test_2026-10-22_PRIVATE/`.

**Authority.** Owner, in chat (`decisions/2026-10-22_nl_test_unseal.md`): "Yes. Unseal the Dutch TEST provinces
(Gelderland, Noord-Brabant, Limburg) for one frozen confirmatory read of D1 vs public ADAF, with the same rule as
OPEN-TRANSFER-1 frozen before any data is read." One read only; the result stands whatever it says; no re-reads and no
tuning.

**State at this freeze.** No AHN4 tile has been read for any TEST core, and no detector has run on any TEST core.
Before this commit: TEST record and core specs (`ott_labels.py`, labels and the AHN kaartblad index only, CPU), and
local synthetic pipeline tests (`test_ot_pipeline.py` ALL_PASS, `ot_score.py --selftest` PASS). The halo context of
OPEN-TRANSFER-1 EVAL mosaics reached over provincial borders in places; proposals there were dropped by its core rule
and never scored or looked at (OPEN-TRANSFER-1 FREEZE §3).

**Capability line:** generalisation. Confirmation of a zero-shot cross-border transfer on held-out provinces of the
same national register and survey.

## 1. Question

OPEN-TRANSFER-1 (freeze `721663442`, result `1eb6ae07d`) found D1 = BEATS_PUBLIC_ADAF on the Dutch EVAL provinces
(D1 − ADAF barrow +11.8 points at matched workload, simultaneous [+6.3, +16.9]; 653 records) and CONT =
LOSES_TO_PUBLIC_ADAF. **Does the same frozen rule, applied once to the sealed TEST provinces, return D1 =
BEATS_PUBLIC_ADAF?** The D1 verdict is the confirmatory question. CONT is scored under the same unchanged family and
its verdict is reported.

## 2. Data and licences (unchanged from OPEN-TRANSFER-1 §2)

- Register: RCE `landschapsatlas:grafheuvels`, CC0 1.0; the same snapshot file (sha256
  `09545f3a8ddb73bc612e70abfca7bfe4587e02325f7d258497a00184c0c5e055`, read from
  `outputs/open_transfer_2026-10-22_PRIVATE/src/`, read-only).
- Terrain: AHN4 DTM 0.5 m, PDOK ATOM kaartbladen COG, CC0 1.0; kaartbladindex sha256
  `07a90cca967cd2c8f86476f90d74c639944fe98aa7e9f2ee65d04f4500f9b127` (same file, read-only).
- Province boundaries: `outputs/benchmark_plan_2026-10-05/rev4_raw/nl_prov.json` sha256 `30c57316…8188` (same).
- Weights: the four OPEN-TRANSFER-1 arms, unchanged (§4).
- Record-level coordinates and proposals stay in `outputs/open_transfer_test_2026-10-22_PRIVATE/`; committed files hold
  aggregates and hashes only.

## 3. Records, cores and exclusions (`ott_labels.py`, frozen here)

- **Cohort:** the OPEN-TRANSFER-1 primary cohort, produced by `ot_labels.records()` imported unchanged (sha256 pinned,
  §9): NL-SCOPE-1 rule, 1,633 national primary records. **TEST_SEALED role = 955 records**, exactly as OPEN-TRANSFER-1
  defined it: Gelderland 745, Noord-Brabant 98, Limburg 112, and 0 in two further TEST_SEALED provinces
  [names redacted] that hold no primary record.
- **Cores (the mirror of OPEN-TRANSFER-1's EVAL core rule):** 5 km squares of the RD New grid (origin 0, 0) holding ≥ 1
  TEST_SEALED record. A core whose rectangle intersects an EVAL-role province (Drenthe, Overijssel, Utrecht,
  Noord-Holland, Friesland, Groningen, Flevoland) is dropped, as EVAL cores touching a sealed province were dropped
  there. Region = the province holding most of the core's TEST records (tie → name order). Halo context may cross a
  provincial or national border; proposals there are dropped by the core rule.
- **Exclusion, stated with counts:** 2 cores dropped for touching an EVAL province, holding **23 Gelderland records**.
  This leaves **69 cores and 932 candidate records**:

| Region | Cores | Candidate records | 60 m groups | AHN4 tiles |
|---|---:|---:|---:|---:|
| GE (Gelderland) | 38 | 722 | 422 | 88 |
| LI (Limburg) | 9 | 112 | 23 | 39 |
| NB (Noord-Brabant) | 22 | 98 | 46 | 62 |
| **Total** | **69** | **932** | 491 | 188 distinct |

- **Eligible records:** as OPEN-TRANSFER-1 (`ot_fetch.py sites`, DTM only, after the gap fill): in a core (half-open)
  and every DTM cell whose centre is within 10 m is finite. Eligibility is decided on terrain alone, before any
  detector runs; the eligible count, the evaluation area (finite core km²) and the pre-fill area are reported in
  RESULT_TEST.md with every exclusion.
- **Geographic separation from EVAL:** TEST cores do not intersect any EVAL-role province; EVAL cores did not intersect
  any TEST province. No record is in both roles.
- **Label limits:** as OPEN-TRANSFER-1 §3 (AMK register points, many symbolic circles; unmatched proposals are not false
  positives by definition). Clustered fields (largest GE core 119 records) are handled by the block bootstrap.

## 4. Arms (`pipeline/ARMS.json`, byte-identical to OPEN-TRANSFER-1's)

| Arm | Registry id | sha256 |
|---|---|---|
| D1 | `D1-F1_BARROW_L_2026092805` (deployable run 05) | `d9ece1fb023d2daf72ab0e5f789b5b89a27af73a28307e6260e4ee2fb5ac501e` |
| CONT | `CONT-e41` | `91091fc6d3b7640bd300c8f7820626ea4ccc5ea3ee8f9f89425000f958bdc383` |
| ADAF_barrow | `ADAF-barrow` (public) | `51b679faf27a6754a1525cec231a7ef1582430d2d6d9e547a39a42622166b216` |
| ADAF_AO | `ADAF-AO` (public) | `a2232cc15d5849d75c79c35af9b8b72cbd512766b880851c9c72493a36c79dbb` |

No arm has been trained on Dutch terrain or Dutch labels.

## 5. Pipeline (identical to OPEN-TRANSFER-1 §5)

`pipeline/` is a copy of `work/open_transfer_2026-10-22/pipeline/` with **path changes only**: the private output
folder (`open_transfer_test_2026-10-22_PRIVATE`), the pipeline folder, the freeze file name (`FREEZE_TEST.md`), the
Colab session name (`ottgpu`), the ledger path, and the `up` concurrency check (cap 4 per
`decisions/2026-10-22_colab_concurrency_4.md`). The diff is recorded in `PIPELINE_DIFF.txt`. Decoder, AHN gap fill
(`fill_gaps`, IDW within 20 m), core rule, thresholds, TF32/cudnn settings and Verden parity gate are unchanged.
Gates: Verden parity PASS on this VM by this `ot_infer.py`; `ot_infer.py freezecheck` records this file committed with
every pinned file unchanged.

## 6. Scoring (`ot_score.py`, run once; identical rule)

`ot_score.py` is OPEN-TRANSFER-1's scorer with the region list (GE, LI, NB), private folder and packet name changed
only. It imports the SCOT-TRANSFER-1 scorer functions unchanged (`scot_score.py` sha256 `00f1fa30…6dc7`).
- One-to-one maximum matching of proposal centroid to record point within **15 m** (sensitivity 10, 25 m); budgets 1, 2,
  5, 10 per km² of each region's evaluation area.
- Workloads: (a) **matched workload** (per region N_r = min(pool X, pool Y), pooled); (b) **5 /km²**.
- Pooling: matched records summed over regions / eligible records summed over regions.
- Cluster bootstrap: 2 km RD blocks holding ≥ 1 eligible record, stratified by region, **10,000 replicates, seed
  20261022**.
- **Primary family, unchanged: the same 8 contrasts.** For X ∈ {D1, CONT}: X − ADAF_barrow and X − ADAF_AO, each at
  matched workload and at 5 /km². Simultaneous intervals: Bonferroni over 8 (percentiles 0.3125 / 99.6875). CONT is not
  demoted; the family is exactly OPEN-TRANSFER-1's.
- **Verdict per primary arm (unchanged):** BEATS_PUBLIC_ADAF if all four of its simultaneous lower bounds > 0;
  LOSES_TO_PUBLIC_ADAF if any of its simultaneous upper bounds < 0; otherwise NOT_DISTINGUISHED.
- **Confirmation reading:** D1 = BEATS_PUBLIC_ADAF on TEST **confirms** OPEN-TRANSFER-1's D1 result. NOT_DISTINGUISHED
  or LOSES does **not confirm** it; the EVAL result then stays development-grade and is reported beside this one.
- Independent recount: `ot_recount.py` (scipy bipartite matching, does not import the scorer) on every arm × region ×
  budget; AGREE required for the result to be reported as checked, DISAGREE is reported as a defect, not re-scored.
- Descriptive (no verdict): ordinary 95 % intervals, D1 − CONT, per-region families, region-equal-weight mean, recall
  at every budget, groups found, recall by polygon area, reach 10 / 25 m.

## 7. What changes

- **D1 BEATS (confirmed):** D1's documented scope in DETECTOR_HISTORY / registry becomes "NW European Pleistocene sand
  lowland barrows: Lower Saxony and the Netherlands (EVAL + held-out TEST provinces)", confirmed on held-out terrain.
  Lower Saxony and English operating choices are unchanged.
- **D1 not confirmed:** the EVAL result stays development-grade; no operating change.
- CONT: reported under the same family; no operating change in any branch (it is not an operating choice for NL).

## 8. Forecasts (Opus, before any TEST output)

| Event | P |
|---|---:|
| D1 BEATS_PUBLIC_ADAF (confirms) | 0.72 |
| D1 NOT_DISTINGUISHED | 0.24 |
| D1 LOSES_TO_PUBLIC_ADAF | 0.04 |
| CONT LOSES_TO_PUBLIC_ADAF | 0.55 |
| CONT BEATS_PUBLIC_ADAF | 0.04 |
| D1 − ADAF barrow at matched workload, point > +5 points | 0.65 |
| D1 − ADAF barrow at matched workload, point ≥ EVAL's +11.8 | 0.30 |
| ADAF barrow pooled recall at 5 /km² ≥ 0.40 | 0.50 |

Basis: EVAL simultaneous lower bound +6.3 with a winner's-curse discount; Veluwe and Kempen are Pleistocene sand
heath/forest like the EVAL provinces, but Gelderland carries 77 % of the records in fewer, denser fields, Limburg's 112
records sit in few groups (23), and the matched-workload contrast against ADAF barrow is the binding one.

## 9. Pins

Pinned file hashes (filled immediately before the freeze commit):

| File | sha256 |
|---|---|
| `work/open_transfer_test_2026-10-22/ott_labels.py` | `6d24ad926213b57173a09a7d1f6a8c7c4382fe049745f109e77e9e966c3053a6` |
| `work/open_transfer_test_2026-10-22/REGIONS_SUMMARY_TEST.json` | `d3a44a40514c212ebda2334e444684d0322a308f8faba880d1939a1696fc78af` |
| `work/open_transfer_test_2026-10-22/ot_score.py` | `ffcb7eb07b231b06f074d0c98a9cc6373639ac82d59ae22e2c8cebfc8457dcd2` |
| `work/open_transfer_test_2026-10-22/ot_recount.py` | `be7547f4529b6125c32d294d2f38055053399ef424674826ce10a9616faa85cd` |
| `work/open_transfer_test_2026-10-22/PIPELINE_DIFF.txt` | `5a5ec94c50ec69ad8f922fa4a46f46349228e5e07cfa8328b8b2fae9631f3f90` |
| `work/open_transfer_test_2026-10-22/pipeline/ot_common.py` | `f7e0e4b9a0824755f4c183d556c8c9f3e667a0a534761eead614bcae74740d54` |
| `work/open_transfer_test_2026-10-22/pipeline/ot_fetch.py` | `70164256f7800c6aba50b6bc461275215422ddfafef5e7ab73e6eac68dee8b1f` |
| `work/open_transfer_test_2026-10-22/pipeline/ot_infer.py` | `c51b075b3d44b8125b4b2b8663a67ce100d8f44cb0f0f87b71546d6877a73595` |
| `work/open_transfer_test_2026-10-22/pipeline/ARMS.json` | `6f3648d84d5add27ee6ad01b3ea113cc1f9c1c4e88edfd8360356fd9aa9ffb9d` |
| `work/open_transfer_test_2026-10-22/pipeline/ot_vm_job.py` | `9b6b929e18159415f82420ba4e1ea592e8d8750c5bb713c1a956b721343ee892` |
| `work/open_transfer_test_2026-10-22/pipeline/ot_bundle.py` | `0957ff0292f4a00dab7579f4f94a1cdc74c0c71c385a7dc9374584d3b572a321` |
| `work/open_transfer_test_2026-10-22/pipeline/ot_vm_setup.py` | `ef2e229b88fdd5a8d80984b67b42440a7208bd27c358b2166b5d7343e5e5a5b1` |
| `work/open_transfer_test_2026-10-22/pipeline/ot_vm.sh` | `277eec1964636babc0fe039e5450fc42d58a0ef1203874871ff3222ad47de26c` |
| `work/open_transfer_test_2026-10-22/pipeline/test_ot_pipeline.py` | `4d7df4d0a9e585c59056d82f240e82c3c38eead95e4a03dc6f7e8784d69d5f90` |
| `work/open_transfer_2026-10-22/ot_labels.py` | `e7cd37700107b8a95be0d9f4c77e9562f184781c25d2e82b78097b1089c77d7f` |
| `work/scot_transfer_2026-10-19/scot_score.py` | `00f1fa3098a6d70bda5ab8a2ab340320589f4ebf668556dca8342183a6756dc7` |
| `work/open_transfer_2026-10-22/pipeline/ARMS.json` | `6f3648d84d5add27ee6ad01b3ea113cc1f9c1c4e88edfd8360356fd9aa9ffb9d` |
| `outputs/open_transfer_2026-10-22_PRIVATE/src/grafheuvels.geojson` | `09545f3a8ddb73bc612e70abfca7bfe4587e02325f7d258497a00184c0c5e055` |
| `outputs/open_transfer_2026-10-22_PRIVATE/src/kaartbladindex.json` | `07a90cca967cd2c8f86476f90d74c639944fe98aa7e9f2ee65d04f4500f9b127` |
| `outputs/open_transfer_test_2026-10-22_PRIVATE/regions/GE/REGION.json` | `06cf755b6498f899990d659937da3058d1cea6700b3ec2dbc63424c8c7250332` |
| `outputs/open_transfer_test_2026-10-22_PRIVATE/regions/GE/CANDIDATE_SITES.json` | `bf1171769963fc027f70e6f2b45c2fd90d518632b2cf88fa64e2f66c049b212b` |
| `outputs/open_transfer_test_2026-10-22_PRIVATE/regions/LI/REGION.json` | `e94cb9743dcbf5c5c50c5157c500165d4c2cb9d001799b19698449befd279e17` |
| `outputs/open_transfer_test_2026-10-22_PRIVATE/regions/LI/CANDIDATE_SITES.json` | `8adf8258b135614e8a194aa783ae7c4fa8cf33d89f833bfc782fbb01ac13b5a6` |
| `outputs/open_transfer_test_2026-10-22_PRIVATE/regions/NB/REGION.json` | `0b8c59ed2ca05d47b5c2051508ab961d7c98869cbf8a9b933bbf8f0a9a8958f6` |
| `outputs/open_transfer_test_2026-10-22_PRIVATE/regions/NB/CANDIDATE_SITES.json` | `4ffc22996d01b7312ff192d8fed86b3499de38285d12578a2462d88dc1280645` |

Checkpoints: the four sha256 in §4 (checked against ARMS.json at bundle build and on the VM after reassembly).

## 10. Resources and stop rules

- One Colab G4 VM, session `ottgpu`, started by this session only (cap 4 VMs total; CU pause threshold 1,700; balance
  2,237 CU at 22:34Z with 2 active assignments). Ceiling 60 CU; whole-job cap 8 h. Every VM step runs under
  `verify/v3/storage_guard.py` on the VM volume (declared peaks per step); `verify/v3/archive_gate_v2.py check` PASS
  locally before launch. Local disk: receipts and proposals only (< 1 GB).
- Every promised results tar is sha256-verified locally before the VM stops.
- Verden parity FAIL → stop and diagnose. Technical failure before scoring → rerun unchanged with an erratum; two
  failures of the same step → stop and diagnose. No re-read after scoring.
