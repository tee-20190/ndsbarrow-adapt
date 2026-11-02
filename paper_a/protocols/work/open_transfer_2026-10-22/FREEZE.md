# OPEN-TRANSFER-1 freeze: do the project's barrow detectors transfer zero-shot to the Netherlands, against an open national register?

Frozen 2026-10-22 by Opus (open-transfer session, line L6 of `work/COORDINATION_2026-10-22.md`). Sole writer of
`work/open_transfer_2026-10-22/` and `outputs/open_transfer_2026-10-22_PRIVATE/`.

**No detector output exists for any Dutch terrain at this freeze.** Before it: register and licence checks, the label
census and region specs (`ot_labels.py`, CPU, no terrain), AHN4 tile download and mosaicking on the VM
(detector-free), site validity (DTM only), synthetic pipeline tests and the Verden parity check (Verden terrain only).
`ot_infer.py run` refuses until `freezecheck` (Windows git) records this file as committed.

**Capability line:** generalisation. Zero-shot transfer to a new country whose monument register is open, with no
local custodian, no local labels in any arm's training, and a sealed national TEST part kept for later.

## 1. Question

D1 (public ADAF barrow + recipe F1 on Lower Saxony register labels) beats public ADAF barrow on three Lower Saxony
districts (Verden, Oldenburg, Göttingen). CONT e41 (England) beats ADAF in Gloucestershire and Hampshire but not in
Wiltshire or the South-West. ENG-ADAPT-1 found that the D1 gain needs local labels. **Does either project detector
beat the public ADAF models on Dutch barrows, with no Dutch training, at equal workload over whole landscapes?** The
Dutch Pleistocene sand landscapes (Drenthe, Overijssel, Utrecht Heuvelrug, Gooi) are close kin of the northern Lower
Saxony heath where D1 was trained; the Netherlands is a different survey (AHN4), register and country.

## 2. Data and licences (verified 2026-10-22)

- **Register:** RCE (Rijksdienst voor het Cultureel Erfgoed) layer `landschapsatlas:grafheuvels`, WFS
  `https://services.rce.geovoorziening.nl/landschapsatlas/wfs`; data.overheid.nl dataset 81525, **CC0 1.0**. Snapshot
  2026-10-22T20:07Z: 2,898 features (hits before = after = returned), sha256 `09545f3a…c055`, 4,560,857 B (private).
- **Terrain:** AHN4 DTM 0.5 m (2020-2022), PDOK ATOM kaartbladen (5 × 6.25 km COGs, EPSG:28992), **CC0 1.0**
  (`<rights>` of the ATOM feed). Index `kaartbladindex.json` sha256 `07a90cca…b127`. No login.
- **Province boundaries:** cached 2024 generalised CBS provinces (`outputs/benchmark_plan_2026-10-05/rev4_raw/nl_prov.json`,
  sha256 `30c57316…8188`, as NL-SCOPE-1).
- **Weights:** ADAF barrow and AO CC BY-SA 4.0 (Zenodo 15848663); D1 and CONT project checkpoints.
- Record-level coordinates and proposals stay in `outputs/open_transfer_2026-10-22_PRIVATE/`; committed files hold
  aggregates and hashes only (the register is public, but the project keeps one rule for all proposals).

## 3. Labels, split and terrain (`ot_labels.py`, frozen before any inference)

- **Primary cohort** = NL-SCOPE-1's fixed rule, unchanged: `bron = AMK`, `inwinning` empty or AMK (the 394 AHN2-derived
  and 1 AHN-derived records are excluded: they are elevation-derived potential mounds, not independent labels), empty
  `opmerking`, polygon area ≥ 16π m² (equivalent diameter ≥ 8 m). Two invalid geometries dropped; NL-SCOPE-1's
  duplicate rule (centroid ≤ 2 m or IoU ≥ 0.5 with an earlier id) removes 9. **1,633 records.** Point = polygon centroid.
- **Split by province (fixed here):**
  - **EVAL** (scored once in this packet): Drenthe, Overijssel, Utrecht, Noord-Holland (Friesland, Groningen,
    Flevoland would be EVAL but hold no primary record). 678 records.
  - **TEST_SEALED** (untouched; reserved for confirming whatever this packet or a successor selects): Gelderland,
    Noord-Brabant, Limburg and two further provinces [names redacted; no primary record]. **955 records.** No AHN tile is read for a core in these provinces,
    no detector runs there, and no TEST record is scored. Halo context of an EVAL core may extend over a provincial
    border; proposals there are dropped (core rule).
- **Cores:** 5 km squares of the RD New grid (origin 0, 0) holding ≥ 1 EVAL record. A core whose rectangle intersects a
  TEST_SEALED province is dropped (2 cores, 25 records). Region = the province holding most of the core's EVAL
  records. **45 cores**: DR 12, OV 21, UT 8, NH 4. Mosaic = core ± 256 m halo from every intersecting AHN4 kaartblad
  (133 tiles, 50.2 GB).
- **Eligible records** (`ot_fetch.py sites`, DTM only): in a core (half-open) and every DTM cell whose centre is within
  10 m is finite. AHN DTM leaves no-data under buildings and water; an eligible barrow has complete terrain within
  10 m. Counts and the measured evaluation area are in section 9.
- **Label limits.** AMK barrows are legally protected, surveyed mounds; many are restored, some levelled. The layer
  has no visibility field; 1,324 of 1,633 primary geometries are regular circles (symbolic extent), so polygon area is
  approximate. Recall against these labels is not a ground-truth detection rate, and unmatched proposals are not
  false positives by definition. Both apply equally to every arm.

## 4. Arms (`pipeline/ARMS.json`, sha256-checked on load)

| Arm | Registry id | sha256 | Training terrain |
|---|---|---|---|
| D1 | `D1-F1_BARROW_L_2026092805` (deployable run 05) | `d9ece1fb…501e` | Lower Saxony FIT districts |
| CONT | `CONT-e41` | `91091fc6…c383` | England (EA LiDAR, NHLE) |
| ADAF_barrow | `ADAF-barrow` (public) | `51b679fa…b216` | ADAF training set |
| ADAF_AO | `ADAF-AO` (public) | `a2232cc1…dbb` | ADAF training set |

No arm has seen Dutch terrain or Dutch labels.

## 5. Pipeline (identical for every arm)

`pipeline/` is SCOT-TRANSFER-1's tested pipeline ported with path changes only, except `ot_fetch.py` (AHN source:
PDOK URL, EPSG:28992, nodata float32 max, Content-MD5 check; plus a guard that no finite value ≥ 1e30 survives the
mask). Decoder unchanged: `tta_run.objects_multi` (0.5 m; 1,024 px tiles, 64 px halo; SLRM radius 20, normalised
±0.5, NaN → 0; 3-channel repeat; sigmoid channel 1; tile max; threshold 0.5, area > 30 m², roundness > 0.4; no NMS).
TF32 and cudnn.benchmark off. Proposals kept iff centroid in core. Gates: Verden parity PASS on this VM by this
`ot_infer.py`; this file committed with every pinned file unchanged (`ot_infer.py freezecheck`). AHN no-data cells
left after the gap fill below reach the decoder as NaN → 0, as in every earlier packet.

**AHN gap fill (fixed before inference; found on the DR fetch, DTM only).** AHN4 DTM 0.5 m is not interpolated: a
cell with no ground point is no-data. The first DR mosaics had 4.5-9.4 % no-data in 280k-420k components per 5.5 km
mosaic, most of 1-50 cells (forest speckle), and only 76 of 284 DR barrows had a complete 10 m disk (median 3 no-data
cells of 1,254). The English EA and Lower Saxony DGM1 products every arm was trained or tested on are gap-interpolated,
so the decoder would otherwise see a speckle artefact no arm has met. `ot_fetch.fill_gaps`: after mosaicking, every
no-data cell within 20 m (40 px) of a measured cell is filled by GDAL inverse-distance interpolation
(`rasterio.fill.fillnodata`, no smoothing); measured cells are unchanged; farther cells (open water, large building
blocks) stay NaN. Site validity and the evaluation area use the filled mosaic; `raw_valid_core_km2` reports the
pre-fill area. The pre-fill DR receipts are kept in `outputs/open_transfer_2026-10-22_PRIVATE/superseded_prefill/`.
The same filled mosaic goes to every arm.

## 6. Scoring (`ot_score.py`, run once)

- The SCOT-TRANSFER-1 scorer's functions, imported unchanged (`scot_score.py` sha256 `00f1fa30…6dc7`): frozen
  `rank_key`; one-to-one maximum matching of proposal centroid to record point within **15 m** (sensitivity 10, 25 m);
  budgets 1, 2, 5, 10 proposals per km² of each region's evaluation area.
- **Workloads:** (a) **matched workload**: per region N_r = min(pool X, pool Y), both arms take their top N_r, pooled;
  (b) **5 /km²** fixed budget.
- **Pooling:** recall = matched records summed over regions / eligible records summed over regions.
- **Cluster bootstrap:** units = 2 km RD blocks holding ≥ 1 eligible record, stratified by region, 10,000 replicates,
  seed 20261022.
- **Primary family:** for X ∈ {D1, CONT}: X − ADAF_barrow and X − ADAF_AO, each at matched workload and at 5 /km²
  (8 contrasts). Simultaneous intervals: Bonferroni over 8 (percentiles 0.3125 / 99.6875).
- **Verdict per primary arm (frozen rule):** **BEATS_PUBLIC_ADAF** if all four simultaneous lower bounds > 0;
  **LOSES_TO_PUBLIC_ADAF** if any simultaneous upper bound < 0; otherwise **NOT_DISTINGUISHED**.
- **Descriptive (no verdict):** each contrast's ordinary 95 % interval; D1 − CONT; per-region families; the
  region-equal-weight mean; recall at every budget and over the whole pool; precision lower bounds; groups found
  (60 m groups with ≥ 1 match); recall by polygon area (< 200, 200-500, ≥ 500 m²); reach 10 / 25 m.

## 7. What changes

- **D1 BEATS:** the Lower Saxony adaptation transfers zero-shot to a second country's register. D1's documented
  scope widens to "NW European Pleistocene sand lowland barrows (Lower Saxony; Dutch EVAL provinces)"; the claim is
  development-grade until the sealed TEST provinces confirm it. DETECTOR_HISTORY / registry are updated.
- **CONT BEATS:** the English model transfers; same treatment.
- **NOT_DISTINGUISHED / LOSES:** no operating change; zero-shot transfer is not shown, and the next step for the
  Netherlands is local adaptation on EVAL labels, confirmed on the sealed TEST provinces.
- Lower Saxony and English operating choices are untouched in every branch.

## 8. Forecasts (Opus, before any output)

| Event | P |
|---|---:|
| D1 BEATS_PUBLIC_ADAF | 0.25 |
| D1 NOT_DISTINGUISHED | 0.55 |
| D1 LOSES | 0.20 |
| CONT BEATS_PUBLIC_ADAF | 0.10 |
| CONT LOSES | 0.35 |
| D1 − ADAF_barrow at 5 /km², point estimate > 0 | 0.65 |
| D1 − ADAF_barrow at 5 /km², ordinary 95 % lower bound > 0 | 0.40 |
| ADAF_barrow pooled recall at 5 /km² ≥ 0.20 | 0.50 |

Basis: D1's gain over ADAF barrow was +3.9 to +6.3 points at 0.5 /km² on three Lower Saxony districts, and the Dutch
EVAL landscapes resemble its training heath; but BEATS needs all four contrasts including AO and matched workload, the
survey and register differ, and 653 records in heavy clusters (largest cores hold 60-70) give wide intervals. CONT
lost or tied against ADAF outside its best English counties.

## 9. Measured inputs and pins

| Region | Cores | Tiles | Eval area km² (finite core) | Candidates | Eligible | Min tile coverage | SITES.json sha256 |
|---|---:|---:|---:|---:|---:|---:|---|
| DR | 12 | 37 | 298.127 | 284 | 284 | 1.0 | `f330d80aab0d99af…` |
| NH | 4 | 14 | 95.517 | 33 | 33 | 1.0 | `935064d7750ad971…` |
| OV | 21 | 63 | 480.506 | 238 | 238 | 0.953556 | `38b480dab6ceaead…` |
| UT | 8 | 23 | 199.516 | 98 | 98 | 1.0 | `11ecb0ecccee444f…` |
| **Total** | | | **1073.665** | 653 | **653** | | |

| Pinned file | sha256 |
|---|---|
| `work/open_transfer_2026-10-22/ot_score.py` | `daaa134b39932fb7f47fc25122829b780692466af64ae14c8297b5682a0bbb0a` |
| `work/open_transfer_2026-10-22/ot_labels.py` | `e7cd37700107b8a95be0d9f4c77e9562f184781c25d2e82b78097b1089c77d7f` |
| `work/open_transfer_2026-10-22/REGIONS_SUMMARY.json` | `bb739c8e9255110a3b96011075c957d7b401ba0ed55197afa7a4d4d9d902d64a` |
| `work/scot_transfer_2026-10-19/scot_score.py` | `00f1fa3098a6d70bda5ab8a2ab340320589f4ebf668556dca8342183a6756dc7` |
| `work/open_transfer_2026-10-22/pipeline/ot_common.py` | `68eafb4a2416da2065f51f407f22e6db4e2cfc011024610ef052f81ac9c19deb` |
| `work/open_transfer_2026-10-22/pipeline/ot_fetch.py` | `ab2ff378a5c2790cfb9284ba0fb85df077e34d2b40aa2fea8415abf3b09afef0` |
| `work/open_transfer_2026-10-22/pipeline/ot_infer.py` | `e0a4be3cac1823fc8babfc2b574109618324256d3cb453ccf7f013c104b662bd` |
| `work/open_transfer_2026-10-22/pipeline/ARMS.json` | `6f3648d84d5add27ee6ad01b3ea113cc1f9c1c4e88edfd8360356fd9aa9ffb9d` |
| `work/open_transfer_2026-10-22/pipeline/ot_vm_job.py` | `94ba4bdc892d77cfeb92fb9e7fe563add33950066432bf66f9feb311cbcda217` |
| `work/open_transfer_2026-10-22/pipeline/ot_bundle.py` | `5337a7e9aa9d42d758e9ea97ea907ac7c291fedde994ccb5180013c5f6796a52` |

## 10. Resources and stop rules

- One Colab G4 VM, session `otgpu` (slot 3 in the coordination file; 2 assignments active at `up`). Ceiling 60 CU;
  whole-job cap 8 h. Every VM step runs under `verify/v3/storage_guard.py` on the VM volume (declared peaks per step);
  `verify/v3/archive_gate_v2.py check` PASS locally before launch. Local disk: receipts and proposals only (< 1 GB).
- Results tars are sha256-verified locally before the VM stops. Detached jobs kept alive by `vm_keepalive.sh`.
- Verden parity FAIL → stop and diagnose. Technical failure before scoring → rerun unchanged with an erratum. Two
  failures of the same step → stop and diagnose.
