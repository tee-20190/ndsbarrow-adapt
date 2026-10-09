# Evaluator v0.1: role-based record treatment (spec, 2026-10-07)

Source: GPT-6 Pro Round 4, R4-6 (`work/gpt6_pro_collab_2026-10-07/ROUND4_REPLY.md:360-425`), adopted in
`ROUND4_REVIEW.md` items 8 and 5. Implementation: `eval_v01.py`. Tests: `tests/test_eval_v01.py`. Real-data checks:
`run_checks.py` → `CHECKS.json`. Void table: `VOIDS.json`. Frozen in `FREEZE.md` / `FREEZE.json`.

## 1. Status and use

| Use | Role of v0.1 |
|---|---|
| Stage 2 controls (`work/stage2_d1_controls_2026-10-07/`) | Prespecified **sensitivity**. The exact Stage 1 scorer stays primary. |
| Oldenburg transfer check (`work/oldenburg_transfer_2026-10-07/`) | Prespecified **sensitivity**. The exact Stage 1 scorer stays primary. |
| Experiments frozen after this file | **Primary**, unless their packet states otherwise. |

v0.1 does not change any completed result. Stage 1's point-only gate is unchanged.

## 2. Record roles

Every register record (district snapshot plus edge supplement) gets exactly one role. Only `void` changes scoring.

| Record type | Role | Treatment |
|---|---|---|
| Eligible individual barrow (an entity of the frozen individual manifest) | `target` | The entity. In N, matched one-to-one. |
| Archaeologically defined cemetery (`objekttyp` Grabhügelfeld) | `container` | Complex metadata (section 5). Not an exclusion. |
| Legal protection polygon (`objekttyp` Grabungsschutzgebiet) | `legal` | Legal metadata. Not an exclusion. Not a complex. Never defines membership. |
| Genuine evaluation void | `void` | Explicit fixed geometry from the void table (section 3). Never taken from the register. |
| Any other record, in valid terrain | `other` | No effect. A proposal on it consumes a slot. This includes register barrows that are not eligible entities. |

**No rule removes proposals by proximity to targets, or to any record, before K.** The only proposal filter is the
void filter.

## 3. Voids

A void entry has `void_id`, `reason_code` (`NO_TERRAIN_DATA`, `OUTSIDE_LICENSED_COVERAGE` or `OTHER_DOCUMENTED`),
a free-text `reason`, `derived_from` and a GeoJSON geometry.

- **Defined before any v0.1 score** of the experiment, in a hashed void table recorded in that experiment's freeze.
- **Never derived from targets, labels or detector output.** `validate_voids` refuses a `derived_from` naming targets,
  entities, references, barrows, proposals, detectors, scores or labels. It also refuses unknown codes, empty reasons,
  duplicate ids and empty geometry.
- **Applied consistently, in this order:**
  1. **Area:** scored area = union of manifest cores − union of voids. K = floor(b × scored km² + 1e-9).
  2. **Proposals:** a proposal whose point is covered by a void (boundary included) is removed before NMS, ranking and
     K. It is neither a hit nor a slot.
  3. **References:** a target whose 15 m reach zone has zero area in the scored area is unreachable. It is removed
     from N and from matching. A partly covered target stays.
- Removal never depends on whether any proposal matches.

**Current tables (`VOIDS.json`): Verden and Oldenburg have no void.** The staged-terrain void gates measured void share
0.0 in every scored window. One Oldenburg positive core is a 0.43 m × 419 m sliver with no 1 m pixel centre; its share
is undefined, and it is not a void.

## 4. Barrow endpoint (unchanged from v0)

For each region, the endpoint uses the frozen individual manifest, `region_score.setup` (sha256 `b3337741…`) and its
15 m footprint reach tree.

- **RAW:** every eligible in-core proposal in `rank_key` order: score, mean probability, area descending, proposal id
  ascending.
- **NMS60:** the frozen `region_score.ranked_pool`, a 60 m greedy NMS over the eligible in-core proposals.
- **Matching:** one-to-one maximum-cardinality matching (scipy) of the top K to the reachable targets.
- **Budgets:** b ∈ {0.5, 1, 2} per km².
- **Reported per decoder:**
  - supply;
  - C = whole-pool maximum matching;
  - G_pool = N − C;
  - per b: K, matches M, recall M/N, ceiling min(1, K/N), matches per 100 slots, and
    G_ranking(b) = min(K, C) − M.

## 5. Complex endpoint: member-supported cemetery recovery

> A cemetery is recovered when a proposal in the top K matches at least one of its eligible, individually located
> member barrows.

**Candidates.** Every `container` record, from the district snapshot and the edge supplement.

**Valid extent.** A container has a valid extent iff its geometry (after `make_valid`) has an area of at least 200 m².

- Smaller polygons are treated as lacking an extent. Oldenburg's 6 m discs are 112 m².
- A single 16 m barrow alone covers about 200 m², so no real cemetery boundary is excluded by this threshold.
- A container without valid extent can still have members through a site-number link. It never has members through
  containment.

**Membership.** Each reachable target is assigned to at most one complex, without consulting detector output.

1. **Register site-number link (source link).** The target and a container record carry the same register site key
   (Gemarkung, FStNr), parsed from `siteName`, and exactly one container carries that key. The Denkmalatlas records
   have no explicit member-to-complex field (`REGISTER_SEMANTICS.json`; property list checked), so the shared site key
   is the only source-provided link. A key carried by two or more containers is ambiguous and is ignored.
2. **Frozen containment rule** (only when step 1 does not assign):
   - the target's footprint centroid is covered by a container polygon with a valid extent;
   - if several qualify, the smallest-area polygon wins;
   - ties go to the lower record id.
3. **Otherwise the target stays unassigned.** It is not a solitary-barrow complex and not part of an extra cemetery.
   - Legal polygons never assign membership.
   - Oldenburg's 10.3 km² protection area is not a complex.
   - Its 26 barrows outside any cemetery stay unassigned unless a site link assigns them.

**Availability.** A complex is available iff it has at least one eligible member. Unavailable complexes are reported,
not scored.

- That covers a polygon with no eligible member, a disc without links, or a cemetery whose members lie outside the
  scored terrain.
- Cemeteries outside the "2-20 recorded members" description are not discarded. They are audited under the same rule.

**Scoring.**

- A proposal-complex edge exists iff the proposal is within reach of any eligible member of an available complex.
- K and the ranked list are those of the barrow endpoint, so proposals still consume the workload allowance.
- The score is a one-to-one maximum matching of the top-K proposals to the available complexes. Consequently:
  - a complex scores at most one hit;
  - one proposal recovers at most one complex;
  - repeated detections of member mounds add nothing.
- **Reported:** available complexes, C, complexes recovered and complex recall at each b and decoder, and the
  membership diagnostics.
- **Not adopted:** the weaker "proposal inside the cemetery polygon" endpoint. It would need polygon areas and a
  spatially uniform proposal baseline.

**Inventory (register and manifest only; `CHECKS.json`):**

| | Verden | Oldenburg |
|---|---:|---:|
| Container records (snapshot + supplement) | 73 | 101 |
| With valid extent / without | 71 / 2 | 49 / 52 |
| Available complexes (≥ 1 eligible member) | 57 | 93 |
| Available without extent (site-link members only) | 0 | 45 |
| Unavailable | 16 | 8 |
| Members by site link / containment | 55 / 376 | 91 / 572 |
| Containment overlaps resolved by smallest area | 0 | 3 |
| Eligible barrows left unassigned | 297 of 728 | 296 of 959 |
| Members per available complex (min / median / max) | 2 / 7 / 20 | 1 / 2 / 109 |
| Legal polygons (never complexes) | 0 | 1 |

## 6. Expected relation to the IGNORE readings, and the check

- **Reading (i), historical: v0.1 equals it.** Reading (i) applies no IGNORE inside positive cores; this is the frozen
  implementation. With an empty void table and container and legal records as metadata, v0.1 scores the same
  proposals, area, K and entities.
  - The background squares of the existing manifests were sampled more than 30 m from register records. That is a
    property of the manifest, shared by every system, and v0.1 inherits it unchanged.
  - **New manifests built for v0.1-primary experiments must not select scored area by proximity to any register
    record.** Only voids may remove terrain.
- **Reading (iii), explicit precedence: v0.1 differs from it.** Reading (iii) also removes, before K, every proposal
  within 30 m of a non-container, non-target record outside the targets' 15 m reach zones, and removes that area.
  v0.1 treats those records as `other`, so proposals there consume slots. Expected: v0.1 ≤ (iii) in recall for a given
  system, with a small difference.
- **Verified numerically on Verden ADAF barrow** (`CHECKS.json`):

  | Decoder | b | v0.1 | Reading (i) | Reading (iii) | v0.1 − (iii), points |
  |---|---|---:|---:|---:|---:|
  | RAW | 0.5 | 0.16896 | 0.16896 | 0.17308 | −0.41 |
  | RAW | 1 | 0.29121 | 0.29121 | 0.29121 | 0.00 |
  | RAW | 2 | 0.39698 | 0.39698 | 0.39698 | 0.00 |
  | NMS60 | 0.5 | 0.16346 | 0.16346 | 0.16758 | −0.41 |
  | NMS60 | 1 | 0.25275 | 0.25275 | 0.25549 | −0.27 |
  | NMS60 | 2 | 0.29258 | 0.29258 | 0.29258 | 0.00 |

  - v0.1 equals (i) exactly at every cell.
  - K: v0.1 211 / 422 / 845; (iii) 211 / 422 / 844.
- **Exact Stage 1 reproduction.** v0.1 with no voids reproduces `work/stage1_d1_v0_2026-10-07/RESULT.json` for ADAF
  barrow and all ten D1 runs on Verden:
  - 440 of 440 values identical (float equality): supply, K, matches, recall, ceiling, matches per 100 slots, slots
    filled and primary, for both decoders and every b.
- **Oldenburg ADAF barrow** (existing frozen proposals, sha256 `21421a9c…`) equals BENCH_V0's v0 values exactly, for
  both decoders, every b and supply.

## 7. Limitations

- **Site-number links.** They are register find-spot identifiers, not membership statements. They are used because
  they are the only source-provided key. 45 of Oldenburg's 93 available complexes rest on them alone.
- **Register incompleteness.**
  - Stated mound counts exceed the individually recorded barrows (Oldenburg 1,534 vs 540).
  - A detector that finds unrecorded members of a known cemetery is not credited at the barrow level.
  - At the complex level such a detector is credited only through a recorded member.
- **Weaker evidence for large complexes.** A complex with many members offers many routes to one hit, so complex
  recovery is weaker evidence of mound detection for large complexes. Report members per complex alongside.
- **Inherited background sampling.** The background sampling of the existing manifests (section 6) favours every system
  equally. It is a disclosed inheritance, not a v0.1 rule.
