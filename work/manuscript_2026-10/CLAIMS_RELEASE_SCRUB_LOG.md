# CLAIMS_RELEASE scrub log (v10 release ledger, 2026-10-13)

Output: `work/manuscript_2026-10/CLAIMS_RELEASE.md` in this repository (SHA-256
`1e1b3a21c381456b7e345b7a790a64ac238397622d65939fcf901c085592aaa4`, 72,593 bytes). It replaces the ledger released on
2026-10-09, which covered draft v7 (215 rows; SHA-256 prefix `882e6f5ff71d2874`).

Inputs (author's working repository, not released): the internal ledger `work/manuscript_2026-10/CLAIMS.md` (SHA-256
`414a66163144ddf896324e03afc863a39436dbb9215309d706b6c1c222112a71`, ledger revision v10 of 2026-10-10) and
`work/manuscript_2026-10/DRAFT_v10.md` (`eaf10e01beea3dc5e0f047c798a163f72cd20f50b79e1b01d4ba2322bb20e259`), used only
for the set of cited [Cnnn] IDs. Prepared by an AI-assisted session under the author's direction. No file named
`_PRIVATE`, `SEALED` or `RESERVED` and no result file was opened; every edit was decided from the ledger text alone.

## Method

The same method as the 2026-10-09 release scrub. Its build script was recovered and re-run on its recorded inputs
(internal ledger at commit `9e9cbbdb`, SHA-256 prefix `8ad2d525f57a9717`; `DRAFT_v7.md` prefix `d45a4f6457819a95`): the output
reproduced the logged hash prefix `0806e395b706b787` exactly. The ledger released on 2026-10-09 differed from that output
only by two later edits: the C143 receipt replacement below and the C212 copyright-year erratum, which the v10 internal
ledger already carries. Steps:

1. Cited set = every Cnnn inside a `[...]` tag of `DRAFT_v10.md`.
2. Release only ledger rows whose ID is cited; omit the rest.
3. Exact-string substitutions on released rows and the footer: the nine 2026-10-09 substitutions, the 2026-10-09
   C143 follow-up, and six new substitutions of the same kinds for text added in v10. Each listed string matched
   exactly once.
4. Release header (scope updated to v10) and a release note before "Numbers that appear in the draft".

No claim, value, status, commit hash or non-private receipt path was changed. The build script
(`work/preprint_2026-10-13/build/make_claims_release_v10.py`) and the scan script (`scan_claims_release.py`, same
folder) are kept in the author's working repository; like the 2026-10-09 scripts they are not released, because the
substitution table quotes the internal text.

## Row counts

| Item | Count |
|---|---|
| Claim rows in internal ledger (C001-C230) | 230 |
| Distinct [Cnnn] IDs cited in DRAFT_v10.md | 225 |
| Rows released | 225 (ID list equals the cited set exactly; checked in script) |
| Rows omitted (not cited in v10) | 5: C139, C146, C166, C167, C168 (the same five as in the v7 release) |
| Rows withheld as location-sensitive | 0 |
| Cited IDs missing from the ledger | 0 (every claim tag cited in DRAFT_v10.md has a row) |
| Released rows with status other than VERIFIED | 0 |
| Reference rows (R01-R16) | 16 in, 16 out (bibliographic; edits only as listed below) |

Rows added compared with the v7 release ledger: C221-C230 and R14-R16 (all new in v10); none removed. Rows annotated
in v10 (listed in the ledger's "v10" count paragraph) carry their v10 annotations.

## Changes by kind

Role words and identifiers in the "Original" column are elided as `<role word>`, `<id>` or `<private path>` so that
this log does not republish them.

| Kind | Original text | Replacement | Count | Rows | Origin |
|---|---|---|---|---|---|
| Session ID removed | session `` `<id>` `` (run-session identifier) | run session (identifier withheld; start time as given in the Value column) | 1 | C191 | 2026-10-09 |
| Private storage detail neutralised | (`<cloud service>` archive, not on local disk; | (private cloud archive, not on local disk; | 1 | C149 | 2026-10-09 |
| Private storage detail neutralised | archived with verified receipts (`<cloud service>` bundle `<id>`) | archived with verified receipts (private cloud archive bundle) | 1 | C211 | 2026-10-09 |
| AI role term rewritten | per `<role word>` reading | per the reading of an AI-assisted sub-session | 1 | R04 | 2026-10-09 |
| AI role term rewritten | import per `<role word>` | import per an AI-assisted sub-session | 1 | C181 | 2026-10-09 |
| AI role term rewritten | by a web-research `<role word>` | by an AI-assisted web-research sub-session | 1 | References intro | 2026-10-09 |
| AI role term rewritten | web-research `<role word>` | AI-assisted web-research sub-session | 1 | C009 | 2026-10-09 |
| Internal role name removed | (`<role word>` instruction) | (by instruction during the v1 ledger session) | 1 | footer "Count (v1)" | 2026-10-09 |
| Local environment detail neutralised | `<local virtual-environment name>` rvt-py 2.2.1 | installed rvt-py 2.2.1 package, | 1 | C177 | 2026-10-09 |
| Working-state receipt replaced | LOG.md:4232; `<agent state file>`:3; | LOG.md:4232; sent-mail record; | 1 | C143 | 2026-10-09 follow-up |
| Session ID removed | (`<role word>` transcript `<id>`.jsonl) | (AI-assisted session transcript; identifier withheld) | 1 | C192 | v10 |
| Session ID removed | (`<role word>` transcript `<id>`.jsonl) | (AI-assisted sub-session transcript; identifier withheld) | 1 | C192 | v10 |
| AI role term rewritten | read by the verification `<role word>` | read by an AI-assisted verification sub-session | 1 | C221 | v10 |
| Internal role name removed | and `<model name>`'s finding A03 | and the external review's finding A03 | 1 | R16 | v10 |
| Private file path neutralised | from `<private path>` and `<private path>` `background` (two manifest build records) | from the private build records of the Verden and Oldenburg entity manifests (not released), field `background` | 1 | C202 | v10 |
| Local environment detail neutralised | deg0.py, `<local virtual-environment name>`, python -B; | deg0.py, local Python environment, python -B; | 1 | C228 | v10 |
| Header rewritten | internal header (DRAFT_v1 scope) | release header: v10 scope, preprint relation, omitted-row pointer, AI-assisted provenance statement ("this session" = the AI-assisted session that made the row's revision, v0-v10), hash conventions | 1 | header | as 2026-10-09 |
| Release note added | none | one paragraph before "Numbers that appear in the draft" | 1 | footer | as 2026-10-09 |

Totals: 16 in-text replacements, 1 header rewrite, 1 note added, 5 rows omitted, 0 withheld.

## Kept unchanged, for the author to review

- Receipt directories `work/gpt6_pro_collab_2026-10-07/` (as in 2026-10-09) and, new in v10,
  `work/gpt6_pro_paper_review_2026-10-10/`. Both are receipt paths that name the external AI model consulted for review;
  the paper's AI statement already names it.
- C008 keeps its v10 annotation that the original claim ("no detector weights") was refuted by primary sources; the
  paper now states the actual position (C221). The status cell still begins VERIFIED because the annotation records
  the check.
- C149 and C211 record the weights as "to be deposited (Zenodo) before publication" / "to be released". Preprint
  version 1 says the deposit is in preparation; the plan and the checkpoint list are the same.
- C226 (target venue and its fee status) is cited in DRAFT_v10.md and released; preprint version 1 drops that line.
- C020 ("Sealed TEST not used") states only that the test set is sealed.
- C187 keeps the stage label `run_s3`; C206 keeps "Co-Authored-By an AI model" (commit attribution).

## Scan output (CLAIMS_RELEASE.md, after final build)

Checks: the list in the 2026-10-09 log, re-implemented (`scan_claims_release.py`; the 2026-10-09 scan script was not
kept). Calibration: on the 2026-10-09 released ledger the re-implementation gives 0 for every leak check and 36
decimal hits, as logged then; its 6-7-digit inspection pattern is slightly broader (5 hits there against 3 logged).

```
## _PRIVATE: 0
## drive_letter_path: 0
## /mnt/: 0
## /home/: 0
## role_names (case-sensitive): 0
## role_names (case-insensitive): 0
## external-model short name (case-sensitive): 0
## OSGB grid ref: 0
## decimal >=4dp (lat/lon): 50
## easting/northing pair: 0
## EPSG + coordinate pair: 0
## standalone 6-7 digit number (inspection): 6
## session-like ID: 0
## SEALED/RESERVED file names: 0
```

## Manual inspection of every remaining hit (all false positives)

Leak hits after justification: **0**.

| Hits | Rows | What it is | Why not a location |
|---|---|---|---|
| 0.123352, 0.083671, 0.172125, 0.081044, 0.152473, 0.1234, 0.060852, 0.046542 | C005, C006, C096, C097, C101, C157 | unrounded recall differences and interval ends, labelled in the row | single values, never a pair |
| 0.09906, 0.0991, 0.09906152 | C085, C213 | ADAF barrow Oldenburg recall 95/959 | labelled as recall |
| 0.168956 | C227 | ADAF barrow Verden v0 recall 123/728 | labelled as recall |
| 1.7217 | C202 | Verden density 728 / 422.84 per km² | arithmetic shown in the row |
| 117.0363, 183.4098, 300.4461 (twice), 199.8779, 300.5239, 122.3971, 422.8432 | C202 | scored-area components in km² | labelled areas; sums shown in the row |
| 3.0023 | C029 | 2,558 positive draws / 852 positive windows | arithmetic shown in the row |
| 10.5281, 10.5334, 10.1016, 10.3390, 10.1080, 10.1007, 10.1002 and suffix fragments (2026.105733, 00934690.2024, 2020.102241, 2412.05203) | C007, C008, C014, C223, R01-R15 | DOI prefixes and suffixes, arXiv number | bibliographic identifiers |
| 105733 (twice), 102241, 9050293 | C007, R01, R10, R12 | article numbers and a DOI suffix | bibliographic identifiers |
| 1582430 | C014 | inside the ADAF barrow checkpoint SHA-256 `51b679fa…7ef1582430d2…` | hash fragment |
| 3855551 | C027 | inside a truncated script hash | hash fragment |

No hit forms an easting/northing or lat/lon pair, and no hit sits beside a lead ID or place name.

The repository leak scan (`LEAK_SCAN.md`, addendum of 2026-10-13) was also run on this ledger and on this log with the
release rule; this ledger is now a file written for the release, so role words and local paths count as defects
there, not as released evidence.
