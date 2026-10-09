# Glossary

The released protocols, results and code are byte-identical to the files that produced the paper (decision D1), so
that the SHA-256 values in the freeze files can be checked. They therefore keep the project's working vocabulary.
This glossary defines it. It is the only file written for the release that uses these words.

| Term | Meaning |
|---|---|
| owner | the author, Thomas Shanahan |
| coordinator | an AI coordinating session that planned, ran and recorded experiments under the author's direction |
| Astra, Sol, Luna | OpenAI GPT-6 Astra, Sol and Luna, used through OpenAI Codex (paper's "Use of AI tools" statement) |
| Codex | OpenAI Codex (paper's AI statement) |
| Pro, GPT-6 Pro | OpenAI GPT-6 in Pro reasoning mode (paper's AI statement) |
| Opus, Fable, Haiku | Anthropic Claude Opus 5.5, Claude Fable 5.1 and Claude Haiku 5.5 (paper's AI statement) |
| `_PRIVATE` (in a file name) | a file the project kept out of public material by default. In this repository, a released file whose name contains `_PRIVATE` was reviewed and reclassified as releasable (below); every other `_PRIVATE` file is withheld and listed in `WITHHELD.md` |
| absolute paths under the WSL Windows-drive mount or a Linux home directory | paths on the author's machine; `release_paths.py` replaces them in a working copy (`README.md`) |
| absolute paths under the Colab content directory | working directory of the remote Colab GPU VM used for training and inference |
| SEALED, sealed TEST | held-out English test terrain that was never opened; it stays sealed and is not identified in this release |
| reserved terrain | terrain set aside for future evaluation; refused by the code, not used in this paper |
| DEV, FIT, CONFIRM, EVAL | dataset roles: development district (Verden), training districts, confirmation district (Oldenburg), evaluation manifest |
| CU | Colab compute units (resource accounting) |

## Files reclassified from `_PRIVATE` to released

These entity manifests were opened and checked before release: they contain only registered monuments from the
Denkmalatlas Niedersachsen (record IDs, footprints, groups, height strata), tile and sampling-cell layout, LGLN tile
pins and inside masks. They contain no detector proposals and no unmatched detections. They are released under
CC BY-SA 4.0 with attribution (`NOTICE`).

| File | Content |
|---|---|
| `outputs/fresh_region_2026-10-03/step0/INDIVIDUAL_MANIFEST_PRIVATE.json` | Verden: 728 register barrows (individual unit), 717 tiles (275 positive, 442 background) |
| `outputs/fresh_region_2026-10-03/step0/GROUP_MANIFEST_PRIVATE.json` | Verden: 300 barrow groups (group unit), same tiles |
| `outputs/fresh_region_2026-10-03/step2/INFER_MANIFEST_PRIVATE.json` | Verden: inference windows for the 717 tiles |
| `outputs/fresh_region_2026-10-03/step2/ENCLOSURE_MANIFEST_PRIVATE.json` | Verden: 15 descriptive windows around registered enclosures (never scored) |
| `outputs/fresh_region_2026-10-03/step2/STAGING_PLAN_PRIVATE.json` | Verden: DGM1 tile URLs, bytes, SHA-256 and window-to-tile plan |
| `outputs/fresh_region_2026-10-03/step2/INSIDE_MASKS_PRIVATE.npz` | Verden: district inside masks for boundary tiles |
| `outputs/fresh_region_2026-10-03/step0/TILE_INDEX_PRIVATE.json` | Verden: LGLN tile index (names, URLs, years, delivery) |
| `outputs/fresh_region_2026-10-03/step0/TILE_LEDGER_PRIVATE.jsonl` | Verden: per-tile download ledger (bytes, SHA-256, grid checks) |
| `outputs/fresh_region_2026-10-03/step0/NODATA_MEMBERS_PRIVATE.json` | Verden: per-barrow terrain-validity checks |
| `outputs/fresh_region_2026-10-03/step0/EDGE_SUPPLEMENT_PRIVATE.json` | Verden: register records of neighbouring districts within 1 km (used as IGNORE) |
| `outputs/adapt_nds_2026-10-05/confirm/run/INDIVIDUAL_MANIFEST_PRIVATE.json` | Oldenburg: 959 register barrows, 699 tiles (259 positive, 440 background) |
| `outputs/adapt_nds_2026-10-05/confirm/run/GROUP_MANIFEST_PRIVATE.json` | Oldenburg: 272 barrow groups, same tiles |
| `outputs/adapt_nds_2026-10-05/confirm/run/INFER_MANIFEST_PRIVATE.json` | Oldenburg: inference windows |
| `outputs/adapt_nds_2026-10-05/confirm/run/STAGING_PLAN_PRIVATE.json` | Oldenburg: DGM1 tile URLs, bytes, SHA-256 and window plan |
| `outputs/adapt_nds_2026-10-05/confirm/run/INSIDE_MASKS_PRIVATE.npz` | Oldenburg: district inside masks |
| `outputs/adapt_nds_2026-10-05/fit/FIT_TABLE_PRIVATE.json` | training districts (Uelzen, Harburg, Heidekreis): 852 positive and 852 background windows with register footprints |
| `outputs/adapt_nds_2026-10-05/fit/FIT_STAGING_PLAN_PRIVATE.json` | training districts: DGM1 tile plan |
| `outputs/adapt_nds_2026-10-05/fit/FIT_STAGING_PINNED_PRIVATE.json` | training districts: tile plan with Content-Length and ETag pins; excluded district outlines |
| `outputs/adapt_nds_2026-10-05/fit/FIT_TILE_INDEX_PRIVATE.json` | training districts: LGLN tile index |
| `outputs/adapt_nds_2026-10-05/fit/FIT_INSIDE_MASKS_PRIVATE.npz` | training districts: inside masks |
