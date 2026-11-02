# CONFIRM-D addendum SEQ-1: DEV decode ran before the OSM manifest (recorded 2026-10-25 20:55Z)

**Deviation.** FREEZE §12 says the OSM manifest is committed **before any decode**. The DK-FUNEN-DEV decode ran before
it. Raised by Codex review PR25-2 (`work/progress_review_2026-10-25_codex/`); accepted.

## Chronology (UTC; from git and file mtimes)

| Time | Event |
|---|---|
| 17:35:09 | `dk_osm.py` snapshot run starts (OSM only; cores in sorted order, DEV cores last) |
| 17:37:44 | FREEZE.md committed (3cd5a9682) |
| ~17:45–18:31 | DEV mosaics staged (15 cores) |
| 18:39:42 | §10.2a parity on 2 DEV cores (D1_05): PASS |
| 18:42:05–19:00:52 | DEV decode, D1_05 + ADAF, 15 cores; assembled 19:01:02 |
| 20:51:35 | latest OSM snapshot written (160/186; no DEV snapshot yet) |
| — | OSM manifest not yet committed; DEV proposals not scored, not opened, no mask built |

## Why this does not change the comparison

- The manifest gate exists so masks cannot adapt to detector output.
- Each snapshot is `cr1_score.core_query(i, j)`, which depends only on the core index, fetched by `cr1_e0.fetch`. Both are
  pinned, and neither reads any decode file.
- The DEV snapshots are fetched after the decode by that fixed script, from that fixed query. Their content cannot
  depend on the proposals.
- DEV is the development slice. It decides only the §5 reach (D1_05 + PROX R15/R25) and is never scored for a D system.
- No DK-JW tile has been decoded.

## Commitments (fixed now, before any further output)

1. The §5 reach decision is computed only after `OSM_MANIFEST.json` is committed.
2. **No DK-JW decode starts before the OSM manifest commit** (strict §12 order for the primary cohort).
3. The final report lists this addendum and the chronology. The literal §12 sequence is reported as **not met for
   DEV**. It is not reported as passed.

## Update 21:30Z: OSM manifest committed

- The `dk_osm.py` run finished at 21:27:23Z: 186/186 snapshots, 165.6 MB, private manifest sha256 `cecdc4bc…6eff`.
- 180 snapshots came from overpass-api.de with OSM base 2026-10-25. Six came from the pinned fallback mirror
  (overpass.kumi.systems), whose OSM base dates from 2026-05-31 to 2026-07-24. One of those six is a DEV core; the
  other five are DK-JW cores.
- They are kept as produced. Re-fetching only those six would select among snapshots. The effect is limited to S-v1
  masks that miss any OSM objects added since those dates, and the effect is reported descriptively.
- The manifest is committed with this note, before the §5 reach scoring and before any DK-JW decode.
