# ADAPT-NDS-1 erratum 1: the scorer's district label (2026-10-05)

**What failed.** The first scoring run (17:13:47Z) stopped with `KeyError: 'district'` at `score_adapt.py:329`. This
was after every refusal check had passed and before any score was computed. The frozen scorer (pinned by freeze
ba07548b at c2ae6c42…) reads `f["confirm"]["district"]` for the descriptive field `RESULT.confirm_district`. FREEZE.json's
`confirm` block holds only paths. The district is recorded in `CONFIRM_SELECTION.json`, which the freeze pins as
`confirm_selection_sha256` (eefd3365…). The unit tests exercise the rules, contrasts and region blocks, but not
`analyse()` against the real freeze, so they did not catch it.

**Repair.** `score_adapt_erratum1.py` is a new, non-frozen wrapper. It:

- imports the frozen scorer unmodified;
- checks the selection file's hash against the freeze;
- supplies that one field (`Oldenburg`);
- runs the frozen `main()` with the same arguments.

No input, rule, threshold, bootstrap draw count or seed changes. The wrapper refuses if the freeze ever records the
field itself. `SCORE_RECEIPT.json` records the hashes of the scorer, the wrapper, the freeze, the selection file and
the result.

**Effect on the result.** None beyond the label. The scorer's bytes still match the freeze pin. All binding and
session checks ran as frozen. `RESULT.json`'s numbers are produced by the frozen code path. They were then recounted by
an independent implementation (`recount_adapt.py`, `RECOUNT.json`).
