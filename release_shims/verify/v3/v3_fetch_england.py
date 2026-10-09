"""Fail-closed replacement for the withheld module verify/v3/v3_fetch_england.py (see WITHHELD.md).

The original module fetches English Environment Agency LiDAR windows and names the sealed English test squares in its
refusal guard, so it is withheld. Released code imports it at module level (work/stage2_2026-09-24/stage2_run.py), so
release_paths.py places this replacement in the runnable working copy only. Importing succeeds; any use raises.
The German scoring, inference and training paths of the paper do not use it.
"""


def __getattr__(name):
    raise RuntimeError(f'v3_fetch_england.{name}: English LiDAR fetcher withheld from this release (WITHHELD.md)')
