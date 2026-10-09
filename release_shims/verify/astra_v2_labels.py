"""Fail-closed replacement for the withheld module verify/astra_v2_labels.py (see WITHHELD.md).

The original module defines the English label rules and the sealed English test squares. It is withheld because it
identifies sealed test terrain. Released code imports it transitively (score_v0.py -> region_score.py ->
score_holdout.py -> r2_masks.py), so release_paths.py places this replacement in the runnable working copy only.

POS, EXCLUDE and square() reproduce the original's label-term rules and the generic British National Grid letter
function; they identify no terrain. TEST and VAL fail closed: any use raises, so no code path can run that needs the
sealed squares (English mask building and English holdout scoring). The German scoring, inference and training paths
of the paper do not use them.
"""
import re

TERMS = ['moated site', 'moat', 'ringwork', 'motte', 'castle mound', 'hillfort', 'hill fort', 'enclosure', 'henge',
         'barrow', 'bowl barrow', 'round barrow', 'camp']
POS = r'\b(?:' + '|'.join(map(re.escape, TERMS)) + r')\b'
EXCLUDE = r'\b(?:site of|Roman villa|cross|church|bridge|wall|stone|cairn)\b'


def square(x, y):
    """Two-letter 100 km British National Grid square of an easting/northing."""
    e, n = int(x) // 100000, int(y) // 100000
    a = (19 - n) - (19 - n) % 5 + (e + 10) // 5
    b = ((19 - n) * 5) % 25 + e % 5
    return ''.join(chr(65 + v + (v >= 8)) for v in (a, b))


class _Withheld:
    """Raises on any use."""

    def _refuse(self, *_a, **_k):
        raise RuntimeError('sealed English test squares are withheld from this release (WITHHELD.md)')

    __contains__ = __iter__ = __len__ = __getitem__ = values = keys = items = _refuse


TEST = _Withheld()
VAL = _Withheld()
