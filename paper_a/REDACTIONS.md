# Redactions and exclusions in `paper_a/`

Files are released byte-for-byte except where listed here. Each redaction replaced one passage; nothing else in the
file changed. The original SHA-256 is the value pinned by the freezes and by the weights deposit's `CODE_PINS.json`;
the released SHA-256 is the file in this folder. The removed text is not reproduced here. For each change this file
gives the released line(s) and what kind of text was removed; a holder of the original file can check it against
the original SHA-256.

## Redacted files

### `paper_a/code/work/open_transfer_test_2026-10-22/ot_score.py`

- Original SHA-256: `ffcb7eb07b231b06f074d0c98a9cc6373639ac82d59ae22e2c8cebfc8457dcd2`
- Released SHA-256: `64ca9166b693a42acd6982556514c330fbe3611c01d75291085bc69cdf4dd2e2`

Change 1, released line 10. Removed: an absolute path on the author's machine (Windows and WSL forms). Released passage:

```text
ot_score.py <repo-root>   # scores once
```

### `paper_a/code/tools/clean_start/run_sd_prox.py`

- Original SHA-256: `e5a458ac9a56b09d6be742e064d04d7b639f0efcd8800e5bd4e4204dfbb021df`
- Released SHA-256: `ee43a1f134a41e0d21c85949c8d481e205dd8627745fe9ee8046c8d37ea0cfbf`

Change 1, released line 3. Removed: the file name of the clean-start example, which is the 5 km grid index of a Lower Saxony development core. Released passage:

```text
tools/clean_start/examples/<example>.json [--out DIR]
```

### `paper_a/code/work/open_transfer_2026-10-22/ot_score.py`

- Original SHA-256: `daaa134b39932fb7f47fc25122829b780692466af64ae14c8297b5682a0bbb0a`
- Released SHA-256: `c8021134e5db4af92015373751473fa12adf8b1609eff012c832768c01f7fcf1`

Change 1, released line 10. Removed: an absolute path on the author's machine (Windows and WSL forms). Released passage:

```text
ot_score.py <repo-root>   # scores once
```

### `paper_a/code/work/open_transfer_2026-10-22/ot_labels.py`

- Original SHA-256: `e7cd37700107b8a95be0d9f4c77e9562f184781c25d2e82b78097b1089c77d7f`
- Released SHA-256: `86d5ffcd36418e68ef22d995df7f4bc835fc98f96146f807b590258c4218a9f9`

Change 1, released line 4. Removed: an absolute Linux home path that contains the author's user name. Released passage:

```text
  <venv>/bin/python -I work/open_transfer_2026-10-22/ot_labels.py census|build
```

Change 2, released line 25. Removed: the names of two provinces in the sealed TEST role that hold no primary record
and remain reserved. The released tuple lists only the three TEST provinces that were unsealed and scored, so the
census function no longer reproduces the original core exclusions near those two provinces. Released passage:

```text
TEST_SEALED = ("Gelderland", "Noord-Brabant", "Limburg")  # + two reserved provinces with no primary record [names redacted; see paper_a/REDACTIONS.md]
```

### `paper_a/protocols/work/open_transfer_2026-10-22/FREEZE.md`

- Original SHA-256: `fcd61bec63cbd1ca4bf0c2f498d5bc650484528d2b788798395407b6e9b7c5ef`
- Released SHA-256: `d923dfc5ef8b3557ad2fe7abc80b348d0c49206ce834cea33a6dbdaec7f13fef`

Change 1, released lines 45-46. Removed: the names of the same two reserved provinces. Released passage:

```text
  - **TEST_SEALED** (untouched; reserved for confirming whatever this packet or a successor selects): Gelderland,
    Noord-Brabant, Limburg and two further provinces [names redacted; no primary record]. **955 records.** No AHN tile is read for a core in these provinces,
```

### `paper_a/protocols/work/open_transfer_test_2026-10-22/FREEZE_TEST.md`

- Original SHA-256: `5c29460618b7b490adc9b5d2da73b35183aa157454e3d8e88ca22e7eebd38102`
- Released SHA-256: `7d9b20e67b0d90e0dc941b25614e69169f95cfc6a0933511e37383ae1187223d`

Change 1, released lines 44-45. Removed: the names of the same two reserved provinces (twice). Released passage:

```text
  defined it: Gelderland 745, Noord-Brabant 98, Limburg 112, and 0 in two further TEST_SEALED provinces
  [names redacted] that hold no primary record.
```

### `paper_a/protocols/work/confirm_d_2026-10-25/FREEZE.md`

- Original SHA-256: `2ff181aa7d120b3a7a7d34554e7e7eff2cc9e629d713c630ecead14f42376b94`
- Released SHA-256: `0fe4e95e1bc1a19ee5f50d54078e6dac20c0630065f9713be7aa2ccc94ddbcb5`

Change 1, released lines 286-287. Removed: the names of three Danish regions reserved for later confirmation and their register record counts. Released passage:

```text
- The other Danish regions stay unread. They are reserved for a later confirmation or for discovery [region names
  and record counts redacted for this release; see paper_a/REDACTIONS.md].
```

### `paper_a/protocols/work/goettingen_transfer_2026-10-20/SPEC.md`

- Original SHA-256: `7a38f98264cabd84328694a9bcf7737f2d704ef9995aa99bf0938e699f2449a6`
- Released SHA-256: `fe64592cd8a1ce7bd3758580a6407120bb08cdeb52ef6912cf098d049da7268d`

Change 1, released line 24. Removed: the name of a Lower Saxony district reserved as a sealed confirmation district. Released passage:

```text
- rank 2 is D4R's reserved confirmation district [name redacted] and stays sealed;
```

### `paper_a/protocols/work/dk_regsup1_2026-10-29/FREEZE.md`

- Original SHA-256: `dec0e419bb7f5293f04bfbaa2f4607555e8ebfc9bb959bbb66b6a31d6a6ff67b`
- Released SHA-256: `2dd1efbd7dd4ead3f6b547b2a69cfa0e362d80f6196480aaff03b649fbdc9e02`

Change 1, released lines 25-26. Removed: the name of a reserved Danish region (twice: in the text and in a decision-record file name). Released passage:

```text
Stage B ([reserved region; name redacted]) is not part of this freeze; it needs its own FREEZE, and the
owner's conditional unseal ([decision record; file name redacted]) applies only if §8 passes.
```

Change 2, released line 90. Removed: the name of a heritage custodian and the number of places referred to it. Released passage:

```text
  proposals [custodian correspondence detail redacted]. A positive's own locality records are not "other".
```

Change 3, released lines 188-189. Removed: the name of a reserved Danish region. Released passage:

```text
as one frozen read of a reserved region
  [name redacted] with its own FREEZE committed before any of its DHM tiles is fetched.
```

Change 4, released line 190. Removed: the name of a reserved Danish region. Released passage:

```text
The reserved region stays reserved.
```

Change 5, released lines 228-229. Removed: the names of the reserved Danish regions. Released passage:

```text
DK-JW was exposed by CONFIRM-D; this stage reads nothing else. The reserved Danish regions [names redacted] and the
sealed Dutch TEST are not read. No other DHM tile is fetched.
```

## Excluded files

Pinned in the weights deposit's `CODE_PINS.json` but not released here:

| File | Original SHA-256 | Reason |
|---|---|---|
| `tools/clean_start/restore.py` | `fae219648dcb719390ccc26511c9068455c63de52f00d353231e30020ab5f830` | its only function is to download inputs from the author's private release; its usage line names the development-core example |
| `tools/clean_start/README.md` | `9fed3534753042a31e088b7190ad30d9c02786af79a32cfd0f9b112582b21394` | names the author's private repository and private release, a local home path and the development-core example |
| `tools/clean_start/examples/<example>.json` (name withheld) | `d5f72bf1b8b0a2f6ad21f922941c5f3025c412a42ea644c428a2ba66c91eb0c3` | manifest of a Lower Saxony development core: its grid identity and extent coordinates, a local home path and the private release name |
