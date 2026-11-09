# Aggregate-results verification demo (laptop CPU)

This demo verifies published numbers from released aggregate files. It does not run detection.

`recompute_headlines.py` reads two released aggregate result files of the November 2026 preprint
(`paper_a/`, https://doi.org/10.5281/zenodo.23874293, not peer reviewed), checks their SHA-256 against
`paper_a/MANIFEST.sha256`, and recomputes:

- **Dutch sealed test** (D1 vs the public ADAF barrow model at matched review workload): the hit counts 623 and 516 of
  932 registered barrows from the paired counts (both / only D1 / only ADAF), recall, and the +11.5-point difference.
  The stored simultaneous 95 % interval (+7.4 to +15.5 points) is printed and checked for consistency, not recomputed.
- **Denmark CONFIRM-D stage A** (the later two-checkpoint system S_D + PROX, not the D1 checkpoint above): S_D + PROX 84.2 % vs ADAF barrow plain 72.2 % of 5,273 protected barrows on 171 unseen
  cores at 2 proposals/km², each checked to equal a whole number of barrows.

## Why this demo and not inference

Running a checkpoint needs a 320 MB weights file from Zenodo, the ADAF code checkout and its AiTLAS wheel, the pinned
GPU-era environment (`requirements-lock.txt`: torch 2.14 with CUDA 13 wheels) and LGLN DGM1 tiles. Scoring the output
against registered monuments needs register geometry. This demo needs none of that: only the Python standard library
and two small files already in this repository. The inference route is documented in the top-level `README.md`
("Regenerating proposals" and "Reproduction check").

## Setup and command

Requirements: Python 3 (tested on 3.12.14 and 3.14.7; written for 3.9+), no packages, no network. From the repository root:

```bash
git clone https://github.com/tee-20190/ndsbarrow-adapt
cd ndsbarrow-adapt
python demo/recompute_headlines.py      # use python3 on macOS/Linux if python is not found
```

Expected output: `demo/expected_output.txt`. The last line is `Result: ALL CHECKS PASS` and the exit status is 0; any
changed input byte or count gives `FAIL` lines and exit status 1.

Run record (2026-11-08): WSL Ubuntu 22.04, Python 3.12.14, exit 0, 0.10 s wall time, about 16 MB peak memory; output
identical to `expected_output.txt`. Also exit 0 under Windows Python 3.14.7. A copy with one paired count changed
(only-D1 150 to 151) failed three checks and exited 1.

## Limitations

- It recomputes arithmetic from released aggregates. It does not rerun detection or scoring, and it cannot recompute
  the cluster-bootstrap interval: that needs per-site hits, which are not released (`paper_a/README.md`, "Data").
- The numbers are registered-monument recall at a fixed review workload, not discovery precision. Unmatched proposals
  are not counted as false positives or as discoveries.

## Attribution and licences

- This demo: Apache License 2.0 (top-level `LICENSE`), © Thomas Shanahan; written by AI coding agents under his
  direction.
- The detectors compared were adapted from the public ADAF barrow model: Kostovska, A., Kocev, D., Čož, N., Kokalj, Ž.,
  Curran, S., Corns, A. (2025), ADAF machine learning model weights v1.0.0, Zenodo, doi:10.5281/zenodo.15848663,
  CC BY-SA 4.0; ADAF code https://github.com/EarthObservation/adaf, Apache-2.0, © ZRC SAZU; method in Čož et al.
  (2026), Journal of Archaeological Science: Reports 71:105733.
- Data behind the result files (not included, not needed here): Netherlands AHN4 DTM (PDOK) and RCE
  `landschapsatlas:grafheuvels`, CC0 1.0; Denmark SDFI Danmarks Højdemodel (free-data terms with attribution) and
  Slots- og Kulturstyrelsen, Fund og Fortidsminder.
- Trained weights: https://doi.org/10.5281/zenodo.23862470, CC BY-SA 4.0.
