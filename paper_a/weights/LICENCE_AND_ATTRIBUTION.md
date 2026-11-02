# Licence and attribution

## Model weights: CC BY-SA 4.0

Every checkpoint in this deposit is a derivative of the public ADAF barrow weights:

> Kostovska, A., Kocev, D., Čož, N., Kokalj, Ž., Curran, S., Corns, A. (2025), ADAF machine learning model weights (v1.0.0), Zenodo, doi:10.5281/zenodo.15848663, CC BY-SA 4.0.
>
> The ADAF software and method are described in Čož et al. (2026), Journal of Archaeological
> Science: Reports 71:105733.

The ADAF weights are referenced, not redistributed here. As the ADAF licence requires, the derived weights are shared
under the **Creative Commons Attribution-ShareAlike 4.0 International** licence
(https://creativecommons.org/licenses/by-sa/4.0/). Any shared copy or further derivative must keep this licence and
attribution.

Chain for each family (parent → checkpoint):

| Family | Parent | Training data used to make it |
|---|---|---|
| D1 (D1-F1_BARROW_L_2026092805…14) | ADAF barrow | Lower Saxony |
| D_k (DT1D-F1_BARROW_L_k) | dense SSL encoder SSL_D_k ← ADAF barrow | Lower Saxony (SSL and F1) |
| A_k (DT1A-F1_BARROW_L_k) | ADAF barrow | Lower Saxony |
| B, G arms (DT1B/DT1G) | ADAF barrow (G: pooled SSL encoder ← ADAF barrow) | Lower Saxony |
| C_k (DT1D-F1_BARROW_L_1010…1515), replays | SSL_D_k ← ADAF barrow | Lower Saxony |
| SSL_D_k encoders | ADAF barrow | Lower Saxony terrain (no labels) |
| FAINT-LEARN-2 arms | D_202 | Lower Saxony |
| EB (EB_BARROW_EN_…) | ADAF barrow | England |
| CONT-e41 | earlier project English model (R2 line) ← ADAF | England |

## Data used to train them (not redistributed)

- **Lower Saxony.** Terrain: LGLN (2026), Digitales Geländemodell DGM1, CC BY 4.0, modified by resampling and residual
  transforms. Labels: Niedersächsisches Landesamt für Denkmalpflege, Denkmalatlas Niedersachsen, CC BY-SA 4.0,
  modified by rasterisation.
- **England.** Terrain: Environment Agency 1 m composite DTM, Open Government Licence v3.0. Labels: Historic England,
  National Heritage List for England scheduled monuments, Open Government Licence v3.0.

No checkpoint in this deposit was trained on Danish or Dutch data. Danish (Fund og Fortidsminder, SDFI DHM) and Dutch
(AHN4, RCE) data were used only to evaluate them.

## Excluded on licence grounds

The DK-REGSUP-1 arm-A checkpoints (`RS1A-F1_BARROW_L_F*_*`, 45 files) were trained partly on Fund og Fortidsminder
register geometry. The agency states CC0 for its textual data and does not name a licence for geometry. They are not
in this deposit until the agency has been asked (see `WITHHELD.md`).

## Code

The code in `paper_a_code.zip` is © Thomas Shanahan, Apache License 2.0, as in the earlier release
(github.com/tee-20190/ndsbarrow-adapt). It imports the ADAF code (EarthObservation/adaf), which carries its own
licence.
