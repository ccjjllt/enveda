# CASMI26 extlib 052

MS2 reference spectra of structures **not** in competition `train.parquet` (InChIKey14 excluded).
Built for round 052 external-library overlay on the 051 (alexchilton 0.480) base.

## Contents
- `refs.parquet` — ionmode, prec, mz, it, key (InChIKey14), smiles_canonical, source, license
- `extlib_config.json` — pinned sha256, TAU, K, LIB_TAU, selftest path
- `selftest.json` — three self-hit cases the overlay must reproduce

## Sources / licenses
| source | license | notes |
|---|---|---|
| senanuretin GNPS CC0 expanded + GNPS2 snapshot | CC0-1.0 | few non-train left |
| yehudaitkin zenodo novel top128 | CC-BY-4.0 | train pairs removed upstream |
| MSnLib v8 MS2 (MassSpecGym / zenodo) | CC-BY-4.0 | |
| kizhowl MassBank original permissive (CC0/CC-BY subset) | CC0 / CC-BY-4.0 per record | dl-de excluded |
| takumuhata/casmi26-extlib | CC0-1.0 | top-128 padded peaks |

MassBank 2026.03 CC0 Kaggle mirror had 0 non-train after exclusion (all already in train).

## Calibration
TAU=0.55, K=3, LIB_TAU=0.9 chosen on 2500 Enveda-180 timsTOF train molecules (any hit = false): false-hit rate 8/2500 = 0.32%.
