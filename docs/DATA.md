# Data acquisition and provenance

**Dataset:** *Human Activity Recognition Using Smartphones* by Reyes-Ortiz, Anguita, Ghio, Oneto and Parra. UCI Machine Learning Repository, DOI [10.24432/C54S4K](https://doi.org/10.24432/C54S4K), **CC BY 4.0**. It is public and de-identified, and downloading it needs no registration or application.

`python -m har_repro.cli prepare-data` does the following:

1. **Reuses a verified archive** already in `data/raw/` if its SHA-256 matches.
2. **Otherwise accepts a local archive** passed with `--local-archive`, if its SHA-256 equals the official outer or inner archive.
3. **Otherwise downloads** from the sources in `configs/data_sources.toml`, in order. It accepts a file **only if its SHA-256 matches**:

   | Order | Source | Archive | Expected SHA-256 | Measured speed (2026-11-25) |
   |---|---|---|---|---|
   | 1 | UCI legacy path `…/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip` | inner | `2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0` | about 63 KB/s (16 min) |
   | 2 | UCI current URL `…/static/public/240/human+activity+recognition+using+smartphones.zip` | outer | `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031` | 4–17 KB/s (about 1 h) |
   | 3 | Internet Archive capture of URL 2 (memento 2026-09-28 21:28:28 GMT) | outer | same as 2 | about 500 KB/s (2 min) |

4. **Checks the archives' internal CRCs**, verifies the inner archive's SHA-256, and extracts it with path-traversal protection. macOS metadata (`__MACOSX/`, `.DS_Store`) is skipped.
5. **Verifies the SHA-256 of the 10 extracted files** the pipeline reads or documents.

**Cross-source evidence (2026-11-25):**
- Source 2 and source 3 are byte-identical.
- The inner archive inside source 2 is byte-identical to source 1.
- All 74 inner archive members have identical SHA-256 across the three sources.

**Key extracted files (SHA-256):**

| File | SHA-256 |
|---|---|
| `features.txt` | `b384889885b1c8680ecf250e0e605b4c79536cdf09ce9f2fa7225a3c6773b5c7` |
| `activity_labels.txt` | `f6e6b292704438261f0283b2a82659ab0661447dc16520938bdf379ed1bf8de0` |
| `train/X_train.txt` | `9c1246099c8d5463779eec5a3845cc3898df9d8c715441ab2a1232d4421fc42f` |
| `train/y_train.txt` | `415b3357e2e2d5e70a45c781947c6bdd80e0410a579d862eedcd6bbce3c694ef` |
| `train/subject_train.txt` | `66e1e2fbe9b396fa5155f531dc3348c803f503be663e2318772d49947db9153a` |
| `test/X_test.txt` | `99035209add50d17650e847d8a4658d784a5707466ebb7c2deb97c28e5250a78` |
| `test/y_test.txt` | `2a94cb53ca46b956c1b2aebd4a09a72badd8a8e9ea9cdfe9f31b3306a08666bb` |
| `test/subject_test.txt` | `63b8c577f4a85431c06bd87f6384608c90a9f72f05c7956dd21255dc63515c07` |

**Verified properties** (`tests/test_data_integrity.py`, `results/data_validation/data_validation.json`):
- Shapes are 7,352 × 561 and 2,947 × 561.
- Participants: 21 train and 9 test, with no overlap. Test participants: 2, 4, 9, 10, 12, 13, 18, 20, 24.
- Label counts — train 1226 / 1073 / 986 / 1286 / 1374 / 1407; test 496 / 471 / 420 / 491 / 532 / 537. The test counts equal the class totals implied by the paper's confusion matrix.
- No non-finite values. Both partitions span exactly [−1, 1].
- No duplicate rows, and no rows shared between train and test.
- Parsed values are bitwise equal to exact decimal parsing.

**Storage:** `data/` holds `raw/` and `extracted/` and is git-ignored. Neither archives nor extracted files are committed. Set the `HAR_DATA_DIR` environment variable to use another location.
