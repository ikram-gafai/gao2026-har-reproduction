# Reproducing Gao et al. (2026): compact vs full XGBoost on UCI HAR

This is an independent, pre-registered reproduction of one key result from:

> Gao H., Wang N., Wan Z., Qu R., Wang Y., Xu L., Meng X. (2026). *Compact and Explainable Human Activity Recognition with Boruta-RFE Feature Selection and SHAP Interpretation.* Applied Sciences 16(20):10046. https://doi.org/10.3390/app162010046

**Target result (paper, Table 1):** on the official subject-disjoint UCI HAR test set, XGBoost scores
- **95.55%** with all 561 features;
- **93.89%** with the authors' 100-feature subset.

That is a **1.66 percentage-point** cost for 82.17% fewer input columns.

**What this repository reproduces:** refits of the two published *final* XGBoost configurations, using the paper's published feature list (Supplementary Table S2).

**What it does not reproduce:** the Boruta → RFE feature *selection* procedure, the grouped-CV tuning, or the SHAP analysis.

**Conditional result:** nine of the 100 published feature names correspond to three columns each. The authors' choice cannot be recovered, so the compact result depends on a pre-registered mapping rule (see [docs/FEATURE_MAPPING.md](docs/FEATURE_MAPPING.md)).

> **Status: Phase 1 (data pipeline) complete. No model has been trained and there are no results yet.**

## Quick start

Requirements: Python 3.11, about 0.5 GB of free disk space, and a network connection for the first data download.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps

# 1. Download (or reuse) the official UCI HAR archive, verify its SHA-256, extract, and verify file hashes
python -m har_repro.cli prepare-data
#    ...or use an archive you already have (accepted only if its checksum matches):
python -m har_repro.cli prepare-data --local-archive "path/to/human+activity+recognition+using+smartphones.zip"

# 2. Check that the committed 100-feature mapping regenerates exactly from the official features.txt
python -m har_repro.cli build-mapping --check

# 3. Structural data validation report (fits no model, makes no predictions)
python -m har_repro.cli validate-data

# 4. Tests (synthetic plus structural checks on the real data)
python -m pytest
```

The data goes under `./data` (git-ignored). Set `HAR_DATA_DIR` to use another location. The download tries the official UCI legacy path, then the official current URL, then an Internet Archive capture of the official URL. A file is accepted only if its SHA-256 matches; see [docs/DATA.md](docs/DATA.md).

## Repository layout

```
configs/
  experiment.toml                     frozen protocol parameters (models, seeds, bootstrap, decision rule)
  data_sources.toml                   download sources, archive and file SHA-256 checksums
  gao2026_supplementary_table_S2_100_features.csv   paper's S2 (byte-exact copy, CC BY 4.0)
  compact100_mapping.json             generated name -> index mapping (X primary; Y, Z sensitivity)
src/har_repro/
  cli.py                              prepare-data | build-mapping | validate-data
  config.py, paths.py
  data/acquire.py                     download, checksum, safe extraction, file verification
  data/loader.py                      official 561-feature loader, label encoding, identical-column detection
  features/mapping.py                 S2 name -> column mapping (ascending original-index order)
  models/xgb_params.py                frozen config -> xgboost.train parameters (no fitting)
tests/                                pytest suite; tests marked `data` need prepared data
docs/                                 PROTOCOL, FEATURE_MAPPING, DATA, ENVIRONMENT
results/                              outputs (only data validation so far)
```

## Protocol

The pre-registered protocol is in [docs/PROTOCOL.md](docs/PROTOCOL.md) and is machine-readable in `configs/experiment.toml`. In summary:
- the official split only;
- both final configurations with `tree_method="exact"`, `base_score=0.5` and 1 thread;
- five paired seeds (2026–2030);
- a participant-level paired bootstrap (10,000 replicates, seed 1729);
- outcome categories fixed in advance.

The primary implementation is Python xgboost 3.2.0. The paper used R xgboost 1.7.7.1, so this is a cross-language approximation and not a bit-identical replay. Python xgboost 1.7.6 is reserved for an optional, separately labelled version-parity sensitivity check.

## Licences and attribution

- **Code:** MIT.
- **UCI HAR dataset:** CC BY 4.0 (Reyes-Ortiz et al., DOI 10.24432/C54S4K). It is downloaded, not redistributed.
- **Supplementary Table S2:** © the authors, CC BY 4.0, from https://www.mdpi.com/article/10.3390/app162010046/s1.
