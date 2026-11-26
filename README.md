# Reproducing Gao et al. (2026): the accuracy cost of a 100-feature HAR representation

This is an independent, pre-registered reproduction of the subject-independent accuracy cost reported for replacing all 561 engineered UCI HAR features with a published 100-feature subset in XGBoost activity recognition (Gao et al., *Applied Sciences*, 2026).

### **[Read the Evidence Note (PDF)](paper/evidence_note.pdf)**

The one-page scientific evidence note covers:
- the research question and why this paper was chosen;
- the three main methodological decisions, each with the alternative not chosen;
- the measured results against the published ones, with their uncertainty and limitations;
- the proposed next experiment.

Editable Typst source: [`paper/evidence_note.typ`](paper/evidence_note.typ). Build steps are in [Rebuild the evidence note](#5-rebuild-the-evidence-note).

## Results summary

| Quantity | Published | This work, seed 2026 (reference) | This work, five seeds (mean ± SD) |
|---|---:|---:|---:|
| Full-561 accuracy | 95.55% (2816/2947) | 95.25% (2807/2947) | 95.44 ± 0.16% |
| Compact-100 accuracy (X-first) | 93.89% (2767/2947) | 94.16% (2775/2947) | 93.70 ± 0.27% |
| Δ, full − compact | +1.66 pp | +1.09 pp | +1.74 ± 0.38 pp |
| 95% participant-bootstrap CI of Δ | [+0.49, +3.18] | [+0.29, +2.11] | — |
| Full-561 macro-F1 | 0.9550 | 0.9519 | 0.9539 |
| Compact-100 macro-F1 | 0.9384 | 0.9413 | 0.9365 |

**Reading the table:**
- The bootstrap CI applies **only to the seed-2026 model pair**. Seed-to-seed variation is the SD.
- Under the pre-registered criterion, the agreement is *Strong*: both reference-seed accuracies are within ±1.0 pp of the published values, and Δ > 0 lies within ±1.0 pp of +1.66 pp. The five-seed means meet the same criterion.
- This is a descriptive label, not an equivalence test.
- All published values lie inside the five-seed ranges. Per-seed values are in [results/summary/results_table.md](results/summary/results_table.md).

**Exploratory mapping sensitivity** (not part of the primary analysis):

| Compact-100 mapping | Accuracy (five seeds) | Mean Δ |
|---|---:|---:|
| X-first (primary) | 93.70 ± 0.27% | +1.74 pp |
| Y axis | 93.97 ± 0.10% | +1.47 pp |
| Z axis | 94.29 ± 0.12% | +1.15 pp |

Δ is positive in all 15 compact runs. Its size depends on the mapping.

## Installation, reproduction and verification

**Requirements:**
- Python 3.11 (the reported run used 3.11.3).
- Git, because `run-experiments` records and checks the commit.
- About 1 GB of free disk space for the environments and the extracted dataset.
- Internet access for the first data download.
- A CPU is enough: training uses one thread and no GPU.

All commands run from the repository root. The commands differ between shells only when creating and activating the environments; every other step is identical in both.

### 1. Environment

Linux / macOS (bash or zsh):

```bash
git clone https://github.com/ikram-gafai/gao2026-har-reproduction.git
cd gao2026-har-reproduction
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

Windows (PowerShell):

```powershell
git clone https://github.com/ikram-gafai/gao2026-har-reproduction.git
cd gao2026-har-reproduction
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process RemoteSigned` once in that session.

### 2. Data

```bash
python -m har_repro.cli prepare-data
python -m har_repro.cli build-mapping --check
python -m har_repro.cli validate-data
```

- `prepare-data` downloads the archive, verifies its SHA-256 and extracts it into `./data`.
  - It tries three sources in order: the official UCI legacy path, the official current URL, then an Internet Archive capture of the official URL. A file is accepted only if its SHA-256 matches.
  - `--local-archive PATH` accepts an existing official ZIP if its checksum matches.
  - The `HAR_DATA_DIR` environment variable changes the data location.
- `build-mapping --check` requires the committed feature mapping to regenerate exactly from `features.txt`.
- `validate-data` writes a structural report and fits no model. It rewrites `results/data_validation/data_validation.json`; on a correct dataset, only the `generated_utc` timestamp changes.

### 3. Verify the committed evidence

```bash
python -m pytest
python -m har_repro.cli verify-results
python -m har_repro.cli verify-results --refit compact100-X__seed2026
python -m har_repro.cli verify-results --refit full561__seed2026
```

`pytest` runs the 80 tests. `verify-results` audits the committed outputs in a few seconds:
- it recomputes every run's metrics from the saved predictions;
- it checks the labels and participant IDs against the official test split;
- it checks that the saved predictions agree with the saved probabilities;
- it regenerates the summary, the 10,000-replicate bootstrap, the outcome decision and the results table, and requires them to match the committed files byte for byte;
- it checks the repeatability record.

`--refit RUN_ID` retrains that run from the public data and requires bit-identical probabilities. It took about 40 s for the compact model and about 6.5 min for the full model on one thread.

### 4. Re-run the full experiment (optional)

```bash
python -m har_repro.cli run-experiments --results-dir rerun_results
```

This runs all 20 pre-registered fits:
- full and compact models on seeds 2026–2030;
- the exploratory Y and Z mappings on the same seeds.

It then writes the summary, bootstrap and decision to `rerun_results/`, and runs a fresh-process repeatability check. It refuses to start on unverified data, with uncommitted changes to the source code, configurations or tests, or if the target folder already holds runs. On the reference machine, fit times were 313–386 s for each full model and 32–52 s for each compact model, all on one thread (`fit_seconds` in each `run_info.json`).

### 5. Rebuild the evidence note

The note uses a separate environment with matplotlib, typst and pypdf. Typst is installed from PyPI, so no LaTeX installation is needed. Run this in a new terminal from the repository root.

Linux / macOS:

```bash
python3.11 -m venv .venv-paper
source .venv-paper/bin/activate
python -m pip install -r paper/requirements-paper.txt
python paper/build_note.py
python paper/check_numbers.py
```

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv-paper
.\.venv-paper\Scripts\Activate.ps1
python -m pip install -r paper/requirements-paper.txt
python paper/build_note.py
python paper/check_numbers.py
```

- `build_note.py` regenerates both figures from `results/summary/` and compiles `paper/evidence_note.typ` into `paper/evidence_note.pdf`. It checks that the result is one A4 page under 10 MB.
- `check_numbers.py` requires every number in the PDF to trace to the committed results, the frozen configuration or the paper.

## Methodology

### Paper reproduced

> H. Gao, N. Wang, Z. Wan, R. Qu, Y. Wang, L. Xu, X. Meng. *Compact and Explainable Human Activity Recognition with Boruta-RFE Feature Selection and SHAP Interpretation.* Applied Sciences 16(20):10046, 2026. https://doi.org/10.3390/app162010046

The paper reports results on the official subject-disjoint UCI HAR test set:

| Model | Test accuracy |
|---|---:|
| XGBoost, all 561 engineered features | 95.55% |
| XGBoost, published 100-feature subset | 93.89% |
| Difference, Δ = Acc<sub>full</sub> − Acc<sub>compact</sub> | **+1.66 percentage points (pp)** |

The subset removes 82.17% of the input columns. This repository refits both published final configurations from the public data and measures that comparison under a protocol committed before any model was fitted.

**Why this paper:**
- it was published in October 2026;
- its data are public and de-identified;
- its central comparison is reported numerically, with the final hyperparameters and the 100 feature names (Supplementary Table S2);
- a documented search found no author-released code.

### Scope

**Reproduced:**
- the final full-feature and compact-feature XGBoost classifiers, using the published configurations;
- their test-accuracy comparison on the official split;
- five pre-declared training seeds;
- a paired participant-level bootstrap of Δ.

**Not reproduced:**
- Boruta filtering and recursive feature elimination (RFE), with their grouped cross-validation;
- the original hyperparameter search;
- the SHAP analysis.

Agreement here therefore concerns the published classifiers built on the published feature list. It does not show that the selection procedure would recover that list.

### Data and protocol

- **Dataset:** UCI HAR, Reyes-Ortiz et al., [doi:10.24432/C54S4K](https://doi.org/10.24432/C54S4K), CC BY 4.0. It covers 30 adults wearing a waist-mounted smartphone, six activities, and 2.56 s windows with 50% overlap, each described by 561 precomputed features.
- **Split:** the official participant-disjoint split, with 7,352 training windows from 21 participants and 2,947 test windows from 9 participants. The test set is never used for fitting, tuning or any mapping decision.
- **Integrity:** archives and extracted files are checked against SHA-256 values ([docs/DATA.md](docs/DATA.md)).
- **Models:** native `xgboost.train` with exact greedy splits, `base_score = 0.5`, one thread, `multi:softprob` and six classes.

  | | Full-561 | Compact-100 |
  |---|---:|---:|
  | Input columns | 561 | 100 (published S2 names) |
  | Boosting rounds | 400 | 300 |
  | `max_depth` / `eta` / `subsample` | 3 / 0.1 / 0.8 | 3 / 0.1 / 0.8 |
  | `colsample_bytree` | 1.0 | 0.8 |

  The two published configurations differ in more than their inputs, so Δ is a comparison of the published configurations. It does not isolate the effect of removing features.
- **Nine ambiguous feature names:**
  - Nine S2 names (`bandsEnergy()` bands 1–8, 1–16 and 1–24 of fBodyAcc, fBodyAccJerk and fBodyGyro) occur three times in `features.txt`, once per sensor axis. No indices were published.
  - The pre-registered primary mapping uses the first occurrence (X axis). Mappings to Y and Z are reported only as exploratory sensitivity analyses.
  - X-first is a convention, not the authors' confirmed columns ([docs/FEATURE_MAPPING.md](docs/FEATURE_MAPPING.md)).
- **Seeds and uncertainty:**
  - Seeds 2026–2030 were declared in advance, with 2026 as the reference.
  - A paired participant-level bootstrap resamples the 9 test participants with their windows pooled (10,000 replicates, seed 1729).
  - The full protocol is in [docs/PROTOCOL.md](docs/PROTOCOL.md) and is machine-readable in [configs/experiment.toml](configs/experiment.toml).

## Limitations

1. **Mapping:** the compact input depends on the X-first convention for nine names.
2. **Identical columns:** five pairs among the 100 selected columns are numerically identical (95 distinct inputs), so SHAP attributions between those pairs are not identifiable.
3. **Implementation:** Python xgboost 3.2.0 approximates the authors' R xgboost 1.7.7.1, whose tree method and random state were not reported. Random draws cannot be matched across languages.
4. **Seed variation:** Δ ranges from +1.09 to +2.07 pp across the five seeds.
5. **Sample size:** nine test participants make every interval coarse.
6. **Data:** UCI HAR is a controlled laboratory dataset and supports no inference about free-living or clinical use.
7. **Confounded comparison:** the two configurations differ in boosting rounds and column subsampling as well as in their inputs.

## Repository layout

```
configs/      experiment.toml (frozen protocol parameters) · data_sources.toml (URLs, SHA-256)
              compact100_mapping.json (S2 names -> column indices) · S2 supplementary table (CC BY 4.0)
src/har_repro/
  cli.py                  prepare-data | build-mapping | validate-data | run-experiments | run-one | summarize | verify-results
  data/                   checksum-verified acquisition and safe extraction; official 561-feature loader
  features/mapping.py     S2 name -> column mapping (ascending original-index order)
  models/                 frozen config -> xgboost parameters; training on the training partition only
  evaluation/             metrics, paired participant bootstrap, pre-registered outcome rule
  experiments/            run plan, manifest, summary, repeatability check, results audit
tests/        80 tests: synthetic implementation tests and structural checks on the real data
results/      manifest.json · runs/ (20 runs: predictions, probabilities, metrics, configs) · summary/
              repeatability/ · data_validation/ (see results/README.md)
paper/        evidence_note.pdf + evidence_note.typ (Typst source) · make_figures.py · build_note.py
              check_numbers.py (audits every number in the PDF) · requirements-paper.txt
docs/         PROTOCOL.md · FEATURE_MAPPING.md · DATA.md · ENVIRONMENT.md
```

## Environment of the reported run

- **Hardware:** AMD Ryzen 7 8845HS with 16 logical CPUs (one used for training) and 15.2 GB RAM, running Windows 11. CPU only.
- **Software:** Python 3.11.3, xgboost 3.2.0, numpy 2.4.6, pandas 3.1.1, scikit-learn 1.9.1 and scipy 1.17.1. The full lockfile is [`requirements-lock.txt`](requirements-lock.txt); see also [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Provenance

| Commit | Content | Original commit ID | Tree (identical) |
|---|---|---|---|
| `ab93dfe16baa189523916476288cfe56fe4cc406` | Pre-registration: protocol, frozen configuration, data pipeline and mapping, committed before any model was fitted | `e14633759ae9a82b51617e70fd2613ff2059901e` | `ad4a3e82206f0593669a7556c950a91d6c9cfc58` |
| `2678acadd135a4bc751a4c32e1d279976b2a3c51` | Frozen implementation that produced the results | `ca2b88017070d0bdbbbfb02129b7ffc60427edf3` | `c798fc82ee8eb64d45a9c63a16d6fea51e0cca01` |
| `be86b6d2ad4c1673cbbd6864377e413d8272e373` | Unedited experiment outputs and the audit command | `6039ca6b1bfe78c8dd47cff21943c8037cd10517` | `9fd9c4cf357759a40ccda3809a90f529012e8ea5` |

**Original commit IDs:**
- These three commits were re-created on 26 Nov 2026, before the repository was first published, to correct their author metadata.
- Their file trees and timestamps are unchanged. Only the commit IDs differ.
- `results/manifest.json` was written by the experiment run and is not edited, so it still cites the original implementation ID `ca2b8801`.
- To confirm that the code is identical, check that `git rev-parse 2678acad^{tree}` prints `c798fc82ee8eb64d45a9c63a16d6fea51e0cca01`.

Trained boosters (`results/runs/*/model.json`) are not committed. They can be re-created from the code and seeds.

## Citation and licences

**Paper:** H. Gao, N. Wang, Z. Wan, R. Qu, Y. Wang, L. Xu, X. Meng. Compact and explainable human activity recognition with Boruta-RFE feature selection and SHAP interpretation. *Applied Sciences* 16(20):10046, 2026. doi:[10.3390/app162010046](https://doi.org/10.3390/app162010046).

**Dataset:** J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto, X. Parra. Human Activity Recognition Using Smartphones [dataset]. UCI Machine Learning Repository, 2013. doi:[10.24432/C54S4K](https://doi.org/10.24432/C54S4K).

- **Code:** MIT ([LICENSE](LICENSE)).
- **UCI HAR dataset:** CC BY 4.0, Reyes-Ortiz, Anguita, Ghio, Oneto and Parra. It is downloaded, not redistributed.
- **Supplementary Table S2** (`configs/gao2026_supplementary_table_S2_100_features.csv`): © the authors of Gao et al. (2026), CC BY 4.0, from https://www.mdpi.com/article/10.3390/app162010046/s1.
