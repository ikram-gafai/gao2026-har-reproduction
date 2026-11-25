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

> **Status:** the pre-registered experiments have been run.
> - Pre-registration: commit `e1463375`.
> - Frozen implementation that produced the results: commit `ca2b8801`.
> - Outcome under the pre-registered criteria: **strong numerical agreement**. The compact result is conditional on the declared feature mapping.

## Results (pre-registered primary analysis)

**Setup:**
- Official subject-disjoint UCI HAR test set: 2,947 windows from 9 participants.
- Compact model uses the pre-registered **X-first** mapping.
- Paper values are from Gao et al. (2026), Table 1 and Section 3.2.
- Seed 2026 was declared as the reference before any run. The five-seed column covers seeds 2026–2030; no seed was selected by score.

| Metric | Paper | Reproduction, seed 2026 (reference) | Five-seed mean ± SD |
|---|---:|---:|---:|
| Full-561 accuracy | 95.55% (2816/2947) | 95.25% (2807/2947) | 95.44 ± 0.16% |
| Compact-100 accuracy | 93.89% (2767/2947) | 94.16% (2775/2947) | 93.70 ± 0.27% |
| **Full − Compact accuracy gap** | **+1.66 pp** | **+1.09 pp** | **+1.74 ± 0.38 pp** |
| Full-561 macro-F1 | 0.9550 | 0.9519 | 0.9539 |
| Compact-100 macro-F1 | 0.9384 | 0.9413 | 0.9365 |
| Full-561 balanced accuracy (paper's one-vs-rest definition) | 0.9726 | 0.9707 | 0.9719 |
| Compact-100 balanced accuracy (paper's definition) | 0.9625 | 0.9642 | 0.9614 |
| Full-561 macro-AUC (one-vs-rest) | 0.9984 | 0.9982 | 0.9984 |
| Compact-100 macro-AUC (one-vs-rest) | 0.9972 | 0.9973 | 0.9972 |
| Input columns | 561 → 100 (−82.17%) | 561 → 100 (−82.17%); 540 → 95 distinct vectors | — |

**Uncertainty of the gap.** The participant-level paired bootstrap 95% CI is **+0.29 to +2.11 pp** for the **seed-2026 model pair only**, against the paper's +0.49 to +3.18 pp.
- Settings: 10,000 resamples of the 9 test participants, seed 1729.
- This interval describes participant sampling for that one pair of fitted models. It is **not** an interval around the five-seed mean.
- Seed-to-seed variation is reported separately as the SD above.

**Pre-registered outcome:** *strong numerical agreement (conditional compact reproduction)*.
- At seed 2026, both accuracies are within ±1.0 pp of the paper, and the gap is positive and within ±1.0 pp of +1.66 pp.
- The five-seed means also meet these thresholds.
- Per-seed results and the machine-generated table are in [`results/summary/results_table.md`](results/summary/results_table.md).

**Exploratory mapping sensitivity** (not part of the primary analysis, and not used to choose a mapping). Here all nine ambiguous features are moved together to another sensor-axis block:

| Compact-100 mapping | Accuracy, mean ± SD over seeds 2026–2030 | Mean Full − Compact gap |
|---|---:|---:|
| X-first (primary) | 93.70 ± 0.27% | +1.74 pp |
| Y block | 93.97 ± 0.10% | +1.47 pp |
| Z block | 94.29 ± 0.12% | +1.15 pp |

The gap is positive in all 15 compact runs. Its size, however, depends on which columns the nine ambiguous names refer to.

## Interpreting the result: limitations

- **Nine ambiguous features.**
  - Nine of the paper's 100 feature names (`bandsEnergy` bands 1–8, 1–16 and 1–24 of fBodyAcc, fBodyAccJerk and fBodyGyro) occur three times in `features.txt`, once per sensor-axis block. The paper gives no index or axis.
  - X-first is a **pre-registered convention, not the authors' confirmed mapping**.
  - The exploratory Y and Z mappings change mean compact accuracy by up to +0.58 pp.
  - The *direction* of the accuracy–compactness trade-off is robust; its *size* is conditional on the mapping.
  - See [docs/FEATURE_MAPPING.md](docs/FEATURE_MAPPING.md).
- **Five identical column pairs.**
  - Five pairs among the 100 selected columns are numerically identical (for example `tBodyAccMag-std()` and `tGravityAccMag-std()`), so the compact model has 95 distinct inputs.
  - The paper's 82.17% reduction counts named columns.
  - SHAP importance between identical columns is not identifiable. This reproduction does not reproduce the SHAP analysis.
- **R vs Python.**
  - The paper used R 4.3.3 with R xgboost 1.7.7.1 and did not report the tree method, objective or random state.
  - This reproduction uses Python xgboost 3.2.0 with exact greedy splitting, `base_score=0.5` and one thread.
  - Random draws cannot be matched across languages, because R xgboost uses R's RNG.
  - The observed differences from the paper (at most 0.57 pp, for the gap) are the same size as our seed-to-seed variation (SD 0.16–0.38 pp). The two sources cannot be separated with these runs.
- **Nine test participants.**
  - Each interval rests on 9 people.
  - At seed 2026 the gap is concentrated in a few participants: +3.74, +3.64 and +2.08 pp for participants 10, 2 and 9, against at most +0.62 pp for the others (`results/runs/*/metrics.json`, `per_participant`).
- **Scope.**
  - This reproduces the final classifiers using the published feature list. It does **not** re-derive Boruta → RFE selection, the grouped-CV tuning or the SHAP analysis.
  - UCI HAR is laboratory data from 30 adults.

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

### Running the pre-registered experiments

```bash
python -m har_repro.cli run-experiments
```

This single command does the following:
1. **Checks before it starts.** It refuses to run if the data fails its checksums, if `src/`, `configs/` or `tests/` have uncommitted changes, or if `results/runs/` already contains runs.
2. **Records a manifest.** `results/manifest.json` captures the git commit, package versions, config hashes and the run plan *before any test prediction*.
3. **Runs the 20 pre-registered fits.** Each one trains on the official training partition only and predicts the test partition once:
   - Full-561 and Compact-100 (X mapping) on seeds 2026–2030;
   - the exploratory Compact-100 Y and Z mappings on the same seeds.
4. **Writes the summary.** `results/summary/` gets the five-seed statistics, the paired participant bootstrap (seed-2026 pair, 10,000 replicates, seed 1729), the pre-registered outcome category, and a machine-generated `results_table.md`.
5. **Checks repeatability.** It re-runs both primary models at seed 2026 in a fresh process and requires bit-identical probabilities and labels (`results/repeatability/repeatability.json`).

### Checking the committed results

```bash
python -m har_repro.cli verify-results                                   # audit saved outputs (a few seconds)
python -m har_repro.cli verify-results --refit compact100-X__seed2026    # also retrain one run (about 1 min)
```

`verify-results` audits the committed outputs without trusting them:
- It recomputes every run's metrics from the saved predictions.
- It checks labels and subject IDs against the official test partition.
- It checks that predictions match the saved probabilities.
- It regenerates the summary, the 10,000-replicate participant bootstrap, the decision and the results table, and requires them to be byte-identical to the committed files.
- It checks the repeatability record.
- With `--refit RUN_ID` it retrains that configuration from the public data and requires bit-identical probabilities.

Re-running `run-experiments` into a new `--results-dir` regenerates everything. On the reference machine, with one training thread, the recorded fit times (`fit_seconds` in each `run_info.json`) are:
- full-561: 313–386 s per run;
- compact-100: 32–52 s per run;
- all 20 runs together: about 42 min, plus the repeatability re-fits.

Other commands:
- `python -m har_repro.cli summarize` recomputes the summary from saved runs.
- `python -m har_repro.cli run-one --model {full561|compact100} [--variant X|Y|Z] --seed N --out DIR` fits a single configuration.

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
  models/xgb_params.py                frozen config -> xgboost.train parameters
  models/train.py                     fit on training partition only; predict probabilities and labels
  evaluation/metrics.py               accuracy, macro-F1, paper balanced accuracy, macro-AUC, confusion, per-participant
  evaluation/bootstrap.py             paired participant-level (cluster) bootstrap of the accuracy gap
  evaluation/decision.py              pre-registered outcome categories (A1-R1 precedence)
  experiments/runner.py               run plan, per-run outputs, manifest, summary, repeatability check
  experiments/verify.py               audit of saved results (verify-results)
tests/                                pytest suite; tests marked `data` need prepared data
docs/                                 PROTOCOL, FEATURE_MAPPING, DATA, ENVIRONMENT
results/                              manifest, 20 runs (predictions, probabilities, metrics, configs), summary,
                                      bootstrap, decision, repeatability (see results/README.md)
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
