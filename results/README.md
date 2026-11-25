# Results

`data_validation/data_validation.json` contains Phase 1 structural checks. No model was fitted and no prediction was made when producing it.

**There are no model results yet.**

`python -m har_repro.cli run-experiments` writes the following. All files are machine-generated and never edited by hand.

| Path | Content |
|---|---|
| `manifest.json` | Git commit, environment, config hashes and run plan, recorded before any test prediction |
| `runs/<run_id>/predictions.csv` | Test row, participant, true and predicted labels, correctness, class probabilities |
| `runs/<run_id>/proba.npy` | Exact float32 probabilities, used for bitwise repeatability checks |
| `runs/<run_id>/metrics.json` | Counts, accuracy, macro-F1, paper balanced accuracy, macro-AUC, confusion matrix, per-class and per-participant results |
| `runs/<run_id>/run_info.json` | Seed, columns, timing, explicit parameters and the full effective booster configuration |
| `runs/<run_id>/model.json` | Trained booster (git-ignored; re-creatable from code and seed) |
| `summary/summary.json`, `bootstrap.json`, `decision.json`, `results_table.md` | Aggregates, participant bootstrap, outcome category and results table |
| `repeatability/repeatability.json` | Comparison against a fresh-process re-run at seed 2026 |
