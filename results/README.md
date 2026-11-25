# Results

**Provenance:**
- Produced by `python -m har_repro.cli run-experiments` at commit `ca2b88017070d0bdbbbfb02129b7ffc60427edf3` (recorded in `manifest.json`), on 25 Nov 2026.
- The pre-registered protocol was frozen at commit `e14633759ae9a82b51617e70fd2613ff2059901e`.

All files are machine-generated and have not been edited by hand. `python -m har_repro.cli verify-results` re-derives the metrics and summaries from the saved predictions and checks them against these files.

| Path | Content |
|---|---|
| `manifest.json` | Git commit, environment, config hashes and run plan, recorded before any test prediction |
| `runs/<run_id>/predictions.csv` | Test row, participant ID, true and predicted labels, correctness, six class probabilities |
| `runs/<run_id>/proba.npy` | Exact float32 probabilities |
| `runs/<run_id>/metrics.json` | Counts, accuracy, macro-F1, paper-definition balanced accuracy, macro-AUC, confusion matrix, per-class and per-participant results |
| `runs/<run_id>/run_info.json` | Seed, columns, fit and predict time, explicit parameters and the full effective booster configuration |
| `summary/results_table.md` | Paper vs reproduction table (seed-2026 reference and five-seed statistics) and the exploratory mapping sensitivity |
| `summary/summary.json` | Per-seed and five-seed statistics, and the sensitivity runs |
| `summary/bootstrap.json` | Paired participant-level bootstrap for the **seed-2026 model pair** (10,000 replicates, seed 1729) |
| `summary/decision.json` | Pre-registered outcome category and the supplementary five-seed-mean classification |
| `repeatability/` | Fresh-process re-runs of both primary models at seed 2026, and the bitwise comparison |
| `data_validation/data_validation.json` | Phase 1 structural data checks |

**Run identifiers:**
- Primary: `full561__seedN` and `compact100-X__seedN`.
- Exploratory sensitivity: `compact100-Y__seedN` and `compact100-Z__seedN`.

**Not committed:**
- trained boosters (`model.json`, 1.7–2 MB each; re-creatable from code and seed);
- the local console log.
