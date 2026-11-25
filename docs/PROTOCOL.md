# Pre-registered reproduction protocol (v2 + Addendum A1)

Frozen on 25 Nov 2026, before any model was fitted or any test-set prediction was made. The machine-readable form is `configs/experiment.toml`, and `tests/test_config.py` asserts that it agrees with this document.

## 1. Paper, question and boundary

**Paper:** Gao H., Wang N., Wan Z., Qu R., Wang Y., Xu L., Meng X. (2026). *Compact and Explainable Human Activity Recognition with Boruta-RFE Feature Selection and SHAP Interpretation.* Applied Sciences 16(20):10046. https://doi.org/10.3390/app162010046 (CC BY 4.0).

**Question:** does the authors' published 100-feature representation keep most of the subject-independent test accuracy of XGBoost trained on all 561 UCI HAR features?

**Published targets (Table 1):**

| Model | Correct / total | Accuracy |
|---|---:|---:|
| Full XGBoost, 561 columns | 2,816 / 2,947 | 95.55% |
| Compact XGBoost, 100 columns | 2,767 / 2,947 | 93.89% |
| Full − Compact | | +1.66 pp |
| Participant-bootstrap 95% CI of the gap | | +0.49 to +3.18 pp |

**Feature-count reduction:** (561 − 100) / 561 = 82.17% of *input columns*. Five pairs of the 100 columns are numerically identical, so there are 95 distinct vectors (see `FEATURE_MAPPING.md`).

**Scope:** an independent refit of the two published *final* XGBoost configurations, using the paper's published feature names. This does **not** reproduce Boruta filtering, RFE, grouped-CV tuning or the SHAP analysis. The compact result is **conditional on a declared mapping** for nine names that have no unique column index.

## 2. Data and split

- Official UCI HAR files, checksum-verified (`DATA.md`).
- Official subject-disjoint split: 7,352 training windows from 21 participants, 2,947 test windows from 9 participants.
- No re-splitting, no pooling of train and test, and no imputation, scaling, selection or tuning fitted on test data.
- The features are used as distributed (already in [−1, 1]). The upstream normalization scope is unknown and is disclosed as a limitation.
- Labels: original 1–6 are encoded as 0–5. Activity names are kept for reports.

## 3. Compact feature mapping

- **91 names** map to a single original index.
- **9 `bandsEnergy` names** occur three times each. The primary rule takes the **first occurrence (X block)**. This convention was fixed before modelling, not chosen by performance.
- Training-data correlations support the X/Y/Z identity of the three blocks. They do not reveal which block the authors used.
- Compact columns are ordered by **ascending original index**, not by the SHAP ranking in S2. Column order affects what `colsample_bytree` samples for a given seed.
- **Exploratory sensitivity:** all nine moved together to the Y block, and separately to the Z block. Every result is reported. These runs are never used to choose the primary mapping.
- There are 3⁹ = 19,683 possible mappings. No outcome-guided search is allowed.

## 4. Models (both explicit; native `xgboost.train`)

| Parameter | Full-561 | Compact-100 |
|---|---:|---:|
| Boosting rounds | 400 | 300 |
| `max_depth` | 3 | 3 |
| `eta` | 0.10 | 0.10 |
| `colsample_bytree` | 1.0 | 0.8 |
| `subsample` | 0.8 | 0.8 |
| `gamma` | 0 | 0 |
| `min_child_weight` | 1 | 1 |
| `base_score` | 0.5 | 0.5 |
| `tree_method` | exact | exact |
| `objective` | multi:softprob | multi:softprob |
| `num_class` | 6 | 6 |
| Threads | 1 | 1 |

**Settings the paper does not report, which are therefore assumptions here:**
- objective;
- tree method;
- random state and fold assignments;
- some R implementation defaults;
- normalization scope.

**XGBoost version, decided before any experiment:**
- **Primary implementation:** Python xgboost **3.2.0** with the native `xgboost.train` API, exact greedy, `base_score=0.5`, 1 thread, and the frozen seeds.
- The original paper used **R xgboost 1.7.7.1** (R 4.3.3).
- **Optional version-parity sensitivity:** Python xgboost **1.7.6**, from the same 1.7 release series. If run, it is labelled separately. It never replaces the primary implementation and is never chosen on test performance.

**Implementation differences and how they are handled:**
- The original analysis used R 4.3.3 with xgboost 1.7.7.1. This Python reproduction approximates it; it is not a bit-identical replay.
- R xgboost ignores the `seed` parameter and uses R's `set.seed()` (xgboost 1.7.0 parameter documentation). Equal integer seeds therefore do **not** mean equal random draws.
- Current xgboost estimates a class-specific intercept unless `base_score` is a real number. Version 1.7 used a uniform 0.5, so `base_score=0.5` is pinned here.
- Parameters not listed above keep library defaults. They are recorded from each trained booster's saved configuration.

## 5. Seeds and repeatability

- **Main:** seeds 2026, 2027, 2028, 2029 and 2030, for both models (paired). All runs are reported with their mean and SD and the distribution of the paired gap.
  - Seed 2026 is the pre-declared reference pair. It is not selected by score.
- **Sensitivity (A1-R3):** the Y and Z compact variants on the same five seeds.
- **Repeatability (A1-R3):** both primary models at seed 2026 are re-run in a fresh process. Predicted labels and probabilities must be bit-identical. Any non-determinism is reported, not tuned away.

## 6. Evaluation

**Before any test prediction:**
- Freeze the data checksums, index manifests, configurations, versions, threads, seeds, evaluation code and outcome categories (commit or manifest).
- No tuning, axis selection or seed selection may use test performance.

**For each run, record:**
- number correct and accuracy;
- macro-F1 (unweighted mean of class F1);
- confusion matrix and per-class recall;
- per-participant correct/total;
- the paired Full − Compact gap.

**Balanced accuracy,** if reported, uses the paper's definition: the macro mean of one-vs-rest (sensitivity + specificity)/2. That is not scikit-learn's `balanced_accuracy_score`.

**Participant bootstrap (A1-R2):**
- Seed-2026 model pair.
- 9 test participants drawn with replacement, all their windows pooled and counted at their multiplicity.
- Both accuracies computed on the same pooled sample, then the paired difference.
- 10,000 replicates, percentile 2.5% / 97.5%, **bootstrap seed 1729**.
- Nine participants limit the precision of this interval.

## 7. Outcome categories (A1-R1: evaluated in this order; the first match decides)

1. **Not numerically reproduced:** the seed-2026 Full − Compact gap is ≤ 0 pp, whatever else holds.
2. **Strong numerical agreement (conditional compact reproduction):** both seed-2026 accuracies are within ±1.0 pp of the published values, **and** the seed-2026 gap is positive and within ±1.0 pp of +1.66 pp. Whether the five-seed means also meet these thresholds is reported alongside.
3. **Partial numerical reproduction:** Full-561 is within ±1.0 pp of 95.55%, **or** at least one model is within ±2.0 pp of its published value with a positive gap. The report states which sub-result matched.
4. **Not numerically reproduced:** otherwise.

These thresholds are descriptive labels fixed before evaluation. They are not tests of equivalence. The substantive evidence is the counts, the paired gap, the participant-bootstrap interval, the sensitivity runs and the disclosed assumptions.

## 8. Explicit non-goals

- No re-derivation of Boruta → RFE.
- No SHAP ranking reproduction. SHAP is not identifiable between identical columns.
- No hyperparameter search.
- No claim that the compact columns are the authors' exact columns.
