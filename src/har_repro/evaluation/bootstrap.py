"""Paired participant-level (cluster) bootstrap of the Full - Compact accuracy difference.

Protocol (v2 section 6, Addendum A1-R2):
- each replicate draws as many participants as there are test participants (9), with replacement;
- every window of each drawn participant is pooled, counted once per time the participant is drawn;
- both models' accuracies are computed on that same pooled sample, and their difference is taken;
- 10,000 replicates, percentile 2.5% / 97.5%, fixed seed 1729.

Generator: numpy.random.default_rng(seed) (PCG64). Draws: rng.integers(0, k, size=(B, k)) over participants
sorted by id. Percentiles: numpy.percentile with the default 'linear' method.
Pooled accuracy is computed from per-participant counts: sum_s m_s * correct_s / sum_s m_s * n_s, where m_s is
the number of times participant s was drawn. This equals concatenating the drawn participants' windows
(tests/test_bootstrap.py checks it against an explicit concatenation).
"""

from __future__ import annotations

import numpy as np


def participant_counts(subjects: np.ndarray, correct: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.array([np.sum(subjects == s) for s in ids], dtype=np.int64)
    c = np.array([np.sum(correct[subjects == s]) for s in ids], dtype=np.int64)
    return n, c


def draw_participants(k: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, k, size=(replicates, k))


def paired_participant_bootstrap(subjects: np.ndarray, correct_a: np.ndarray, correct_b: np.ndarray,
                                 replicates: int, seed: int, percentiles=(2.5, 97.5)) -> dict:
    """Bootstrap of 100 * (accuracy_a - accuracy_b) in percentage points (a = Full, b = Compact)."""
    subjects = np.asarray(subjects)
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    if not (len(subjects) == len(correct_a) == len(correct_b)):
        raise ValueError("subjects and correctness vectors must align (same test windows, same order)")
    ids = np.unique(subjects)
    k = len(ids)
    n_s, ca_s = participant_counts(subjects, correct_a, ids)
    _, cb_s = participant_counts(subjects, correct_b, ids)
    draws = draw_participants(k, replicates, seed)
    mult = np.zeros((replicates, k), dtype=np.int64)
    np.add.at(mult, (np.repeat(np.arange(replicates), k), draws.ravel()), 1)
    denom = mult @ n_s
    acc_a = (mult @ ca_s) / denom
    acc_b = (mult @ cb_s) / denom
    diff_pp = 100.0 * (acc_a - acc_b)
    lo, hi = np.percentile(diff_pp, percentiles)
    observed = 100.0 * (correct_a.mean() - correct_b.mean())
    return {
        "unit": "participant",
        "n_participants": int(k),
        "participant_ids": [int(s) for s in ids],
        "replicates": int(replicates),
        "seed": int(seed),
        "generator": "numpy.random.default_rng (PCG64); integers(0, k, size=(replicates, k))",
        "percentile_method": "numpy.percentile linear",
        "observed_diff_pp": float(observed),
        "ci_pp": [float(lo), float(hi)],
        "percentiles": list(percentiles),
        "bootstrap_mean_pp": float(diff_pp.mean()),
        "bootstrap_sd_pp": float(diff_pp.std(ddof=1)),
        "fraction_le_zero": float(np.mean(diff_pp <= 0)),
    }
