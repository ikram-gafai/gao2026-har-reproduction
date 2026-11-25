"""Paired participant-level bootstrap on synthetic correctness vectors."""

import numpy as np
import pytest

from har_repro.evaluation.bootstrap import draw_participants, paired_participant_bootstrap


@pytest.fixture
def toy():
    rng = np.random.default_rng(3)
    subjects = np.repeat([2, 4, 9, 10, 12, 13, 18, 20, 24], rng.integers(20, 40, size=9))
    a = rng.random(len(subjects)) < 0.95
    b = rng.random(len(subjects)) < 0.92
    return subjects, a, b


def test_vectorised_pooling_equals_explicit_concatenation(toy):
    subjects, a, b = toy
    res = paired_participant_bootstrap(subjects, a, b, replicates=200, seed=1729)
    ids = np.unique(subjects)
    draws = draw_participants(len(ids), 200, 1729)
    explicit = []
    for row in draws:
        idx = np.concatenate([np.flatnonzero(subjects == ids[j]) for j in row])   # repeated draws kept at multiplicity
        explicit.append(100 * (a[idx].mean() - b[idx].mean()))
    lo, hi = np.percentile(explicit, [2.5, 97.5])
    assert res["ci_pp"] == pytest.approx([lo, hi], abs=1e-12)
    assert res["bootstrap_mean_pp"] == pytest.approx(np.mean(explicit), abs=1e-12)


def test_deterministic_given_seed_and_sensitive_to_seed(toy):
    subjects, a, b = toy
    r1 = paired_participant_bootstrap(subjects, a, b, 1000, 1729)
    r2 = paired_participant_bootstrap(subjects, a, b, 1000, 1729)
    r3 = paired_participant_bootstrap(subjects, a, b, 1000, 1730)
    assert r1 == r2 and r1["ci_pp"] != r3["ci_pp"]
    assert r1["n_participants"] == 9 and r1["seed"] == 1729


def test_identical_models_give_zero_interval(toy):
    subjects, a, _ = toy
    r = paired_participant_bootstrap(subjects, a, a, 500, 1729)
    assert r["ci_pp"] == [0.0, 0.0] and r["observed_diff_pp"] == 0.0


def test_observed_difference_is_pooled_window_accuracy(toy):
    subjects, a, b = toy
    r = paired_participant_bootstrap(subjects, a, b, 10, 1729)
    assert r["observed_diff_pp"] == pytest.approx(100 * (a.mean() - b.mean()))


def test_misaligned_inputs_rejected(toy):
    subjects, a, b = toy
    with pytest.raises(ValueError):
        paired_participant_bootstrap(subjects[:-1], a, b, 10, 1729)
