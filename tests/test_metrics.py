"""Metric definitions on hand-checkable synthetic predictions."""

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from har_repro.evaluation.metrics import classification_metrics

NAMES = ["A", "B", "C", "D", "E", "F"]


@pytest.fixture
def example():
    y_true = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5])
    y_pred = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 2, 5, 5, 5])
    subjects = np.array([2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4])
    proba = np.full((len(y_true), 6), 0.02)
    proba[np.arange(len(y_true)), y_pred] = 0.9
    return y_true, y_pred, proba, subjects


def test_counts_accuracy_and_confusion(example):
    y_true, y_pred, proba, subjects = example
    m = classification_metrics(y_true, proba, y_pred, subjects, NAMES)
    assert m["n"] == 14 and m["n_correct"] == 11 and m["accuracy"] == pytest.approx(11 / 14)
    cm = np.array(m["confusion_matrix"]["rows_true_cols_pred"])
    assert cm.sum() == 14 and np.trace(cm) == 11 and cm[0, 1] == 1 and cm[2, 3] == 1 and cm[4, 2] == 1


def test_macro_f1_matches_sklearn(example):
    y_true, y_pred, proba, subjects = example
    m = classification_metrics(y_true, proba, y_pred, subjects, NAMES)
    assert m["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert m["macro_f1"] == pytest.approx(np.mean([c["f1"] for c in m["per_class"].values()]))


def test_paper_balanced_accuracy_differs_from_sklearn(example):
    y_true, y_pred, proba, subjects = example
    m = classification_metrics(y_true, proba, y_pred, subjects, NAMES)
    manual = []
    n = len(y_true)
    for c in range(6):
        tp = np.sum((y_true == c) & (y_pred == c)); fn = np.sum((y_true == c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c)); tn = n - tp - fn - fp
        manual.append((tp / (tp + fn) + tn / (tn + fp)) / 2)
    assert m["balanced_accuracy_paper"] == pytest.approx(np.mean(manual))
    assert m["sklearn_balanced_accuracy_macro_recall"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
    assert m["balanced_accuracy_paper"] != pytest.approx(m["sklearn_balanced_accuracy_macro_recall"])


def test_per_participant_counts(example):
    y_true, y_pred, proba, subjects = example
    m = classification_metrics(y_true, proba, y_pred, subjects, NAMES)
    assert m["per_participant"] == {2: {"correct": 5, "total": 7}, 4: {"correct": 6, "total": 7}}


def test_macro_auc_is_one_for_perfect_ranking():
    y = np.repeat(np.arange(6), 3)
    proba = np.eye(6)[y] * 0.88 + 0.02          # rows sum to exactly 1, as softprob outputs do
    m = classification_metrics(y, proba, y, np.ones_like(y), NAMES)
    assert m["macro_auc_ovr"] == pytest.approx(1.0) and m["accuracy"] == 1.0
