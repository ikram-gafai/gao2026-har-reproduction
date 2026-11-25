"""Evaluation metrics, following the paper's definitions where it states them.

- accuracy and integer number correct
- macro-F1: unweighted mean of per-class F1 (scikit-learn average="macro"). This matches the paper: its
  Table S4 class F1 values average to the reported 0.9384.
- balanced_accuracy_paper: macro mean over classes of one-vs-rest (sensitivity + specificity) / 2. This is the
  paper's definition (Table S4: 0.9625). It is NOT scikit-learn's balanced_accuracy_score (= macro recall), which
  is reported separately and labelled as such.
- macro_auc_ovr: unweighted mean of one-vs-rest ROC-AUC (paper Section 2.6).
- confusion matrix (rows = true, columns = predicted), per-class precision/recall/F1/specificity
- per-participant correct/total counts (needed for the participant-level bootstrap)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray,
                           class_names: Sequence[str]) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    k = len(class_names)
    labels = list(range(k))
    n = len(y_true)
    correct = y_true == y_pred
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    sens, spec, prec, f1s = [], [], [], []
    for c in labels:
        tp = int(cm[c, c])
        fn = int(cm[c, :].sum() - tp)
        fp = int(cm[:, c].sum() - tp)
        tn = int(n - tp - fn - fp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        sens.append(recall), spec.append(specificity), prec.append(precision), f1s.append(f1)
        per_class[class_names[c]] = {"support": tp + fn, "correct": tp, "precision": precision, "recall": recall,
                                     "f1": f1, "specificity": specificity,
                                     "balanced_accuracy_ovr": (recall + specificity) / 2}

    per_participant = {}
    for s in np.unique(subjects):
        m = subjects == s
        per_participant[int(s)] = {"correct": int(correct[m].sum()), "total": int(m.sum())}

    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    return {
        "n": n,
        "n_correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "macro_f1": macro_f1,
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(sens)),
        "balanced_accuracy_paper": float(np.mean([(a + b) / 2 for a, b in zip(sens, spec)])),
        "sklearn_balanced_accuracy_macro_recall": float(np.mean(sens)),
        "macro_auc_ovr": float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)),
        "confusion_matrix": {"labels": list(class_names), "rows_true_cols_pred": cm.tolist()},
        "per_class": per_class,
        "per_participant": per_participant,
    }
