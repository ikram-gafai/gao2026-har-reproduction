"""Load the official UCI HAR 561-feature representation without altering its split, order or values."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureTable:
    indices: tuple[int, ...]      # original 1-based indices from features.txt
    names: tuple[str, ...]        # original names (not de-duplicated)

    def __len__(self) -> int:
        return len(self.names)


@dataclass(frozen=True)
class UCIHAR:
    features: FeatureTable
    activity_names: dict[int, str]          # original label (1..6) -> name
    X_train: np.ndarray                     # float64, shape (n_train, 561), original column order
    X_test: np.ndarray
    y_train: np.ndarray                     # original labels 1..6 (int64)
    y_test: np.ndarray
    subject_train: np.ndarray               # participant ids (int64)
    subject_test: np.ndarray
    source_root: Path = field(compare=False)


def read_features(path: Path) -> FeatureTable:
    idx, names = [], []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        i, name = line.split(" ", 1)
        idx.append(int(i))
        names.append(name)
    if idx != list(range(1, len(idx) + 1)):
        raise ValueError("features.txt indices are not contiguous from 1")
    return FeatureTable(tuple(idx), tuple(names))


def read_activity_labels(path: Path) -> dict[int, str]:
    out = {}
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if line.strip():
            k, v = line.split()
            out[int(k)] = v
    return out


def read_matrix(path: Path, n_cols: int) -> np.ndarray:
    """Whitespace-delimited numeric text -> float64 array.

    pandas' default C parser is used; tests/test_data_integrity.py asserts bitwise equality with Python's
    exact float() parsing for both official partitions, so any parser change would be detected."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=np.float64)
    if df.shape[1] != n_cols:
        raise ValueError(f"{path.name}: expected {n_cols} columns, found {df.shape[1]}")
    return df.to_numpy(dtype=np.float64, copy=True)


def read_vector(path: Path) -> np.ndarray:
    vals = [int(tok) for tok in Path(path).read_text(encoding="ascii").split()]
    return np.asarray(vals, dtype=np.int64)


def load_uci_har(root: Path) -> UCIHAR:
    """Load the official partition exactly as distributed (no shuffling, no scaling, no merging)."""
    root = Path(root)
    feats = read_features(root / "features.txt")
    labels = read_activity_labels(root / "activity_labels.txt")
    data = {}
    for part in ("train", "test"):
        X = read_matrix(root / part / f"X_{part}.txt", len(feats))
        y = read_vector(root / part / f"y_{part}.txt")
        s = read_vector(root / part / f"subject_{part}.txt")
        if not (len(X) == len(y) == len(s)):
            raise ValueError(f"{part}: row counts differ (X={len(X)}, y={len(y)}, subject={len(s)})")
        data[part] = (X, y, s)
    return UCIHAR(
        features=feats, activity_names=labels,
        X_train=data["train"][0], X_test=data["test"][0],
        y_train=data["train"][1], y_test=data["test"][1],
        subject_train=data["train"][2], subject_test=data["test"][2],
        source_root=root,
    )


def encode_labels(y: np.ndarray, activity_names: dict[int, str]) -> np.ndarray:
    """Protocol label map: original 1..6 -> 0..5 (activity names preserved via activity_names)."""
    allowed = set(activity_names)
    if sorted(allowed) != list(range(1, len(allowed) + 1)):
        raise ValueError("activity labels must be 1..K")
    present = set(np.unique(y).tolist())
    if not present <= allowed:
        raise ValueError(f"unexpected labels {sorted(present - allowed)}")
    return (np.asarray(y, dtype=np.int64) - 1)


def decode_labels(y_enc: np.ndarray, activity_names: dict[int, str]) -> list[str]:
    return [activity_names[int(v) + 1] for v in y_enc]


def identical_column_groups(X: np.ndarray) -> list[list[int]]:
    """Groups (0-based column positions) of columns that are exactly equal across all rows."""
    groups: dict[bytes, list[int]] = {}
    Xc = np.ascontiguousarray(X.T)
    for j in range(Xc.shape[0]):
        groups.setdefault(Xc[j].tobytes(), []).append(j)
    return [g for g in groups.values() if len(g) > 1]
