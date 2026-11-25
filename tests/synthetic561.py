"""Synthetic dataset in the official UCI HAR file layout with 561 columns (for Phase 2 implementation tests).

Feature names at the mapped compact indices are the real S2 names (taken from the committed mapping manifest),
so the runner's name cross-check is exercised. Values are random with a class-dependent signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from har_repro.cli import MAPPING_FILE

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


def write_synthetic_561(root: Path, train_subjects=(1, 3, 5, 7, 9, 11), test_subjects=(2, 4, 6),
                        per_subject_class: int = 4, seed: int = 0) -> Path:
    mapping = json.loads(MAPPING_FILE.read_text(encoding="ascii"))
    names = [f"synthetic_feature_{i}" for i in range(1, 562)]
    for v in ("X", "Y", "Z"):
        for idx, name in zip(mapping["variants"][v]["indices_1based"], mapping["variants"][v]["names"]):
            names[idx - 1] = name
    rng = np.random.default_rng(seed)
    informative = np.array(mapping["variants"]["X"]["indices_0based"][:30] + list(range(0, 561, 37)))
    root.mkdir(parents=True, exist_ok=True)
    (root / "features.txt").write_text("".join(f"{i} {n}\n" for i, n in enumerate(names, 1)), encoding="ascii")
    (root / "activity_labels.txt").write_text("".join(f"{i} {n}\n" for i, n in enumerate(LABELS, 1)), encoding="ascii")
    centers = rng.normal(0, 0.6, size=(6, informative.size))
    for part, subjects in (("train", train_subjects), ("test", test_subjects)):
        rows, ys, ss = [], [], []
        for s in subjects:
            for c in range(6):
                for _ in range(per_subject_class):
                    x = rng.normal(0, 0.5, size=561)
                    x[informative] += centers[c]
                    rows.append(np.clip(x, -1, 1)), ys.append(c + 1), ss.append(s)
        d = root / part
        d.mkdir(exist_ok=True)
        (d / f"X_{part}.txt").write_text("\n".join(" ".join(f"{v: .16e}" for v in r) for r in rows) + "\n")
        (d / f"y_{part}.txt").write_text("\n".join(map(str, ys)) + "\n")
        (d / f"subject_{part}.txt").write_text("\n".join(map(str, ss)) + "\n")
    return root
