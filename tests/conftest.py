from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from har_repro.config import load_data_sources
from har_repro.data.acquire import IntegrityError, verify_extracted
from har_repro.data.loader import load_uci_har
from har_repro.paths import dataset_root


@pytest.fixture(scope="session")
def verified_root() -> Path:
    """The real, checksum-verified dataset; tests marked `data` skip if it has not been prepared."""
    root = dataset_root()
    if not (root / "features.txt").exists():
        pytest.skip("UCI HAR dataset not prepared; run `python -m har_repro.cli prepare-data`")
    try:
        verify_extracted(root, load_data_sources()["expected"]["files"])
    except IntegrityError as exc:
        pytest.fail(f"prepared dataset failed checksum verification: {exc}")
    return root


@pytest.fixture(scope="session")
def uci(verified_root):
    return load_uci_har(verified_root)


# ---------------------------------------------------------------- synthetic UCI-HAR-shaped dataset
SYN_FEATURES = [
    "aAcc-mean()-X", "aAcc-mean()-Y", "aAcc-mean()-Z",
    "fB-bandsEnergy()-1,8", "fB-bandsEnergy()-9,16",      # block X
    "fB-bandsEnergy()-1,8", "fB-bandsEnergy()-9,16",      # block Y
    "fB-bandsEnergy()-1,8", "fB-bandsEnergy()-9,16",      # block Z
    "mag-mean()", "mag-sma()",                              # identical columns in the synthetic data
]
SYN_LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


def write_synthetic_dataset(root: Path, n_train: int = 24, n_test: int = 12, seed: int = 0,
                            train_subjects=(1, 3, 5), test_subjects=(2, 4)) -> dict:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    (root / "features.txt").write_text("".join(f"{i} {n}\n" for i, n in enumerate(SYN_FEATURES, 1)), encoding="ascii")
    (root / "activity_labels.txt").write_text("".join(f"{i} {n}\n" for i, n in enumerate(SYN_LABELS, 1)), encoding="ascii")
    out = {}
    for part, n, subjects in (("train", n_train, train_subjects), ("test", n_test, test_subjects)):
        d = root / part
        d.mkdir(exist_ok=True)
        X = np.round(rng.uniform(-1, 1, size=(n, len(SYN_FEATURES))), 8)
        X[:, 10] = X[:, 9]  # mag-sma() == mag-mean()
        y = np.tile(np.arange(1, 7), n // 6 + 1)[:n]
        s = np.resize(np.asarray(subjects), n)
        # 17 significant digits round-trip exactly, so loaded values must equal the generated doubles bit-for-bit.
        (d / f"X_{part}.txt").write_text("\n".join("  " + " ".join(f"{v: .16e}" for v in row) for row in X) + "\n")
        (d / f"y_{part}.txt").write_text("\n".join(str(v) for v in y) + "\n")
        (d / f"subject_{part}.txt").write_text("\n".join(str(v) for v in s) + "\n")
        out[part] = (X, y, s)
    return out


@pytest.fixture
def synthetic_root(tmp_path) -> tuple[Path, dict]:
    root = tmp_path / "UCI HAR Dataset"
    arrays = write_synthetic_dataset(root)
    return root, arrays
