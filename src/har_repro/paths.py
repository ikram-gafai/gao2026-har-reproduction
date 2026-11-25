"""Project paths. Override the data location with the HAR_DATA_DIR environment variable."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"


def data_dir() -> Path:
    """Root data directory (default: <project>/data). Contains raw/ and extracted/."""
    return Path(os.environ.get("HAR_DATA_DIR", PROJECT_ROOT / "data")).resolve()


def raw_dir() -> Path:
    return data_dir() / "raw"


def extracted_dir() -> Path:
    return data_dir() / "extracted"


def dataset_root() -> Path:
    """Directory holding the extracted 'UCI HAR Dataset' files."""
    return extracted_dir() / "UCI HAR Dataset"
