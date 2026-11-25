"""Deterministic loading of the frozen TOML configuration files."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any, Mapping

from .paths import CONFIG_DIR

EXPERIMENT_FILE = CONFIG_DIR / "experiment.toml"
DATA_SOURCES_FILE = CONFIG_DIR / "data_sources.toml"

_MODEL_KEYS = (
    "feature_set", "num_boost_round", "max_depth", "eta", "colsample_bytree", "subsample", "gamma",
    "min_child_weight", "base_score", "tree_method", "objective", "num_class", "nthread",
)


def load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def canonical_json(obj: Any) -> str:
    """Key-sorted, whitespace-free JSON, used for stable hashing and manifests."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_digest(obj: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(obj).encode("ascii")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_experiment_config(path: Path = EXPERIMENT_FILE) -> dict[str, Any]:
    cfg = load_toml(path)
    validate_experiment_config(cfg)
    return cfg


def load_data_sources(path: Path = DATA_SOURCES_FILE) -> dict[str, Any]:
    cfg = load_toml(path)
    if not cfg.get("sources"):
        raise ValueError("data_sources.toml defines no sources")
    for src in cfg["sources"]:
        if src.get("kind") not in {"inner", "outer"}:
            raise ValueError(f"source {src.get('name')!r} has invalid kind {src.get('kind')!r}")
    return cfg


def validate_experiment_config(cfg: Mapping[str, Any]) -> None:
    """Structural validation only. Values are checked against the protocol in tests/test_config.py."""
    for name in ("full561", "compact100"):
        model = cfg["models"][name]
        missing = [k for k in _MODEL_KEYS if k not in model]
        extra = [k for k in model if k not in _MODEL_KEYS]
        if missing or extra:
            raise ValueError(f"model {name}: missing={missing} unexpected={extra}")
    if cfg["seeds"]["reference"] not in cfg["seeds"]["main"]:
        raise ValueError("reference seed must be one of the main seeds")
    if cfg["bootstrap"]["model_seed"] != cfg["seeds"]["reference"]:
        raise ValueError("bootstrap must use the reference seed model pair")
    if cfg["mapping"]["primary_variant"] in cfg["mapping"]["sensitivity_variants"]:
        raise ValueError("primary mapping variant must not also be a sensitivity variant")
