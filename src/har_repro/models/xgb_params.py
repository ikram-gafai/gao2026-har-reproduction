"""Translate a frozen model configuration into native `xgboost.train` parameters (no fitting here)."""

from __future__ import annotations

from typing import Any, Mapping

# Booster parameters passed to xgboost.train. num_boost_round and feature_set are handled separately.
BOOSTER_KEYS = (
    "objective", "num_class", "tree_method", "base_score", "nthread",
    "max_depth", "eta", "subsample", "colsample_bytree", "gamma", "min_child_weight",
)


def booster_params(model_cfg: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Return the explicit parameter dict for one model and seed. Unlisted parameters keep library defaults,
    which are recorded from the trained booster's saved configuration at run time."""
    params = {k: model_cfg[k] for k in BOOSTER_KEYS}
    params["seed"] = int(seed)
    return params


def num_boost_round(model_cfg: Mapping[str, Any]) -> int:
    return int(model_cfg["num_boost_round"])
