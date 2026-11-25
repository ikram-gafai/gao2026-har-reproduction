"""Fit one frozen XGBoost configuration on the training partition and predict class probabilities."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import xgboost as xgb

from .xgb_params import booster_params, num_boost_round


@dataclass
class FittedModel:
    booster: xgb.Booster
    params: dict[str, Any]
    num_boost_round: int
    fit_seconds: float

    def saved_config(self) -> dict[str, Any]:
        """Full effective configuration as stored by xgboost, including library defaults."""
        return json.loads(self.booster.save_config())


def fit(X_train: np.ndarray, y_train_enc: np.ndarray, model_cfg: Mapping[str, Any], seed: int) -> FittedModel:
    """Train on the training partition only. No evaluation set, no early stopping, no test data."""
    params = booster_params(model_cfg, seed)
    rounds = num_boost_round(model_cfg)
    dtrain = xgb.DMatrix(np.ascontiguousarray(X_train), label=np.asarray(y_train_enc))
    t0 = time.perf_counter()
    booster = xgb.train(params, dtrain, num_boost_round=rounds, evals=[], verbose_eval=False)
    return FittedModel(booster, params, rounds, time.perf_counter() - t0)


def predict_proba(model: FittedModel, X: np.ndarray) -> np.ndarray:
    """Class probabilities (multi:softprob output), shape (n, num_class), float32 as returned by xgboost."""
    proba = model.booster.predict(xgb.DMatrix(np.ascontiguousarray(X)))
    if proba.ndim != 2 or proba.shape[1] != model.params["num_class"]:
        raise ValueError(f"unexpected probability shape {proba.shape}")
    return proba


def predict_labels(proba: np.ndarray) -> np.ndarray:
    """Encoded class with the highest probability (ties -> lowest class index, as numpy argmax)."""
    return np.argmax(proba, axis=1).astype(np.int64)
