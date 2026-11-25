"""The frozen XGBoost settings are accepted and honoured by the installed xgboost (synthetic data only).

These tests fit tiny models on random synthetic data to confirm parameter handling (exact updater, scalar
base_score, single thread) and same-seed determinism. They do not touch UCI HAR and are not the experiments.
"""

import json

import numpy as np
import pytest
import xgboost as xgb

from har_repro.config import load_experiment_config
from har_repro.models.xgb_params import booster_params


def _synthetic(seed=0, n=180, d=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.repeat(np.arange(6), n // 6)
    X[:, 0] += y   # some signal
    return X, y


@pytest.mark.parametrize("name", ["full561", "compact100"])
def test_saved_config_reflects_frozen_settings(name):
    cfg = load_experiment_config()["models"][name]
    X, y = _synthetic()
    booster = xgb.train(booster_params(cfg, 2026), xgb.DMatrix(X, label=y), num_boost_round=5)
    saved = json.loads(booster.save_config())["learner"]
    base = saved["learner_model_param"]["base_score"].strip("[]").split(",")
    assert [float(v) for v in base] == [0.5] * 6                    # uniform scalar intercept, not estimated
    gb = saved["gradient_booster"]["gbtree_train_param"]
    assert gb["tree_method"] == "exact" and gb["updater"] == "grow_colmaker,prune"
    assert saved["generic_param"]["nthread"] == "1" and saved["generic_param"]["seed"] == "2026"
    assert saved["objective"]["name"] == "multi:softprob"


def test_same_seed_gives_identical_probabilities_and_different_seed_differs():
    cfg = load_experiment_config()["models"]["compact100"]   # subsample and colsample both 0.8
    X, y = _synthetic()
    dtrain, dpred = xgb.DMatrix(X, label=y), xgb.DMatrix(X)
    run = lambda s: xgb.train(booster_params(cfg, s), dtrain, num_boost_round=10).predict(dpred)
    a, b, c = run(2026), run(2026), run(2027)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
