"""Deterministic configuration loading and agreement with the locked protocol (v2 + Addendum A1)."""

import copy

import pytest

from har_repro.config import (config_digest, load_data_sources, load_experiment_config,
                              validate_experiment_config)
from har_repro.models.xgb_params import booster_params, num_boost_round

# Values transcribed from the protocol's frozen configuration table (v2 section 5) and Addendum A1.
LOCKED = {
    "full561": dict(num_boost_round=400, max_depth=3, eta=0.10, colsample_bytree=1.0, subsample=0.8, gamma=0.0,
                    min_child_weight=1.0, base_score=0.5, tree_method="exact", objective="multi:softprob",
                    num_class=6, nthread=1, feature_set="all561"),
    "compact100": dict(num_boost_round=300, max_depth=3, eta=0.10, colsample_bytree=0.8, subsample=0.8, gamma=0.0,
                       min_child_weight=1.0, base_score=0.5, tree_method="exact", objective="multi:softprob",
                       num_class=6, nthread=1, feature_set="compact100"),
}


def test_loading_is_deterministic():
    a, b = load_experiment_config(), load_experiment_config()
    assert a == b
    assert config_digest(a) == config_digest(b)
    assert config_digest(load_data_sources()) == config_digest(load_data_sources())


@pytest.mark.parametrize("name", ["full561", "compact100"])
def test_model_configs_match_locked_protocol(name):
    assert load_experiment_config()["models"][name] == LOCKED[name]


def test_seeds_bootstrap_and_mapping_match_protocol():
    cfg = load_experiment_config()
    assert cfg["seeds"]["main"] == [2026, 2027, 2028, 2029, 2030]
    assert cfg["seeds"]["reference"] == 2026
    assert cfg["seeds"]["sensitivity"] == [2026, 2027, 2028, 2029, 2030]      # A1-R3
    assert cfg["seeds"]["repeatability"] == 2026                              # A1-R3
    assert cfg["bootstrap"] == {"unit": "participant", "model_seed": 2026, "seed": 1729,   # A1-R2
                                "replicates": 10000, "percentiles": [2.5, 97.5]}
    assert cfg["mapping"]["primary_variant"] == "X"
    assert cfg["mapping"]["sensitivity_variants"] == ["Y", "Z"]
    assert cfg["mapping"]["column_order"] == "ascending_original_index"
    assert cfg["decision_rule"] == {                                                       # v2 section 6 + A1-R1
        "order": ["not_reproduced_if_reference_gap_le_0", "strong", "partial", "not_reproduced"],
        "strong_model_tolerance_pp": 1.0, "strong_gap_tolerance_pp": 1.0,
        "partial_full_tolerance_pp": 1.0, "partial_any_tolerance_pp": 2.0}
    assert cfg["data"]["label_encoding"] == "original_minus_one"


def test_primary_xgboost_version_is_the_installed_one():
    import xgboost
    impl = load_experiment_config()["implementation"]
    assert impl["primary_xgboost_python"] == xgboost.__version__ == "3.2.0"
    assert impl["version_parity_sensitivity_xgboost_python"] == "1.7.6"
    assert impl["version_parity_sensitivity_status"] == "optional"


def test_published_targets_are_internally_consistent():
    p = load_experiment_config()["published"]
    assert round(p["full561_correct"] / p["n_test"], 4) == p["full561_accuracy"]
    assert round(p["compact100_correct"] / p["n_test"], 4) == p["compact100_accuracy"]
    assert round(100 * (p["full561_accuracy"] - p["compact100_accuracy"]), 2) == p["gap_pp"]


@pytest.mark.parametrize("name", ["full561", "compact100"])
def test_booster_params_are_explicit_and_seeded(name):
    cfg = load_experiment_config()["models"][name]
    params = booster_params(cfg, seed=2026)
    assert params["seed"] == 2026
    assert params["tree_method"] == "exact" and params["base_score"] == 0.5 and params["nthread"] == 1
    assert "num_boost_round" not in params and "feature_set" not in params
    assert num_boost_round(cfg) == LOCKED[name]["num_boost_round"]


def test_validation_rejects_inconsistent_config():
    bad = copy.deepcopy(load_experiment_config())
    bad["bootstrap"]["model_seed"] = 2027
    with pytest.raises(ValueError):
        validate_experiment_config(bad)
    bad = copy.deepcopy(load_experiment_config())
    del bad["models"]["full561"]["base_score"]
    with pytest.raises(ValueError):
        validate_experiment_config(bad)
