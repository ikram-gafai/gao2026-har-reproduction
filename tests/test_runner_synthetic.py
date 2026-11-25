"""End-to-end experiment machinery on a synthetic 561-column dataset (never the real UCI HAR test set)."""

import copy
import csv
import json

import numpy as np
import pytest

import har_repro.cli as cli
from har_repro.config import load_experiment_config
from har_repro.data.loader import load_uci_har
from har_repro.experiments.runner import RunSpec, build_plan, execute_run, feature_columns

from synthetic561 import write_synthetic_561

MAPPING = json.loads(cli.MAPPING_FILE.read_text(encoding="ascii"))


@pytest.fixture(scope="module")
def syn_root(tmp_path_factory):
    return write_synthetic_561(tmp_path_factory.mktemp("syn") / "UCI HAR Dataset")


@pytest.fixture
def small_cfg(monkeypatch):
    """Frozen config with fewer seeds (model parameters untouched) to keep tests fast."""
    cfg = copy.deepcopy(load_experiment_config())
    cfg["seeds"]["main"] = [2026, 2027]
    cfg["seeds"]["sensitivity"] = [2026]
    monkeypatch.setattr(cli, "load_experiment_config", lambda *a, **k: cfg)
    return cfg


def test_plan_matches_protocol():
    plan = build_plan(load_experiment_config())
    ids = [s.run_id for s in plan]
    assert len(plan) == 20
    assert ids[:2] == ["full561__seed2026", "compact100-X__seed2026"]
    assert sum(s.group == "main" for s in plan) == 10 and sum(s.group == "sensitivity" for s in plan) == 10
    assert {s.variant for s in plan if s.group == "sensitivity"} == {"Y", "Z"}


def test_feature_columns():
    assert feature_columns(RunSpec("full561", None, 1, "main"), MAPPING, 561) == list(range(561))
    cols = feature_columns(RunSpec("compact100", "X", 1, "main"), MAPPING, 561)
    assert cols == MAPPING["variants"]["X"]["indices_0based"] and cols == sorted(cols) and len(cols) == 100


def test_training_never_sees_test_data(syn_root, tmp_path):
    """Changing test labels and test features must not change the fitted model."""
    cfg = load_experiment_config()
    data = load_uci_har(syn_root)
    spec = RunSpec("compact100", "X", 2026, "main")
    r1 = execute_run(data, spec, cfg, MAPPING, tmp_path / "a", False)
    rng = np.random.default_rng(1)
    altered = copy.copy(data)
    object.__setattr__(altered, "y_test", rng.permutation(data.y_test))
    X_te = data.X_test.copy()
    X_te[:5] = 0.0
    object.__setattr__(altered, "X_test", X_te)
    r2 = execute_run(altered, spec, cfg, MAPPING, tmp_path / "b", False)
    assert (tmp_path / "a" / "model.json").read_bytes() == (tmp_path / "b" / "model.json").read_bytes()
    np.testing.assert_array_equal(r1["proba"][5:], r2["proba"][5:])


def test_run_outputs_are_consistent(syn_root, tmp_path):
    cfg = load_experiment_config()
    data = load_uci_har(syn_root)
    res = execute_run(data, RunSpec("full561", None, 2026, "main"), cfg, MAPPING, tmp_path, False)
    with open(tmp_path / "predictions.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(data.y_test)
    assert sum(r["correct"] == "1" for r in rows) == res["metrics"]["n_correct"]
    proba = np.load(tmp_path / "proba.npy")
    assert proba.dtype == np.float32 and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    info = json.loads((tmp_path / "run_info.json").read_text())
    assert info["data_verified"] is False and info["n_columns"] == 561
    assert info["explicit_params"]["tree_method"] == "exact" and info["num_boost_round"] == 400
    lmp = info["effective_booster_config"]["learner"]["learner_model_param"]
    assert [float(v) for v in lmp["base_score"].strip("[]").split(",")] == [0.5] * 6


def test_full_pipeline_cli_with_repeatability(syn_root, tmp_path, small_cfg):
    out = tmp_path / "results"
    rc = cli.main(["run-experiments", "--data-root", str(syn_root), "--allow-unverified-data",
                   "--results-dir", str(out), "--allow-dirty"])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["plan"] == [s.run_id for s in build_plan(small_cfg)] and manifest["data_verified"] is False
    assert len(list((out / "runs").iterdir())) == 6        # 2 seeds x 2 models + Y + Z
    summary = json.loads((out / "summary" / "summary.json").read_text())
    assert [r["seed"] for r in summary["per_seed"]] == [2026, 2027]
    boot = json.loads((out / "summary" / "bootstrap.json").read_text())
    assert boot["replicates"] == 10000 and boot["seed"] == 1729 and boot["n_participants"] == 3
    decision = json.loads((out / "summary" / "decision.json").read_text())
    assert decision["category"] in {"strong", "partial", "not_reproduced"}
    assert (out / "summary" / "results_table.md").read_text().startswith("# Results (machine-generated")
    rep = json.loads((out / "repeatability" / "repeatability.json").read_text())
    assert rep["passed"] is True and len(rep["checks"]) == 2 and all(c["fresh_process"] for c in rep["checks"])
    # Refuses to overwrite existing runs
    with pytest.raises(SystemExit):
        cli.main(["run-experiments", "--data-root", str(syn_root), "--allow-unverified-data",
                  "--results-dir", str(out), "--allow-dirty"])


def test_refuses_unverified_data_without_flag(syn_root, tmp_path, small_cfg):
    with pytest.raises(SystemExit):
        cli.main(["run-experiments", "--data-root", str(syn_root), "--results-dir", str(tmp_path), "--allow-dirty"])


def test_refuses_uncommitted_source_changes(syn_root, tmp_path, small_cfg, monkeypatch):
    import har_repro.experiments.runner as runner
    monkeypatch.setattr(runner, "git_state", lambda *a, **k: {"commit": "abc", "source_changes_uncommitted": " M src/x.py"})
    with pytest.raises(SystemExit):
        cli.main(["run-experiments", "--data-root", str(syn_root), "--allow-unverified-data",
                  "--results-dir", str(tmp_path)])


def test_verify_results_passes_and_detects_tampering(syn_root, tmp_path, small_cfg):
    from har_repro.experiments.verify import verify_results
    out = tmp_path / "results"
    assert cli.main(["run-experiments", "--data-root", str(syn_root), "--allow-unverified-data",
                     "--results-dir", str(out), "--allow-dirty"]) == 0
    data = load_uci_har(syn_root)
    ok = verify_results(out, small_cfg, data, MAPPING, refit=["compact100-X__seed2026"], log=lambda *_: None)
    assert ok["passed"], ok["problems"]
    assert ok["refits"][0]["bit_identical"] is True
    assert cli.main(["verify-results", "--results-dir", str(out), "--data-root", str(syn_root),
                     "--allow-unverified-data"]) == 0

    # Tampering with a metric, a prediction or the summary must be detected.
    m = out / "runs" / "full561__seed2026" / "metrics.json"
    original = m.read_text()
    m.write_text(original.replace('"n_correct": ', '"n_correct": 1', 1))
    assert not verify_results(out, small_cfg, data, MAPPING, log=lambda *_: None)["passed"]
    m.write_text(original)
    p = out / "runs" / "compact100-X__seed2027" / "predictions.csv"
    lines = p.read_text().splitlines()
    fields = lines[1].split(",")
    fields[6] = "0" if fields[6] == "1" else "1"
    p.write_text("\n".join([lines[0], ",".join(fields)] + lines[2:]) + "\n")
    assert not verify_results(out, small_cfg, data, MAPPING, log=lambda *_: None)["passed"]
    p.write_text("\n".join(lines) + "\n")
    s = out / "summary" / "decision.json"
    s.write_text(s.read_text().replace('"basis"', '"basis_edited"'))
    res = verify_results(out, small_cfg, data, MAPPING, log=lambda *_: None)
    assert not res["passed"] and any("decision.json" in x for x in res["problems"])
