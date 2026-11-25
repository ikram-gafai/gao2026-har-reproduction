"""Experiment orchestration for the pre-registered protocol.

Each run fits on the official training partition only, then predicts the test partition once.
Outputs are machine-written and never edited by hand:

    <results>/manifest.json                      frozen state recorded before any test prediction
    <results>/runs/<run_id>/predictions.csv      row, subject, true/pred labels, correctness, probabilities
    <results>/runs/<run_id>/proba.npy            exact float32 probabilities (bitwise repeatability checks)
    <results>/runs/<run_id>/metrics.json
    <results>/runs/<run_id>/run_info.json        seed, columns, timing, effective booster configuration
    <results>/runs/<run_id>/model.json           trained booster (git-ignored)
    <results>/summary/summary.json, bootstrap.json, decision.json, results_table.md
    <results>/repeatability/repeatability.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .. import __version__
from ..config import file_sha256
from ..data.loader import UCIHAR, encode_labels
from ..evaluation.bootstrap import paired_participant_bootstrap
from ..evaluation.decision import classify_outcome
from ..evaluation.metrics import classification_metrics
from ..models.train import fit, predict_labels, predict_proba
from ..paths import PROJECT_ROOT


@dataclass(frozen=True)
class RunSpec:
    model: str            # "full561" | "compact100"
    variant: str | None   # None for full561; "X" | "Y" | "Z" for compact100
    seed: int
    group: str            # "main" | "sensitivity"

    @property
    def run_id(self) -> str:
        tag = self.model if self.variant is None else f"{self.model}-{self.variant}"
        return f"{tag}__seed{self.seed}"


def build_plan(cfg: Mapping[str, Any], include_sensitivity: bool = True) -> list[RunSpec]:
    primary = cfg["mapping"]["primary_variant"]
    plan = []
    for seed in cfg["seeds"]["main"]:
        plan.append(RunSpec("full561", None, seed, "main"))
        plan.append(RunSpec("compact100", primary, seed, "main"))
    if include_sensitivity:
        for variant in cfg["mapping"]["sensitivity_variants"]:
            for seed in cfg["seeds"]["sensitivity"]:
                plan.append(RunSpec("compact100", variant, seed, "sensitivity"))
    return plan


def feature_columns(spec: RunSpec, mapping_doc: Mapping[str, Any], n_features: int) -> list[int]:
    """0-based column positions: all columns for full561; ascending mapped indices for compact100."""
    if spec.model == "full561":
        return list(range(n_features))
    cols = list(mapping_doc["variants"][spec.variant]["indices_0based"])
    if cols != sorted(cols) or len(set(cols)) != len(cols):
        raise ValueError("compact columns must be unique and in ascending original-index order")
    return cols


def check_feature_names(data: UCIHAR, spec: RunSpec, cols: Sequence[int], mapping_doc: Mapping[str, Any]) -> None:
    if spec.model == "full561":
        return
    expected = mapping_doc["variants"][spec.variant]["names"]
    got = [data.features.names[c] for c in cols]
    if got != expected:
        raise ValueError(f"feature names at mapped columns do not match the mapping manifest ({spec.run_id})")


def _sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def execute_run(data: UCIHAR, spec: RunSpec, cfg: Mapping[str, Any], mapping_doc: Mapping[str, Any],
                out_dir: Path, data_verified: bool, save_model: bool = True) -> dict:
    """Train on the training partition, predict the test partition once, and write all outputs."""
    model_cfg = cfg["models"][spec.model]
    cols = feature_columns(spec, mapping_doc, data.X_train.shape[1])
    check_feature_names(data, spec, cols, mapping_doc)
    class_names = [data.activity_names[i] for i in sorted(data.activity_names)]
    y_train = encode_labels(data.y_train, data.activity_names)
    y_test = encode_labels(data.y_test, data.activity_names)

    fitted = fit(data.X_train[:, cols], y_train, model_cfg, spec.seed)          # training rows only
    t0 = time.perf_counter()
    proba = predict_proba(fitted, data.X_test[:, cols])                          # single test prediction
    predict_seconds = time.perf_counter() - t0
    y_pred = predict_labels(proba)
    metrics = classification_metrics(y_test, proba, y_pred, data.subject_test, class_names)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "proba.npy", proba)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["test_row", "subject", "y_true", "y_true_name", "y_pred", "y_pred_name", "correct"]
                   + [f"p_{c}" for c in class_names])
        for i in range(len(y_test)):
            w.writerow([i, int(data.subject_test[i]), int(y_test[i]), class_names[y_test[i]], int(y_pred[i]),
                        class_names[y_pred[i]], int(y_test[i] == y_pred[i])] + [repr(float(p)) for p in proba[i]])
    if save_model:
        fitted.booster.save_model(str(out_dir / "model.json"))
    run_info = {
        "run_id": spec.run_id, **asdict(spec),
        "data_verified": data_verified,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_columns": len(cols), "columns_0based": cols,
        "columns_sha256": hashlib.sha256(json.dumps(cols).encode()).hexdigest(),
        "num_boost_round": fitted.num_boost_round, "explicit_params": fitted.params,
        "effective_booster_config": fitted.saved_config(),
        "fit_seconds": round(fitted.fit_seconds, 3), "predict_seconds": round(predict_seconds, 3),
        "proba_sha256": _sha256_array(proba), "y_pred_sha256": _sha256_array(y_pred),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=1) + "\n", encoding="utf-8")
    return {"metrics": metrics, "run_info": run_info, "y_pred": y_pred, "proba": proba}


# ----------------------------------------------------------------------------------------------- manifest
def git_state(root: Path = PROJECT_ROOT) -> dict:
    def git(*args):
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    tracked_changes = git("status", "--porcelain", "--untracked-files=no", "--", "src", "configs", "tests",
                          "pyproject.toml", "requirements.txt", "requirements-lock.txt")
    return {"commit": git("rev-parse", "HEAD"), "source_changes_uncommitted": tracked_changes}


def environment_info() -> dict:
    import pandas, scipy, sklearn, xgboost
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "har_repro": __version__, "numpy": np.__version__,
            "pandas": pandas.__version__, "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
            "xgboost": xgboost.__version__}


def write_manifest(results_dir: Path, cfg: Mapping[str, Any], plan: Sequence[RunSpec], data: UCIHAR,
                   data_verified: bool, config_files: Mapping[str, Path], git: Mapping[str, Any]) -> dict:
    manifest = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Frozen state recorded before any test-set prediction of this run batch.",
        "git": dict(git), "environment": environment_info(), "data_verified": data_verified,
        "data_root": "<HAR_DATA_DIR or project data/>" if data_verified else str(data.source_root),
        "config_sha256": {k: file_sha256(p) for k, p in config_files.items()},
        "plan": [s.run_id for s in plan], "seeds": cfg["seeds"], "bootstrap": cfg["bootstrap"],
        "decision_rule": cfg["decision_rule"], "published": cfg["published"],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


# ----------------------------------------------------------------------------------------------- summary
def _load_run(results_dir: Path, run_id: str) -> tuple[dict, dict, np.ndarray]:
    d = results_dir / "runs" / run_id
    metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
    correct = []
    with open(d / "predictions.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    subjects = np.array([int(r["subject"]) for r in rows])
    correct = np.array([r["correct"] == "1" for r in rows])
    return metrics, info, np.vstack([subjects, correct])


def _stats(values: Sequence[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"values": [float(v) for v in a], "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None,
            "min": float(a.min()), "max": float(a.max())}


def summarize(results_dir: Path, cfg: Mapping[str, Any], include_sensitivity: bool = True) -> dict:
    pub = cfg["published"]
    primary = cfg["mapping"]["primary_variant"]
    ref = cfg["seeds"]["reference"]
    main_seeds = cfg["seeds"]["main"]

    per_seed = []
    for seed in main_seeds:
        mf, _, pf = _load_run(results_dir, RunSpec("full561", None, seed, "main").run_id)
        mc, _, pc = _load_run(results_dir, RunSpec("compact100", primary, seed, "main").run_id)
        if not np.array_equal(pf[0], pc[0]):
            raise ValueError("full and compact predictions are not aligned on the same test windows")
        per_seed.append({"seed": seed, "full561_correct": mf["n_correct"], "full561_accuracy": mf["accuracy"],
                         "compact100_correct": mc["n_correct"], "compact100_accuracy": mc["accuracy"],
                         "gap_pp": 100 * (mf["accuracy"] - mc["accuracy"]),
                         "full561_macro_f1": mf["macro_f1"], "compact100_macro_f1": mc["macro_f1"]})

    ref_row = next(r for r in per_seed if r["seed"] == ref)
    _, _, pf = _load_run(results_dir, RunSpec("full561", None, ref, "main").run_id)
    _, _, pc = _load_run(results_dir, RunSpec("compact100", primary, ref, "main").run_id)
    b = cfg["bootstrap"]
    boot = paired_participant_bootstrap(pf[0], pf[1].astype(bool), pc[1].astype(bool),
                                        replicates=b["replicates"], seed=b["seed"], percentiles=tuple(b["percentiles"]))
    boot["model_seed"] = ref
    boot["published_ci_pp"] = pub["bootstrap_ci_pp"]

    decision = classify_outcome(ref_row["full561_accuracy"], ref_row["compact100_accuracy"], pub, cfg["decision_rule"])
    decision["basis"] = f"reference seed {ref} pair (pre-declared)"
    mean_check = classify_outcome(float(np.mean([r["full561_accuracy"] for r in per_seed])),
                                  float(np.mean([r["compact100_accuracy"] for r in per_seed])),
                                  pub, cfg["decision_rule"])
    decision["supplementary_five_seed_mean_classification"] = mean_check

    summary = {
        "reference_seed": ref, "primary_variant": primary, "per_seed": per_seed,
        "five_seed": {"full561_accuracy": _stats([r["full561_accuracy"] for r in per_seed]),
                      "compact100_accuracy": _stats([r["compact100_accuracy"] for r in per_seed]),
                      "gap_pp": _stats([r["gap_pp"] for r in per_seed])},
        "published": pub,
    }
    if include_sensitivity:
        sens = {}
        for variant in cfg["mapping"]["sensitivity_variants"]:
            rows = []
            for seed in cfg["seeds"]["sensitivity"]:
                m, _, _ = _load_run(results_dir, RunSpec("compact100", variant, seed, "sensitivity").run_id)
                full = next(r for r in per_seed if r["seed"] == seed)
                rows.append({"seed": seed, "correct": m["n_correct"], "accuracy": m["accuracy"],
                             "gap_vs_full_same_seed_pp": 100 * (full["full561_accuracy"] - m["accuracy"])})
            sens[variant] = {"per_seed": rows, "accuracy": _stats([r["accuracy"] for r in rows])}
        summary["sensitivity_exploratory"] = sens

    out = results_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    (out / "bootstrap.json").write_text(json.dumps(boot, indent=1) + "\n", encoding="utf-8")
    (out / "decision.json").write_text(json.dumps(decision, indent=1) + "\n", encoding="utf-8")
    (out / "results_table.md").write_text(results_table(summary, boot, decision), encoding="utf-8")
    return {"summary": summary, "bootstrap": boot, "decision": decision}


def results_table(summary: Mapping[str, Any], boot: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    pub, ref = summary["published"], summary["reference_seed"]
    r = next(x for x in summary["per_seed"] if x["seed"] == ref)
    fs = summary["five_seed"]
    n = pub["n_test"]
    pct = lambda v: f"{100 * v:.2f}%"
    def rng(s, scale=100):
        if s["sd"] is None:   # a single seed: SD undefined
            return f"{scale * s['mean']:.2f} (n=1)"
        return f"{scale * s['mean']:.2f} ± {scale * s['sd']:.2f} ({scale * s['min']:.2f}–{scale * s['max']:.2f})"
    lines = [
        "# Results (machine-generated; do not edit)", "",
        f"Reference seed {ref}; five-seed column = mean ± SD (min–max) over seeds {[x['seed'] for x in summary['per_seed']]}.", "",
        "| Result | Paper | Reproduction (seed %d) | Correct/N | Δ vs paper | Five-seed |" % ref,
        "|---|---:|---:|---:|---:|---:|",
        f"| Full-561 accuracy | {pct(pub['full561_accuracy'])} | {pct(r['full561_accuracy'])} | {r['full561_correct']}/{n} | "
        f"{100 * (r['full561_accuracy'] - pub['full561_accuracy']):+.2f} pp | {rng(fs['full561_accuracy'])} |",
        f"| Compact-100 accuracy (X-first mapping) | {pct(pub['compact100_accuracy'])} | {pct(r['compact100_accuracy'])} | "
        f"{r['compact100_correct']}/{n} | {100 * (r['compact100_accuracy'] - pub['compact100_accuracy']):+.2f} pp | "
        f"{rng(fs['compact100_accuracy'])} |",
        f"| Full − Compact gap | {pub['gap_pp']:+.2f} pp | {r['gap_pp']:+.2f} pp | — | {r['gap_pp'] - pub['gap_pp']:+.2f} pp | "
        f"{rng(fs['gap_pp'], scale=1)} pp |",
        f"| Paired participant-bootstrap 95% CI | {pub['bootstrap_ci_pp'][0]:+.2f} to {pub['bootstrap_ci_pp'][1]:+.2f} pp | "
        f"{boot['ci_pp'][0]:+.2f} to {boot['ci_pp'][1]:+.2f} pp | — | — | — |",
        "",
        f"**Pre-registered outcome:** `{decision['category']}` ({decision['rule']}).", "",
    ]
    if "sensitivity_exploratory" in summary:
        lines += ["## Exploratory mapping sensitivity (not used to choose the primary mapping)", "",
                  "| Variant | Accuracy per seed | Mean ± SD |", "|---|---|---:|"]
        for v, s in summary["sensitivity_exploratory"].items():
            per = ", ".join(f"{x['seed']}: {pct(x['accuracy'])}" for x in s["per_seed"])
            lines.append(f"| Compact-100 {v} block | {per} | {rng(s['accuracy'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------- repeatability
def repeatability_check(results_dir: Path, cfg: Mapping[str, Any], extra_cli_args: Sequence[str] = ()) -> dict:
    """Re-run both primary models at the repeatability seed in a fresh Python process and compare bitwise."""
    seed = cfg["seeds"]["repeatability"]
    primary = cfg["mapping"]["primary_variant"]
    checks = []
    for spec in (RunSpec("full561", None, seed, "main"), RunSpec("compact100", primary, seed, "main")):
        out = results_dir / "repeatability" / spec.run_id
        cmd = [sys.executable, "-m", "har_repro.cli", "run-one", "--model", spec.model, "--seed", str(seed),
               "--out", str(out), *extra_cli_args]
        if spec.variant:
            cmd += ["--variant", spec.variant]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"repeatability subprocess failed for {spec.run_id}:\n{proc.stderr}")
        a = np.load(results_dir / "runs" / spec.run_id / "proba.npy")
        b = np.load(out / "proba.npy")
        checks.append({"run_id": spec.run_id, "fresh_process": True,
                       "probabilities_bit_identical": bool(a.dtype == b.dtype and a.shape == b.shape and
                                                           np.array_equal(a.view(np.uint32), b.view(np.uint32))),
                       "labels_identical": bool(np.array_equal(predict_labels(a), predict_labels(b))),
                       "max_abs_prob_diff": float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))})
    report = {"seed": seed, "checks": checks, "passed": all(c["probabilities_bit_identical"] and c["labels_identical"]
                                                             for c in checks)}
    (results_dir / "repeatability").mkdir(parents=True, exist_ok=True)
    (results_dir / "repeatability" / "repeatability.json").write_text(json.dumps(report, indent=1) + "\n",
                                                                       encoding="utf-8")
    return report
